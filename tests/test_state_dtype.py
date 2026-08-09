"""16-bit storage for the recurrent state pool.

**Why this is worth doing at all.** The state is the largest per-sequence term
and the one that caps the batch curve. On Qwen3.5-9B it is 48 MiB per sequence
(24 GDN layers x 32 heads x 128 x 128 x 4 B), read *and* written every decode
step, so at B=64 it is 6.14 GiB of traffic against 7.40 GiB of fp8 weights —
45% of the step. Halving it is the largest remaining lever at the batches braid
wins at, and unlike quantizing weights it costs no extra arithmetic.

**Only the storage narrows.** Both the torch path and the CUDA kernel widen the
column into fp32 on load and narrow once on store, so what a 16-bit pool costs
is one rounding per step rather than a lower-precision scan. That is the claim
this file exists to hold, and it is testable directly: a single step from an
identical starting state must land within one ulp of the storage type, and the
*drift* over many steps must stay bounded rather than compounding.

The gate is deliberately **not** token identity against the fp32 pool. A 16-bit
state rounds every step by construction, so demanding identical tokens would be
demanding that the rounding never lands on a near-tie — which is a statement
about the prompt, not about the implementation. Perplexity is where the cost is
priced (`braid/bench/perplexity.py`), the same split the FP8 work uses.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from conftest import cuda_reclaim

from braid.model.config import ModelConfig
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

NARROW = [torch.float16, torch.bfloat16]
IDS = ["fp16", "bf16"]


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def layer(cfg):
    """One GDN layer's weights — enough to exercise the scan on both paths."""
    idx = next(i for i in range(cfg.num_hidden_layers) if cfg.is_gdn(i))
    ck = load_checkpoint(MODEL_DIR, device="cuda", layers=(idx,),
                         include_embeddings=False)
    w = ck.layer(idx)
    yield w
    del ck, w
    cuda_reclaim()


def _x(cfg, B, T, seed=13, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(B, T, cfg.hidden_size, generator=g, device="cuda", dtype=dtype)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# --- the pool itself ----------------------------------------------------------

@pytest.mark.parametrize("dt", NARROW, ids=IDS)
def test_a_narrow_pool_is_half_the_bytes(cfg, dt):
    from braid.model.cache import RecurrentCache

    wide = RecurrentCache(cfg, 4, "cuda", torch.bfloat16)
    narrow = RecurrentCache(cfg, 4, "cuda", torch.bfloat16, state_dtype=dt)
    w = wide.state.numel() * wide.state.element_size()
    n = narrow.state.numel() * narrow.state.element_size()
    assert n * 2 == w, f"{n} vs {w}"
    assert narrow.state.dtype is dt


def test_an_unsupported_state_dtype_is_refused(cfg):
    """fp8 state is refuted outright upstream; it must not be reachable by
    passing a dtype that happens to be narrow."""
    from braid.model.cache import RecurrentCache

    with pytest.raises(ValueError, match="state_dtype"):
        RecurrentCache(cfg, 2, "cuda", torch.bfloat16,
                       state_dtype=torch.float8_e4m3fn)


# --- one step: storage rounding only ------------------------------------------

@pytest.mark.parametrize("dt", NARROW, ids=IDS)
@pytest.mark.parametrize("kernels", [False, True], ids=["torch", "kernels"])
def test_one_step_from_the_same_state_matches_fp32(cfg, layer, dt, kernels):
    """The core claim: with the same starting state, one step differs from the
    fp32 pool only by the *storage* rounding — not by a narrower scan.

    So the tolerance is the storage type's own epsilon, not something tuned. A
    scan that had actually dropped to 16-bit arithmetic would miss this by
    orders of magnitude, because the delta-rule reduction runs over 128 terms.
    """
    from braid.model.cache import RecurrentCache
    from braid.model.gdn import GatedDeltaNet

    B = 4
    mod = GatedDeltaNet(cfg, layer, use_kernels=kernels)
    slots = torch.arange(B, device="cuda")
    slots_i32 = slots.to(torch.int32)

    # A non-trivial starting state, shared by both arms bit-for-bit as far as
    # each pool can represent it.
    g = torch.Generator(device="cuda").manual_seed(5)
    seed_state = torch.randn(B, cfg.gdn.n_heads, cfg.gdn.state_size,
                             cfg.gdn.head_dim, generator=g, device="cuda") * 0.1

    # The CUDA conv kernel requires an fp32 window, which `allocate_cache` sets
    # from `use_kernels`; a hand-built cache has to match it. Both arms use the
    # same conv dtype, so it cannot confound the state comparison.
    conv_dt = torch.float32 if kernels else torch.bfloat16

    outs, states = {}, {}
    for name, sdt in (("fp32", torch.float32), ("narrow", dt)):
        c = RecurrentCache(cfg, B, "cuda", conv_dt, state_dtype=sdt)
        c.state.copy_(seed_state.to(sdt))
        x = _x(cfg, B, 1, dtype=torch.bfloat16)
        with torch.no_grad():
            outs[name] = mod(x, cache=c, slots=slots, slots_i32=slots_i32)
        states[name] = c.state.float().clone()

    eps = torch.finfo(dt).eps
    ry = _rel(outs["narrow"], outs["fp32"])
    rs = _rel(states["narrow"], states["fp32"])
    print(f"\n  {dt} {'kernels' if kernels else 'torch'}: "
          f"y rel_l2={ry:.3e} state rel_l2={rs:.3e} (eps {eps:.1e})")
    # Two roundings can land on the same element: the seed cast and the store.
    assert rs < 4 * eps, f"state moved {rs:.3e}, more than storage rounding"
    assert ry < 40 * eps, f"output moved {ry:.3e}, more than storage rounding"


# --- many steps: the rounding must not compound -------------------------------

@pytest.mark.parametrize("dt", NARROW, ids=IDS)
def test_drift_over_many_steps_stays_bounded(cfg, layer, dt):
    """A per-step rounding that *accumulated* would make long generations
    diverge however small each step's error is. `alpha < 1` decays old state, so
    the error should reach a floor rather than a ramp — this asserts it does, by
    requiring the residual after 64 steps to be no worse than a small multiple
    of the residual after 8.
    """
    from braid.model.cache import RecurrentCache
    from braid.model.gdn import GatedDeltaNet

    B, STEPS = 2, 64
    mod = GatedDeltaNet(cfg, layer, use_kernels=True)
    slots = torch.arange(B, device="cuda")
    slots_i32 = slots.to(torch.int32)

    caches = {n: RecurrentCache(cfg, B, "cuda", torch.float32, state_dtype=sd)
              for n, sd in (("fp32", torch.float32), ("narrow", dt))}
    at8 = None
    for t in range(STEPS):
        x = _x(cfg, B, 1, seed=100 + t, dtype=torch.bfloat16)
        with torch.no_grad():
            for c in caches.values():
                mod(x, cache=c, slots=slots, slots_i32=slots_i32)
        if t == 7:
            at8 = _rel(caches["narrow"].state, caches["fp32"].state)
    at64 = _rel(caches["narrow"].state, caches["fp32"].state)
    print(f"\n  {dt}: state drift after 8 steps {at8:.3e}, after {STEPS} {at64:.3e}")
    assert at64 < max(4 * at8, 8 * torch.finfo(dt).eps), (
        f"drift grew from {at8:.3e} to {at64:.3e} — the rounding is compounding")


# --- it must survive capture --------------------------------------------------

@pytest.mark.parametrize("dt", NARROW, ids=IDS)
def test_a_narrow_pool_replays_bit_identically_under_capture(dt):
    """The kernel dispatches on the pool's dtype. A graph captured over the
    wrong instantiation would still run."""
    import gc

    from braid.model.engine import Engine
    from braid.model.graph import GraphedDecoder

    B = 4
    ck = load_checkpoint(MODEL_DIR, device="cuda")
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16,
                                 use_kernels=True, state_dtype=dt)
    del ck
    cache = dec = None
    try:
        assert eng.allocate_cache(16, max_slots=2).layers[0].state.dtype is dt
        cache = eng.allocate_cache(96, max_slots=B + 1)
        g = torch.Generator(device="cuda").manual_seed(5)
        for slot in range(B + 1):
            cache.reset_slot(slot)
        for row in range(B):
            eng.forward(torch.randint(0, 1000, (1, 6 + row), generator=g,
                                      device="cuda"), cache.select([row]))

        snap = cache.snapshot()
        tokens = torch.arange(100, 100 + B, device="cuda")[:, None]
        slots = torch.arange(B, device="cuda")

        eager = eng.decode_step(tokens, cache.select(slots.tolist())).clone()
        cache.restore(snap)
        dec = GraphedDecoder(eng, cache, buckets=(B,))
        cache.restore(snap)
        replay = dec.step(tokens, slots)
        assert torch.equal(eager, replay), f"{dt}: replay differs from eager"
    finally:
        del eng, cache, dec
        gc.collect()
        torch.cuda.empty_cache()
