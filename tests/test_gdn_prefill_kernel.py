"""The chunk scan kernel: T tokens per launch instead of T launches.

`gdn_prefill` exists for one reason — braid's prefill ran the one-token step in
a Python loop over the sequence axis, one iteration per *column*, each a handful
of small kernels. Batching the rows (`seq_lens`) divided that fixed cost across
B sequences and was worth 6.96x; it did not remove it, and prefill is still 70%
of the served wall clock at c=64. This kernel keeps the state in registers for
the whole chunk so a `[B, T]` forward costs one launch per layer.

**The arithmetic is deliberately unchanged**, line for line with
`gdn_decode_kernel`. So the gates here are not "is the answer plausible" but:

  1. **It equals the one-token kernel applied T times, bit for bit.** That is
     the strongest statement available and the one that matters: prefill and
     decode must remain the same function, or generation drifts after the
     prompt. A chunkwise reformulation could not pass this, which is exactly
     why this is a chunk-*cached scalar* loop and not a WY/UT variant.
  2. **A pad column is an exact identity on the state.** Ragged batches feed
     `alpha = 1, beta = 0` for padding, and `S' = 1*S + k (x) 0` has to be `S`
     bit-for-bit, not approximately.
  3. **Chunking is invisible.** Feeding 2 x T/2 must leave the same state as
     one T, which is what a scheduler's chunked prefill relies on.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from conftest import cuda_reclaim

from braid.model.cache import RecurrentCache
from braid.model.config import ModelConfig
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def mod():
    from braid.kernels.loader import load_gdn

    return load_gdn()


def _inputs(cfg, B, T, seed=3):
    g = cfg.gdn
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda",
                           dtype=torch.float32).contiguous() * scale

    q = r(B, T, g.n_groups, g.state_size)
    k = r(B, T, g.n_groups, g.state_size)
    v = r(B, T, g.n_heads, g.head_dim)
    # alpha in (0, 1) as exp(A * softplus(.)) always is; beta in (0, 1).
    alpha = torch.rand(B, T, g.n_heads, generator=gen, device="cuda").contiguous()
    beta = torch.rand(B, T, g.n_heads, generator=gen, device="cuda").contiguous()
    return q, k, v, alpha, beta


def _pool(cfg, B, dtype=torch.float32, seed=11):
    g = cfg.gdn
    gen = torch.Generator(device="cuda").manual_seed(seed)
    p = torch.randn(B, g.n_heads, g.state_size, g.head_dim, generator=gen,
                    device="cuda") * 0.1
    return p.to(dtype).contiguous()


# --- gate 1: the chunk equals the one-token kernel, T times -------------------

@pytest.mark.parametrize("T", [1, 4, 17, 128])
@pytest.mark.parametrize("B", [1, 5])
def test_chunk_equals_the_decode_kernel_applied_t_times(cfg, mod, T, B):
    """Bit-identical, not merely close. Both kernels run the same instructions
    in the same order on the same values; the only difference is where the state
    lives between tokens. Anything less than equality means it does not."""
    g = cfg.gdn
    q, k, v, alpha, beta = _inputs(cfg, B, T)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    chunk_pool = _pool(cfg, B)
    y_chunk = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_prefill(chunk_pool, slots, q, k, v, alpha, beta, y_chunk)

    step_pool = _pool(cfg, B)
    y_step = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    for t in range(T):
        out = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
        mod.gdn_decode(step_pool, slots, q[:, t].contiguous(), k[:, t].contiguous(),
                       v[:, t].contiguous(), alpha[:, t].contiguous(),
                       beta[:, t].contiguous(), out)
        y_step[:, t] = out

    assert torch.equal(y_chunk, y_step), (
        f"B={B} T={T}: outputs differ by "
        f"{(y_chunk - y_step).abs().max().item():.3e}")
    assert torch.equal(chunk_pool, step_pool), (
        f"B={B} T={T}: final state differs by "
        f"{(chunk_pool - step_pool).abs().max().item():.3e}")


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_a_narrow_state_pool_still_matches_the_decode_kernel(cfg, mod, dt):
    """The chunk kernel dispatches on pool dtype the same way decode does. It
    loads once and stores once per chunk where decode does it per token, so the
    two agree only if neither is rounding in between — which is the claim."""
    g = cfg.gdn
    B, T = 3, 8
    q, k, v, alpha, beta = _inputs(cfg, B, T, seed=9)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    chunk_pool = _pool(cfg, B, dtype=dt)
    y_chunk = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_prefill(chunk_pool, slots, q, k, v, alpha, beta, y_chunk)

    step_pool = _pool(cfg, B, dtype=dt)
    for t in range(T):
        out = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
        mod.gdn_decode(step_pool, slots, q[:, t].contiguous(), k[:, t].contiguous(),
                       v[:, t].contiguous(), alpha[:, t].contiguous(),
                       beta[:, t].contiguous(), out)

    # Not bit-identical here and it should not be: the per-token arm rounds to
    # `dt` T times, the chunk arm once. Both are correct; the chunk arm is the
    # more accurate of the two, which is the direction to be off in.
    rel = ((chunk_pool.float() - step_pool.float()).norm()
           / step_pool.float().norm()).item()
    print(f"\n  {dt}: chunk vs per-token state rel_l2 {rel:.3e} "
          f"(eps {torch.finfo(dt).eps:.1e})")
    assert rel < 8 * torch.finfo(dt).eps


# --- gate 2: padding is an exact identity -------------------------------------

def test_a_pad_column_does_not_move_the_state(cfg, mod):
    """`alpha = 1, beta = 0` must leave the state bit-identical. Ragged prefill
    is built on this: a padded row's state has to be the object it would have
    been had the row not been in the batch."""
    g = cfg.gdn
    B, T = 4, 6
    q, k, v, alpha, beta = _inputs(cfg, B, T, seed=21)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    # Row b owns the first (b + 2) columns; the rest are padding.
    lens = torch.tensor([2, 3, 4, 6], device="cuda")
    live = (torch.arange(T, device="cuda")[None] < lens[:, None])[:, :, None]
    alpha_m = torch.where(live, alpha, 1.0).contiguous()
    beta_m = torch.where(live, beta, 0.0).contiguous()

    padded = _pool(cfg, B)
    y = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_prefill(padded, slots, q, k, v, alpha_m, beta_m, y)

    # Each row run alone, to its own true length, in its own single-slot pool.
    for b in range(B):
        n = int(lens[b])
        solo = _pool(cfg, B)[b:b + 1].contiguous()
        y1 = torch.empty(1, n, g.n_heads, g.head_dim, device="cuda")
        mod.gdn_prefill(solo, torch.zeros(1, dtype=torch.int32, device="cuda"),
                        q[b:b + 1, :n].contiguous(), k[b:b + 1, :n].contiguous(),
                        v[b:b + 1, :n].contiguous(),
                        alpha_m[b:b + 1, :n].contiguous(),
                        beta_m[b:b + 1, :n].contiguous(), y1)
        assert torch.equal(padded[b], solo[0]), (
            f"row {b} (len {n}) state moved under {T - n} pad columns")
        assert torch.equal(y[b, :n], y1[0]), f"row {b} outputs moved"


# --- gate 3: chunking is invisible --------------------------------------------

@pytest.mark.parametrize("cut", [1, 7, 64])
def test_two_chunks_leave_the_same_state_as_one(cfg, mod, cut):
    g = cfg.gdn
    B, T = 2, 96
    q, k, v, alpha, beta = _inputs(cfg, B, T, seed=33)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    whole = _pool(cfg, B)
    y_whole = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_prefill(whole, slots, q, k, v, alpha, beta, y_whole)

    parts = _pool(cfg, B)
    y_parts = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    for lo, hi in ((0, cut), (cut, T)):
        out = torch.empty(B, hi - lo, g.n_heads, g.head_dim, device="cuda")
        mod.gdn_prefill(parts, slots, q[:, lo:hi].contiguous(),
                        k[:, lo:hi].contiguous(), v[:, lo:hi].contiguous(),
                        alpha[:, lo:hi].contiguous(), beta[:, lo:hi].contiguous(),
                        out)
        y_parts[:, lo:hi] = out

    assert torch.equal(whole, parts), f"cut at {cut} moved the state"
    assert torch.equal(y_whole, y_parts), f"cut at {cut} moved the outputs"


# --- the layer, end to end ----------------------------------------------------

def test_the_layer_matches_its_own_torch_loop(cfg):
    """`GatedDeltaNet(use_kernels=True)` must agree with the torch loop it
    replaces. Not bit-identical — the kernel l2-normalises in fp32 where the
    torch path follows HF and normalises in the activation dtype, the same
    documented deviation the decode kernel already has."""
    from braid.model.gdn import GatedDeltaNet

    idx = next(i for i in range(cfg.num_hidden_layers) if cfg.is_gdn(i))
    ck = load_checkpoint(MODEL_DIR, device="cuda", layers=(idx,),
                         include_embeddings=False)
    w = ck.layer(idx)
    try:
        B, T = 3, 40
        gen = torch.Generator(device="cuda").manual_seed(4)
        x = torch.randn(B, T, cfg.hidden_size, generator=gen, device="cuda",
                        dtype=torch.bfloat16)
        slots = torch.arange(B, device="cuda")

        outs = {}
        for name, kern in (("torch", False), ("kernel", True)):
            c = RecurrentCache(cfg, B, "cuda",
                               torch.float32 if kern else torch.bfloat16)
            m = GatedDeltaNet(cfg, w, use_kernels=kern)
            with torch.no_grad():
                outs[name] = m(x, cache=c, slots=slots,
                               slots_i32=slots.to(torch.int32))

        rel = ((outs["kernel"].float() - outs["torch"].float()).norm()
               / outs["torch"].float().norm()).item()
        print(f"\n  layer, chunk kernel vs torch loop: rel_l2 {rel:.3e}")
        assert rel < 2e-2, f"chunk kernel differs from the torch loop by {rel:.3e}"
    finally:
        del ck, w
        cuda_reclaim()
