"""In-kernel gate computation: alpha and beta from the raw projections.

**Why these variants exist.** At B=1 the decode step is 91% device-busy, so the
remaining cost is kernels, not gaps — and 24.7% of the busy time is elementwise
work over 1,873 launches. Roughly 8 of those launches per GDN layer per step do
nothing but compute

    beta  = sigmoid(b_raw)                    # in the ACTIVATION dtype, then widened
    alpha = exp(A * softplus(a_raw + dt_bias))  # fp32 throughout

`gdn_decode_raw` / `gdn_prefill_raw` compute that inside the scan kernel — a
few transcendentals per thread against a 2x128-FMA recurrence — and the
[B, T, H] fp32 gate tensors stop existing. In prefill, `seq_lens` also replaces
the caller's `torch.where` pad mask.

**The gate here is bit-identity, and it is a fair demand.** The kernel uses the
same CUDA math-library primitives torch's own elementwise kernels lower to
(`expf`, `log1pf`, IEEE division; this extension builds without fast math), the
same softplus threshold-20 branch, and the same round-through-activation-dtype
for beta's sigmoid. So the raw variants must equal "torch computes the gates,
the precomputed kernel consumes them" to the bit — anything less means the
arithmetic silently changed, which is exactly what greedy decoding amplifies.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from braid.model.config import ModelConfig

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

ACT_DTYPES = [torch.float32, torch.bfloat16]
ACT_IDS = ["fp32-act", "bf16-act"]


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def mod():
    from braid.kernels.loader import load_gdn

    return load_gdn()


def _qkv(cfg, B, T, seed=3):
    g = cfg.gdn
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=gen, device="cuda",
                           dtype=torch.float32).contiguous()

    if T is None:  # decode shapes
        return r(B, g.n_groups, g.state_size), r(B, g.n_groups, g.state_size), \
            r(B, g.n_heads, g.head_dim)
    return r(B, T, g.n_groups, g.state_size), r(B, T, g.n_groups, g.state_size), \
        r(B, T, g.n_heads, g.head_dim)


def _raws(cfg, B, T, dt, seed=7):
    """Raw projections plus the layer constants, shaped as gdn.py holds them."""
    g = cfg.gdn
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (B, g.n_heads) if T is None else (B, T, g.n_heads)
    a_raw = (torch.randn(*shape, generator=gen, device="cuda") * 2).to(dt).contiguous()
    b_raw = (torch.randn(*shape, generator=gen, device="cuda") * 2).to(dt).contiguous()
    # A is already -exp(A_log): strictly negative, so alpha lands in (0, 1).
    A = -torch.exp(torch.randn(g.n_heads, generator=gen, device="cuda") * 0.5)
    dt_bias = torch.randn(g.n_heads, generator=gen, device="cuda") * 0.5
    return a_raw, b_raw, A.contiguous(), dt_bias.contiguous()


def _torch_gates(a_raw, b_raw, A, dt_bias):
    """Exactly gdn.py's spelling — sigmoid in the activation dtype then widened,
    alpha fp32 throughout. This is the reference the kernel must hit to the bit."""
    beta = torch.sigmoid(b_raw).float()
    alpha = torch.exp(A * F.softplus(a_raw.float() + dt_bias))
    return alpha.contiguous(), beta.contiguous()


def _pool(cfg, B, dtype=torch.float32, seed=11):
    g = cfg.gdn
    gen = torch.Generator(device="cuda").manual_seed(seed)
    p = torch.randn(B, g.n_heads, g.state_size, g.head_dim, generator=gen,
                    device="cuda") * 0.1
    return p.to(dtype).contiguous()


# --- decode -------------------------------------------------------------------

@pytest.mark.parametrize("dt", ACT_DTYPES, ids=ACT_IDS)
@pytest.mark.parametrize("B", [1, 5])
def test_decode_raw_gates_match_the_torch_computed_ones(cfg, mod, B, dt):
    g = cfg.gdn
    q, k, v = _qkv(cfg, B, None)
    a_raw, b_raw, A, dt_bias = _raws(cfg, B, None, dt)
    alpha, beta = _torch_gates(a_raw, b_raw, A, dt_bias)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    p1 = _pool(cfg, B)
    y1 = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_decode(p1, slots, q, k, v, alpha, beta, y1)

    p2 = _pool(cfg, B)
    y2 = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_decode_raw(p2, slots, q, k, v, a_raw, b_raw, A, dt_bias, y2)

    assert torch.equal(y1, y2), (
        f"B={B} {dt}: outputs differ by {(y1 - y2).abs().max().item():.3e}")
    assert torch.equal(p1, p2), (
        f"B={B} {dt}: states differ by {(p1 - p2).abs().max().item():.3e}")


@pytest.mark.parametrize("pool_dt", [torch.float16, torch.bfloat16],
                         ids=["fp16-pool", "bf16-pool"])
def test_a_narrow_pool_dispatches_the_raw_variant_too(cfg, mod, pool_dt):
    """The dispatch is (pool dtype) x (activation dtype); this exercises the
    nesting the fp32/fp32 test cannot."""
    g = cfg.gdn
    B, dt = 3, torch.bfloat16
    q, k, v = _qkv(cfg, B, None, seed=9)
    a_raw, b_raw, A, dt_bias = _raws(cfg, B, None, dt, seed=13)
    alpha, beta = _torch_gates(a_raw, b_raw, A, dt_bias)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    p1 = _pool(cfg, B, dtype=pool_dt)
    y1 = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_decode(p1, slots, q, k, v, alpha, beta, y1)

    p2 = _pool(cfg, B, dtype=pool_dt)
    y2 = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_decode_raw(p2, slots, q, k, v, a_raw, b_raw, A, dt_bias, y2)

    assert torch.equal(y1, y2) and torch.equal(p1, p2)


def test_the_softplus_threshold_branch_matches_torch(cfg, mod):
    """torch's softplus is `x > 20 ? x : log1p(exp(x))`. The branch point is a
    real discontinuity in *implementation* (not in value), so the kernel must
    take the same branch at the same inputs — a kernel that used the smooth
    form everywhere would differ in the last bits exactly where exp overflows
    fp32 headroom. Values straddle the threshold on both sides."""
    g = cfg.gdn
    B = 1
    q, k, v = _qkv(cfg, B, None, seed=21)
    _, b_raw, A, _ = _raws(cfg, B, None, torch.float32, seed=23)
    dt_bias = torch.zeros(g.n_heads, device="cuda")

    a_raw = torch.zeros(B, g.n_heads, device="cuda")
    probes = [-80.0, -20.0, 0.0, 19.0, 19.999999, 20.0, 20.000001, 25.0, 80.0]
    a_raw[0, : len(probes)] = torch.tensor(probes, device="cuda")

    alpha, beta = _torch_gates(a_raw, b_raw, A, dt_bias)
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")

    p1 = _pool(cfg, B)
    y1 = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_decode(p1, slots, q, k, v, alpha, beta, y1)

    p2 = _pool(cfg, B)
    y2 = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_decode_raw(p2, slots, q, k, v, a_raw, b_raw, A, dt_bias, y2)

    assert torch.equal(y1, y2) and torch.equal(p1, p2), (
        "gates differ across the softplus threshold")


# --- prefill ------------------------------------------------------------------

@pytest.mark.parametrize("dt", ACT_DTYPES, ids=ACT_IDS)
@pytest.mark.parametrize("T", [1, 17, 96])
def test_prefill_raw_gates_match_the_torch_computed_ones(cfg, mod, T, dt):
    g = cfg.gdn
    B = 3
    q, k, v = _qkv(cfg, B, T, seed=31)
    a_raw, b_raw, A, dt_bias = _raws(cfg, B, T, dt, seed=33)
    alpha, beta = _torch_gates(a_raw, b_raw, A, dt_bias)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    p1 = _pool(cfg, B)
    y1 = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_prefill(p1, slots, q, k, v, alpha, beta, y1)

    p2 = _pool(cfg, B)
    y2 = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_prefill_raw(p2, slots, q, k, v, a_raw, b_raw, A, dt_bias, y2, None)

    assert torch.equal(y1, y2), f"T={T} {dt}: outputs differ"
    assert torch.equal(p1, p2), f"T={T} {dt}: final states differ"


def test_seq_lens_is_the_where_mask_bit_for_bit(cfg, mod):
    """The ragged contract, moved in-kernel: `seq_lens` must produce the state a
    caller-side `torch.where(live, gates, identity)` produced. Same comparison,
    same constants, so the states — and even the garbage pad outputs — must be
    bit-identical between the two spellings."""
    g = cfg.gdn
    B, T = 4, 12
    q, k, v = _qkv(cfg, B, T, seed=41)
    a_raw, b_raw, A, dt_bias = _raws(cfg, B, T, torch.bfloat16, seed=43)
    alpha, beta = _torch_gates(a_raw, b_raw, A, dt_bias)

    lens = torch.tensor([3, 7, 12, 1], device="cuda", dtype=torch.int32)
    live = (torch.arange(T, device="cuda")[None] < lens[:, None])[:, :, None]
    alpha_m = torch.where(live, alpha, 1.0).contiguous()
    beta_m = torch.where(live, beta, 0.0).contiguous()
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    p1 = _pool(cfg, B)
    y1 = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_prefill(p1, slots, q, k, v, alpha_m, beta_m, y1)

    p2 = _pool(cfg, B)
    y2 = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_prefill_raw(p2, slots, q, k, v, a_raw, b_raw, A, dt_bias, y2, lens)

    assert torch.equal(p1, p2), "seq_lens moved the state differently than the mask"
    assert torch.equal(y1, y2), "seq_lens produced different outputs than the mask"


def test_a_len_of_zero_and_a_len_beyond_t_are_legal(cfg, mod):
    """len=0 means every column is a pad (the row's state must not move at all);
    len > T means every column is live (same as passing None). Neither is an
    error — schedulers produce both shapes at chunk boundaries."""
    g = cfg.gdn
    B, T = 2, 6
    q, k, v = _qkv(cfg, B, T, seed=51)
    a_raw, b_raw, A, dt_bias = _raws(cfg, B, T, torch.float32, seed=53)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    lens = torch.tensor([0, T + 5], device="cuda", dtype=torch.int32)
    p = _pool(cfg, B)
    before = p[0].clone()
    y = torch.empty(B, T, g.n_heads, g.head_dim, device="cuda")
    mod.gdn_prefill_raw(p, slots, q, k, v, a_raw, b_raw, A, dt_bias, y, lens)
    assert torch.equal(p[0], before), "len=0 row's state moved"

    # len > T must equal the unmasked run for that row.
    p_ref = _pool(cfg, B)
    y_ref = torch.empty_like(y)
    mod.gdn_prefill_raw(p_ref, slots, q, k, v, a_raw, b_raw, A, dt_bias, y_ref, None)
    assert torch.equal(p[1], p_ref[1]), "len>T row differs from the unmasked run"
    assert torch.equal(y[1], y_ref[1])


# --- the layer, with the flag flipped ------------------------------------------

def test_the_layer_is_bit_identical_with_gates_in_kernel(cfg):
    """`raw_gates` must be a launch-count switch, never a numerics one: the
    same layer, same weights, same cache seed, flag on vs off — decode and
    chunk prefill both — must produce the same bits. This is the wiring-level
    closure of the kernel-level gates above: it catches a mis-passed tensor
    (a_raw for b_raw, a stale slice) that the synthetic tests cannot."""
    from conftest import cuda_reclaim

    from braid.model.cache import RecurrentCache
    from braid.model.gdn import GatedDeltaNet
    from braid.model.loader import load_checkpoint

    idx = next(i for i in range(cfg.num_hidden_layers) if cfg.is_gdn(i))
    ck = load_checkpoint(MODEL_DIR, device="cuda", layers=(idx,),
                         include_embeddings=False)
    w = ck.layer(idx)
    try:
        gen = torch.Generator(device="cuda").manual_seed(4)
        slots = torch.arange(3, device="cuda")
        slots_i32 = slots.to(torch.int32)
        for label, T, lens in (("decode", 1, None), ("prefill", 24, None),
                               ("ragged", 24, torch.tensor([5, 24, 11],
                                                           device="cuda"))):
            x = torch.randn(3, T, cfg.hidden_size, generator=gen, device="cuda",
                            dtype=torch.bfloat16)
            outs, states = {}, {}
            for raw in (True, False):
                c = RecurrentCache(cfg, 3, "cuda", torch.float32)
                m = GatedDeltaNet(cfg, w, use_kernels=True)
                m.raw_gates = raw
                with torch.no_grad():
                    outs[raw] = m(x, cache=c, slots=slots, slots_i32=slots_i32,
                                  seq_lens=lens)
                states[raw] = c.state.clone()
            assert torch.equal(outs[True], outs[False]), (
                f"{label}: raw_gates flipped the layer output by "
                f"{(outs[True].float() - outs[False].float()).abs().max():.3e}")
            assert torch.equal(states[True], states[False]), (
                f"{label}: raw_gates flipped the recurrent state")
    finally:
        del ck, w
        cuda_reclaim()
