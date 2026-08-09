"""The copy kill: strided q/k/v into the scan, bf16 into the conv.

**What was killed, and why the gates are bit-identity.** Per decode step, the
kernel path spent 24 launches casting the conv input to fp32 and (at B>1) 72
launches copying q/k/v out of the conv output to satisfy a contiguity check.
Neither changes a single value: bf16 -> fp32 widening is exact, and a strided
read visits the same elements a copied tensor would hold. So the gates here
demand bit-for-bit equality between the old spelling (cast + copy) and the new
one (widen in-kernel + strided views) — a tolerance would mean the addressing
is wrong, not that the arithmetic is coarse.

The layer-level test at the bottom is the one that would catch a mis-derived
stride: a column slice whose row stride is the FULL conv width, fed to a kernel
that assumed dense rows, reads row 0's tail as row 1's head — plausible values,
wrong sequence, exactly the silent failure class the kernel notes warn about.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from braid.model.config import ModelConfig

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


# --- the conv: bf16 in == fp32-cast in, bit for bit ---------------------------

@pytest.mark.parametrize("B", [1, 6])
def test_bf16_conv_input_matches_the_cast(cfg, mod, B):
    g = cfg.gdn
    C, K = g.conv_channels, g.conv_kernel
    gen = torch.Generator(device="cuda").manual_seed(3)
    x16 = torch.randn(B, C, generator=gen, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(C, K, generator=gen, device="cuda") * 0.3
    bias = torch.randn(C, generator=gen, device="cuda") * 0.1
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    pool1 = torch.randn(B, C, K, generator=gen, device="cuda") * 0.2
    pool2 = pool1.clone()
    out1 = torch.empty(B, C, device="cuda")
    out2 = torch.empty(B, C, device="cuda")

    mod.conv1d_decode(pool1, slots, x16.float().contiguous(), w, bias, out1)
    mod.conv1d_decode(pool2, slots, x16.contiguous(), w, bias, out2)

    assert torch.equal(out1, out2), "bf16 input rounds differently than the cast"
    assert torch.equal(pool1, pool2), "the saved window differs"


# --- the scan: strided q/k/v == contiguous copies, bit for bit ----------------

def _strided_qkv(cfg, B, seed=7):
    """q/k/v as column slices of one [B, C] buffer — the conv output's shape —
    so their row stride is C, not their own inner size."""
    g = cfg.gdn
    gen = torch.Generator(device="cuda").manual_seed(seed)
    C = g.conv_channels
    buf = torch.randn(B, C, generator=gen, device="cuda")
    key = g.n_groups * g.state_size
    q, k, v = torch.split(buf, [key, key, g.inner_size], dim=-1)
    q = q.unflatten(-1, (g.n_groups, g.state_size))
    k = k.unflatten(-1, (g.n_groups, g.state_size))
    v = v.unflatten(-1, (g.n_heads, g.head_dim))
    if B > 1:
        assert not q.is_contiguous(), "fixture must actually be strided"
    return q, k, v


def _pool(cfg, B, seed=11):
    g = cfg.gdn
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(B, g.n_heads, g.state_size, g.head_dim, generator=gen,
                        device="cuda") * 0.1).contiguous()


@pytest.mark.parametrize("B", [1, 5])
def test_strided_qkv_matches_contiguous_in_gdn_decode(cfg, mod, B):
    g = cfg.gdn
    q, k, v = _strided_qkv(cfg, B)
    gen = torch.Generator(device="cuda").manual_seed(9)
    alpha = torch.rand(B, g.n_heads, generator=gen, device="cuda").contiguous()
    beta = torch.rand(B, g.n_heads, generator=gen, device="cuda").contiguous()
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    p1, p2 = _pool(cfg, B), _pool(cfg, B)
    y1 = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
    y2 = torch.empty_like(y1)
    mod.gdn_decode(p1, slots, q.contiguous(), k.contiguous(), v.contiguous(),
                   alpha, beta, y1)
    mod.gdn_decode(p2, slots, q, k, v, alpha, beta, y2)

    assert torch.equal(y1, y2), (
        f"B={B}: strided read differs from the copy by "
        f"{(y1 - y2).abs().max().item():.3e}")
    assert torch.equal(p1, p2)


def test_strided_qkv_matches_contiguous_in_gdn_decode_raw(cfg, mod):
    g = cfg.gdn
    B = 4
    q, k, v = _strided_qkv(cfg, B, seed=21)
    gen = torch.Generator(device="cuda").manual_seed(23)
    a_raw = (torch.randn(B, g.n_heads, generator=gen, device="cuda") * 2
             ).to(torch.bfloat16).contiguous()
    b_raw = (torch.randn(B, g.n_heads, generator=gen, device="cuda") * 2
             ).to(torch.bfloat16).contiguous()
    A = -torch.exp(torch.randn(g.n_heads, generator=gen, device="cuda") * 0.5)
    dt_bias = (torch.randn(g.n_heads, generator=gen, device="cuda") * 0.5).contiguous()
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    p1, p2 = _pool(cfg, B), _pool(cfg, B)
    y1 = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
    y2 = torch.empty_like(y1)
    mod.gdn_decode_raw(p1, slots, q.contiguous(), k.contiguous(), v.contiguous(),
                       a_raw, b_raw, A.contiguous(), dt_bias, y1)
    mod.gdn_decode_raw(p2, slots, q, k, v, a_raw, b_raw, A.contiguous(), dt_bias, y2)

    assert torch.equal(y1, y2) and torch.equal(p1, p2)


def test_an_overlapping_stride_is_refused(cfg, mod):
    """stride(0) below the row's own extent means rows alias — the kernel would
    read row 0's tail as row 1's head and produce plausible garbage. That must
    be a loud error, not an answer."""
    g = cfg.gdn
    B = 3
    gen = torch.Generator(device="cuda").manual_seed(31)
    base = torch.randn(B * 64, g.n_groups * g.state_size, generator=gen,
                       device="cuda")
    bad = base.as_strided((B, g.n_groups, g.state_size),
                          (g.state_size, g.state_size, 1))
    k = torch.randn(B, g.n_groups, g.state_size, generator=gen,
                    device="cuda").contiguous()
    v = torch.randn(B, g.n_heads, g.head_dim, generator=gen,
                    device="cuda").contiguous()
    alpha = torch.rand(B, g.n_heads, generator=gen, device="cuda").contiguous()
    beta = torch.rand(B, g.n_heads, generator=gen, device="cuda").contiguous()
    slots = torch.arange(B, dtype=torch.int32, device="cuda")
    y = torch.empty(B, g.n_heads, g.head_dim, device="cuda")
    with pytest.raises(RuntimeError, match="overlap"):
        mod.gdn_decode(_pool(cfg, B), slots, bad, k, v, alpha, beta, y)


# --- the layer, end to end ----------------------------------------------------

def test_the_kernel_layer_still_tracks_the_torch_path(cfg):
    """The decode path now feeds the scan straight from conv-output views. A
    mis-derived stride would not crash — it would read the wrong rows and stay
    fluent. The torch path never had copies to kill, so agreement with it at
    the layer's established tolerance closes exactly that hole."""
    from conftest import cuda_reclaim

    from braid.model.cache import RecurrentCache
    from braid.model.gdn import GatedDeltaNet
    from braid.model.loader import load_checkpoint

    idx = next(i for i in range(cfg.num_hidden_layers) if cfg.is_gdn(i))
    ck = load_checkpoint(MODEL_DIR, device="cuda", layers=(idx,),
                         include_embeddings=False)
    w = ck.layer(idx)
    try:
        B = 5
        gen = torch.Generator(device="cuda").manual_seed(4)
        x = torch.randn(B, 1, cfg.hidden_size, generator=gen, device="cuda",
                        dtype=torch.bfloat16)
        slots = torch.arange(B, device="cuda")
        seed = torch.randn(B, cfg.gdn.n_heads, cfg.gdn.state_size,
                           cfg.gdn.head_dim, generator=gen, device="cuda") * 0.1

        outs = {}
        for name, kern in (("torch", False), ("kernel", True)):
            c = RecurrentCache(cfg, B, "cuda",
                               torch.float32 if kern else torch.bfloat16)
            c.state.copy_(seed)
            m = GatedDeltaNet(cfg, w, use_kernels=kern)
            with torch.no_grad():
                outs[name] = m(x, cache=c, slots=slots,
                               slots_i32=slots.to(torch.int32))

        rel = ((outs["kernel"].float() - outs["torch"].float()).norm()
               / outs["torch"].float().norm()).item()
        print(f"\n  layer decode, views vs torch path: rel_l2 {rel:.3e}")
        assert rel < 2e-2, f"kernel layer diverged from torch by {rel:.3e}"
    finally:
        del ck, w
        cuda_reclaim()
