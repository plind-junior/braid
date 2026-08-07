"""FP8 W8A8 on the MLP projections.

The interesting thing about this path is that its gate cannot be a tolerance.
e4m3 has 3 mantissa bits, so a single GEMM lands ~3.7e-2 from bf16 — roughly
10x braid's 5e-3 parity gate, and no correct implementation does better. A
tolerance tight enough to catch a bug would fail the working code, and one loose
enough to pass would catch nothing.

So the gates here are structural (does the quantized weight actually round-trip,
does the scale sit where `_scaled_mm` expects it, does the shape guard fire) and
the *numeric* gate lives in `braid/bench/perplexity.py`, where +0.50% on the
whole corpus is a claim worth making. That split is deliberate.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from braid.model.quant import (FP8, FP8_MAX, FP8Weight, fp8_linear, linear,
                               maybe_quantize, quantize_act, quantize_weight)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _w(N=512, K=256, scale=0.02):
    g = torch.Generator(device="cuda").manual_seed(7)
    return torch.randn(N, K, generator=g, device="cuda", dtype=torch.bfloat16) * scale


# --- the weight itself --------------------------------------------------------

def test_quantized_weight_is_half_the_bytes():
    w = _w()
    q = quantize_weight(w)
    assert q.data.dtype is FP8
    assert q.nbytes < w.numel() * 2 / 1.9, "fp8 should be ~half of bf16"


def test_dequantizing_recovers_the_weight_to_fp8_precision():
    """3 mantissa bits is ~6% per element; the tensor-relative error is ~3.6e-2."""
    w = _w()
    q = quantize_weight(w)
    back = q.data.float() * q.scale
    rel = ((back - w.float()).norm() / w.float().norm()).item()
    assert rel < 6e-2, f"round-trip {rel:.3e} is worse than e4m3 allows"
    assert rel > 1e-3, f"round-trip {rel:.3e} is too good — is it really fp8?"


def test_the_scale_is_a_device_tensor_not_a_python_float():
    """A Python float would be re-boxed per call, and an allocation inside a
    captured graph is a capture failure."""
    q = quantize_weight(_w())
    assert isinstance(q.scale, torch.Tensor) and q.scale.is_cuda
    assert q.scale.shape == (1, 1) and q.scale.dtype is torch.float32


def test_an_all_zero_weight_does_not_produce_nans():
    q = quantize_weight(torch.zeros(64, 32, device="cuda", dtype=torch.bfloat16))
    assert torch.isfinite(q.data.float()).all() and (q.scale > 0).all()


@pytest.mark.parametrize("shape", [(30, 256), (512, 250)])
def test_shapes_scaled_mm_cannot_take_are_refused(shape):
    w = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiples of 16"):
        quantize_weight(w)
    # ...and `maybe_quantize` declines rather than raising, because it sweeps a
    # whole model whose projection widths differ.
    assert maybe_quantize(w, True) is w


def test_maybe_quantize_is_a_no_op_when_disabled():
    w = _w()
    assert maybe_quantize(w, False) is w
    assert isinstance(maybe_quantize(w, True), FP8Weight)


# --- the GEMM -----------------------------------------------------------------

def test_fp8_linear_agrees_with_bf16_to_fp8_precision():
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    w = _w()
    got = fp8_linear(x, quantize_weight(w))
    ref = F.linear(x, w)
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 8e-2, f"rel {rel:.3e} — worse than e4m3 quantization explains"
    cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.99, f"cosine {cos:.6f} — the product is wrong, not just coarse"


def test_leading_dims_survive():
    x = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    out = fp8_linear(x, quantize_weight(_w()))
    assert out.shape == (2, 3, 512) and out.dtype is torch.bfloat16


def test_linear_dispatches_on_weight_type():
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    w = _w()
    torch.testing.assert_close(linear(x, w), F.linear(x, w), rtol=0, atol=0)
    assert linear(x, quantize_weight(w)).shape == (4, 512)


def test_activation_quantization_uses_the_full_fp8_range():
    """A scale that leaves headroom is throwing away mantissa bits for nothing."""
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    xq, sx = quantize_act(x)
    assert xq.dtype is FP8 and sx.shape == (1, 1)
    peak = xq.float().abs().max().item()
    assert peak > FP8_MAX * 0.5, f"peak {peak} — the scale wastes range"


# --- end to end ---------------------------------------------------------------

def test_quantized_mlp_tracks_the_bf16_one():
    from types import SimpleNamespace

    from braid.model.mlp import MLP

    # A stub, not a real ModelConfig: MLP reads exactly one field, and building
    # a whole config here would couple this test to rope validation it does not
    # exercise. `test_full_forward` covers the real config.
    cfg = SimpleNamespace(hidden_act="silu")
    g = torch.Generator(device="cuda").manual_seed(3)
    r = lambda N, K: torch.randn(N, K, generator=g, device="cuda",
                                 dtype=torch.bfloat16) * 0.05
    w = {"mlp.gate_proj": r(512, 256), "mlp.up_proj": r(512, 256),
         "mlp.down_proj": r(256, 512)}

    x = torch.randn(2, 1, 256, device="cuda", dtype=torch.bfloat16)
    ref = MLP(cfg, w, quant=False)(x)
    got = MLP(cfg, w, quant=True)(x)

    assert got.shape == ref.shape and got.dtype is ref.dtype
    cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.99, f"cosine {cos:.6f}"


def test_quantized_mlp_reports_itself():
    """`Engine` has to be able to state the split it actually applied — a
    projection silently declined for its shape is the failure mode."""
    from braid.model.quant import maybe_quantize
    assert isinstance(maybe_quantize(_w(512, 256), True), FP8Weight)
    assert not isinstance(maybe_quantize(_w(512, 250), True), FP8Weight)
