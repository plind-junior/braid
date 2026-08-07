"""FP8 W8A8 linear, via `torch._scaled_mm`.

**Why this and nothing else.** Decode is weight-bound and the bf16 GEMMs are
already at 86% of a same-tensor streaming read — 99–105% on every shape that
dominates the bytes (`docs/runbooks/decode-profile.md`). There is no
kernel-choice win left; the only lever on the GEMM is fewer bytes. Of the
reduced-byte paths on this card:

    torch._weight_int8pack_mm     0.06x bf16 -- M separate GEMVs
    NVFP4 on GDN projections      -9% to -20% decode; tuned FP16 beats it
    torch._scaled_mm fp8          the only one that runs

**Per-tensor scales, and that was forced by measurement, not chosen.** The
natural design is per-token activation scales and per-output-channel weight
scales, because THESIS prices one per-tensor scale over a heterogeneous fused
pack at +4% PPL. Measured on this box (`scripts/fp8_scalemode_probe.py`,
M=16, HBM-resident, us per call):

    shape           bf16    a=tensor b=tensor    a=row b=row
    mlp.gate/up     32.4        16.7  1.94x       20.9  1.55x
    mlp.down        50.7        20.6  2.46x       56.8  0.89x   <- loses to bf16
    out_proj        23.2        12.3  1.89x       27.9  0.84x   <- loses to bf16
    gdn.in_proj_z   17.8         9.5  1.87x       19.6  0.91x   <- loses to bf16

Rowwise is a slower cuBLAS path on sm_120 and on half these shapes it is slower
than not quantizing at all. **Mixing modes is not supported** — a per-tensor
activation scale with per-channel weight scales raises `RuntimeError`, so the
accurate-and-fast combination does not exist here. That leaves per-tensor.

THESIS's +4% is not automatically this configuration's cost: it measured one
scale over a **fused pack** holding q/k/v/gate/beta row groups, where one amax
is dominated by the largest group. A single MLP projection is homogeneous. That
is a different claim and it is gated separately, on perplexity, not assumed.

**e4m3, not e5m2.** e4m3 has 3 mantissa bits and a 448 max; e5m2 trades two
mantissa bits for range that weights at this scale do not use. Three mantissa
bits is ~3.7e-2 relative per GEMM, roughly 10x bf16 — which is why the gate
here is perplexity and not the 5e-3 parity tolerance the rest of braid uses.

Everything must survive CUDA-graph capture: static shapes, no host syncs, no
allocation that depends on device values. The amax is a device-side reduce and
is fine; `.item()` anywhere in this file would be a capture failure.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
# Scales are fp32 and never zero: an all-zero tensor would divide by zero and
# produce NaNs that stay NaN through every later gate.
EPS = 1e-12


@dataclass
class FP8Weight:
    """A `[N, K]` weight held as fp8 with one dequant scale for the tensor.

    `scale` is `[1, 1]` fp32 rather than a Python float because `_scaled_mm`
    wants a device tensor, and building one per call would allocate inside a
    captured graph.
    """

    data: torch.Tensor       # [N, K] float8_e4m3fn
    scale: torch.Tensor      # [1, 1] float32
    shape: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data.numel() + 4


def _amax_scale(t: torch.Tensor) -> torch.Tensor:
    return (t.detach().float().abs().amax() / FP8_MAX).clamp(min=EPS).reshape(1, 1)


def quantize_weight(w: torch.Tensor) -> FP8Weight:
    """`[N, K]` -> fp8 with one per-tensor scale. Done once, at load."""
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(w.shape)}")
    N, K = w.shape
    if K % 16 or N % 16:
        raise ValueError(
            f"_scaled_mm needs both weight dims to be multiples of 16; got "
            f"[{N}, {K}]. Leave this projection in bf16.")
    scale = _amax_scale(w)
    data = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return FP8Weight(data=data.contiguous(), scale=scale, shape=(N, K))


def quantize_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`[M, K]` -> fp8 plus a `[1, 1]` scale. Dynamic, per call.

    Dynamic rather than calibrated: a static scale would be one fewer reduction
    but would also be sized by whatever the worst token in some calibration set
    looked like, and would drift with the prompt distribution. The reduction is
    over a `[16, 2560]` tile — 40 KiB — and is **not** free at ~16 us a call,
    which is why callers are expected to quantize once and reuse across
    projections that share an activation (`fp8_matmul` takes the pre-quantized
    form for exactly that reason).
    """
    scale = _amax_scale(x)
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


def fp8_matmul(xq: torch.Tensor, sx: torch.Tensor, w: FP8Weight,
               out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The GEMM alone, on an already-quantized `[M, K]` activation.

    `_scaled_mm` computes `(mat1 * scale_a) @ (mat2 * scale_b)`, so the scales
    are dequant factors: the arithmetic is the same product with the rounding
    moved into fp8.
    """
    return torch._scaled_mm(xq, w.data.t(), scale_a=sx, scale_b=w.scale,
                            out_dtype=out_dtype)


def fp8_linear(x: torch.Tensor, w: FP8Weight) -> torch.Tensor:
    """`F.linear(x, w)` for a single call site. Leading dims are preserved.

    Pays a fresh activation quantization every call. Prefer `quantize_act` once
    plus `fp8_matmul` per projection wherever an activation feeds more than one.
    """
    *lead, K = x.shape
    xq, sx = quantize_act(x.reshape(-1, K))
    return fp8_matmul(xq, sx, w, out_dtype=x.dtype).reshape(*lead, w.shape[0])


def linear(x: torch.Tensor, w: torch.Tensor | FP8Weight) -> torch.Tensor:
    """Dispatch on the weight's type, so call sites do not branch."""
    return fp8_linear(x, w) if isinstance(w, FP8Weight) else F.linear(x, w)


def maybe_quantize(w: torch.Tensor, enabled: bool) -> torch.Tensor | FP8Weight:
    """Quantize if asked and if the shape allows it; otherwise leave it alone.

    Declining a shape silently would be worse than either refusing or forcing,
    so `Engine.quant_report()` states the split that was actually applied.
    """
    if not enabled or w.dim() != 2:
        return w
    N, K = w.shape
    return w if (K % 16 or N % 16) else quantize_weight(w)
