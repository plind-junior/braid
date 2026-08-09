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

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Which families of projection may be quantized. Selected by name rather than by
# one boolean because they are not equally safe and not equally worth it, and
# the measurement has to be able to say which of them was on.
#
#   mlp    gate/up/down. 54% of weight bytes, three homogeneous matrices.
#   attn   q(+gate)/k/v/o on the 8 full_attention layers.
#   gdn    in_proj_qkv, in_proj_z and out_proj on the 24 linear_attention layers.
#          NOT in_proj_a / in_proj_b -- see `GatedDeltaNet.__init__`.
#   head   lm_head. Sits directly on the greedy token; off unless asked for.
GROUPS = ("mlp", "attn", "gdn", "head")


def parse_groups(spec: str | Iterable[str] | None) -> frozenset[str]:
    """`"mlp,attn"` / `("mlp", "attn")` / `"all"` / `""` -> a set of group names.

    Unknown names raise. A silent typo here would produce a run that looks
    quantized in the command line and is not, which is the kind of thing that
    reaches a published table.
    """
    if not spec:
        return frozenset()
    names = spec.split(",") if isinstance(spec, str) else list(spec)
    names = [n.strip() for n in names if n.strip()]
    if "all" in names:
        return frozenset(GROUPS)
    bad = sorted(set(names) - set(GROUPS))
    if bad:
        raise ValueError(f"unknown quant group(s) {bad}; known: {list(GROUPS)}")
    return frozenset(names)


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


def _act_scale(t: torch.Tensor) -> torch.Tensor:
    """Per-tensor dequant scale for an **activation**.

    Three kernels — cast, abs, reduce — and it runs inside a captured graph on
    every decode step, so kernel count is what matters here and the tensor is a
    few tens of KiB. Contrast `_weight_scale`.
    """
    return (t.detach().float().abs().amax() / FP8_MAX).clamp(min=EPS).reshape(1, 1)


def _weight_scale(t: torch.Tensor) -> torch.Tensor:
    """Per-tensor dequant scale for a **weight**, without copying it.

    Same value as `_act_scale`, computed the other way round, because the
    trade-off is inverted: this runs once at load and never inside a graph, and
    the tensors are large. `t.float().abs().amax()` allocates a full fp32 image
    — 3.8 GiB for a `[248320, 4096]` lm_head — which OOMs a card already holding
    the model, which is exactly when quantizing it is worth doing. Two bare
    reductions allocate two scalars.
    """
    t = t.detach()
    return ((torch.maximum(t.amax().abs(), t.amin().abs()).float() / FP8_MAX)
            .clamp(min=EPS).reshape(1, 1))


# Largest fp32 temporary `quantize_weight` will build at once. The cast has to
# go through fp32 (`_scaled_mm` wants the division done there), so a big weight
# is converted in row bands instead of whole.
_CAST_CHUNK_BYTES = 256 * 2 ** 20


def quantize_weight(w: torch.Tensor) -> FP8Weight:
    """`[N, K]` -> fp8 with one per-tensor scale. Done once, at load."""
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(w.shape)}")
    N, K = w.shape
    if K % 16 or N % 16:
        raise ValueError(
            f"_scaled_mm needs both weight dims to be multiples of 16; got "
            f"[{N}, {K}]. Leave this projection in bf16.")
    scale = _weight_scale(w)
    data = torch.empty((N, K), dtype=FP8, device=w.device)
    rows = max(1, min(N, _CAST_CHUNK_BYTES // (K * 4)))
    for i in range(0, N, rows):
        band = w[i:i + rows]
        data[i:i + rows] = (band.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return FP8Weight(data=data, scale=scale, shape=(N, K))


_FUSED = None
_FUSED_STATE = "not tried"


def warm_fused(enable: bool = True) -> bool:
    """Load the fused activation-quantization kernel, or turn it off.

    **Called up front rather than lazily on purpose.** The first call would
    otherwise JIT-compile, and the first call can land inside a CUDA graph
    capture, where building an extension is not something to discover. `Engine`
    calls this when it is built with any FP8 group.

    `warm_fused(False)` forces the torch spelling, which is what the parity gate
    uses to compare the two. They are asserted bit-identical, so this is a
    performance switch and never a numerics one — there is no "which arm did I
    measure" ambiguity to resolve later.

    Returns whether the fused path is active. A failure to build is reported
    through `fused_state()` rather than raised: the fallback is exact, so a box
    without a working toolchain should still run, but it must not do so
    *silently*, because the whole point is a launch-count change that would
    otherwise vanish from a benchmark with no trace.
    """
    global _FUSED, _FUSED_STATE
    if not enable:
        _FUSED, _FUSED_STATE = None, "disabled by caller"
        return False
    if _FUSED is not None:
        return True
    try:
        from braid.kernels.loader import load_gdn

        mod = load_gdn()
        if not hasattr(mod, "quantize_act"):
            _FUSED, _FUSED_STATE = None, "extension has no quantize_act"
            return False
        _FUSED, _FUSED_STATE = mod, "active"
        return True
    except Exception as e:  # noqa: BLE001 - reported, not swallowed; see docstring
        _FUSED = None
        # split, not splitlines: an exception constructed with no args has an
        # empty str(), and "".splitlines()[0] would crash the report-and-fall-
        # back path for exactly the failure class it exists to absorb.
        _FUSED_STATE = f"unavailable: {type(e).__name__}: {str(e).split(chr(10), 1)[0][:120]}"
        return False


def fused_state() -> str:
    """One line on whether the fused quantizer is running. Printed by benches."""
    return _FUSED_STATE


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
    # Nine kernels in the torch spelling below, two in the fused one, and this
    # runs ~129 times per decode step — measured at ~1.18 ms of a 9.75 ms step
    # at B=1 on the 9B, which is 12%. `braid/kernels/csrc/quant_act.cu` carries
    # the argument that the two are bit-identical; `tests/test_quant.py` asserts
    # it rather than trusting it.
    if _FUSED is not None and x.is_cuda and x.is_contiguous():
        q, scale = _FUSED.quantize_act(x)
        return q, scale
    scale = _act_scale(x)
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


def project(x: torch.Tensor,
            *weights: torch.Tensor | FP8Weight) -> list[torch.Tensor]:
    """`F.linear(x, w)` for several projections that share one activation.

    The activation is quantized **once** for all of them. That is not a
    micro-optimisation: `quantize_act` is a device-side amax plus a scaled cast
    at ~16 us a call, against a ~16 us saving on the GEMM itself, so paying it
    per projection hands the entire win back. Attention's q/k/v and the GDN
    block's qkv/z all read the same `x`, which is why they call this rather than
    `linear` three times.

    Mixed sets are fine — a weight left in bf16 takes the ordinary path — so a
    group can quantize the matrices worth quantizing and leave the rest alone
    without the caller branching.
    """
    if not any(isinstance(w, FP8Weight) for w in weights):
        return [F.linear(x, w) for w in weights]
    *lead, K = x.shape
    xq, sx = quantize_act(x.reshape(-1, K))
    out = []
    for w in weights:
        if isinstance(w, FP8Weight):
            out.append(fp8_matmul(xq, sx, w, out_dtype=x.dtype)
                       .reshape(*lead, w.shape[0]))
        else:
            out.append(F.linear(x, w))
    return out


def maybe_quantize(w: torch.Tensor, enabled: bool) -> torch.Tensor | FP8Weight:
    """Quantize if asked and if the shape allows it; otherwise leave it alone.

    Declining a shape silently would be worse than either refusing or forcing,
    so `Engine.quant_report()` states the split that was actually applied.
    """
    if not enabled or w.dim() != 2:
        return w
    N, K = w.shape
    return w if (K % 16 or N % 16) else quantize_weight(w)
