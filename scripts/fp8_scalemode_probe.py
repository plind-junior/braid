"""Which operand's scale mode makes rowwise `_scaled_mm` slow?

Measured: tensorwise fp8 beats bf16 comfortably, rowwise often loses -- on
`mlp.down` tensorwise is 20.6 us against rowwise 56.8 us for the same bytes.
That 2.8x is a kernel-selection property, not arithmetic.

It matters which side causes it, because the two scales are not equally
negotiable. THESIS prices collapsing the **weight** scale to per-tensor at +4%
PPL, so `scale_b` has to stay per-output-channel. The **activation** scale is a
free choice: a dynamic per-tensor amax is one reduction over the whole tile and
is not what THESIS measured.

So if the cost is in `scale_a`, there is a fast configuration braid can use:
per-tensor activations, per-channel weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.bench.noise_floor import measure_graphed
from braid.model.quant import FP8, FP8_MAX, quantize_act, quantize_weight

SHAPES = [
    ("mlp.gate/up",   2560,  9216),
    ("mlp.down",      9216,  2560),
    ("out_proj",      4096,  2560),
    ("gdn.in_proj_z", 2560,  4096),
]
M = 16
WORKING_SET = 512 << 20


def bank(make, nbytes):
    return [make() for _ in range(max(2, min(64, -(-WORKING_SET // max(nbytes, 1)))))]


def cycle(b, fn):
    i = [0]

    def step():
        t = b[i[0]]
        i[0] = (i[0] + 1) % len(b)
        return fn(t)
    return step


def timed(make, nbytes, fn):
    b = bank(make, nbytes)
    try:
        r = measure_graphed(cycle(b, fn), inner=32, reps=20)
        return r.median_s * 1e6
    except Exception as e:
        return f"{type(e).__name__[:9]}"
    finally:
        del b
        torch.cuda.empty_cache()


print(f"=== M={M}, us per call, HBM-resident. scale_a = activation, scale_b = weight")
print(f"{'shape':<15}{'K':>7}{'N':>7}{'bf16':>8}"
      f"{'a=T b=T':>10}{'a=T b=row':>11}{'a=row b=T':>11}{'a=row b=row':>13}")

for name, K, N in SHAPES:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    xq, sx_row = quantize_act(x)
    sx_t = (x.detach().float().abs().amax() / FP8_MAX).reshape(1, 1)
    one = torch.tensor(1.0, device="cuda")

    mk8 = lambda: quantize_weight(
        torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
    mk16 = lambda: torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    def sm(t, sa, sb):
        return torch._scaled_mm(xq, t.data.t(), scale_a=sa, scale_b=sb,
                                out_dtype=torch.bfloat16)

    t16 = timed(mk16, N * K * 2, lambda t: F.linear(x, t))
    tt = timed(mk8, N * K, lambda t: sm(t, one, one))
    tr = timed(mk8, N * K, lambda t: sm(t, sx_t, t.scale))
    rt = timed(mk8, N * K, lambda t: sm(t, sx_row, one))
    rr = timed(mk8, N * K, lambda t: sm(t, sx_row, t.scale))

    fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else v
    print(f"{name:<15}{K:>7}{N:>7}{fmt(t16):>8}"
          f"{fmt(tt):>10}{fmt(tr):>11}{fmt(rt):>11}{fmt(rr):>13}")
    del x, xq, sx_row
    torch.cuda.empty_cache()
