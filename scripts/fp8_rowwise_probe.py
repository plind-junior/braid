"""Rowwise `_scaled_mm` on sm_120: does it run, is it fast, what does it cost?

The earlier `gemm_paths` probe used **scalar** scales and read 1.95x. Rowwise is
a different cuBLAS path and is the only form braid can use -- THESIS prices one
per-tensor scale over a heterogeneous pack at +4% PPL. First pass at M=16 came
back at 0.47-0.87x, i.e. *losing to bf16*, on everything but `lm_head`.

That number confounds two things, so this splits them:

  fp8 (full)      quantize the activation, then GEMM -- what a call site pays
  fp8 (pre-q)     GEMM only, activation already fp8 -- the kernel's own speed
  fp8 (tensor)    scalar scales both sides -- is *rowwise* the slow path?

The difference between the first two is per-call overhead that can be amortized
across projections sharing an activation (gate/up share x; q/k/v share x; GDN's
four input projections share x). The difference between rowwise and tensorwise
is not amortizable and would be a property of the kernel.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.bench.noise_floor import measure_graphed
from braid.model.quant import FP8, FP8_MAX, fp8_linear, quantize_act, quantize_weight

SHAPES = [
    ("mlp.gate/up",   2560,   9216),
    ("mlp.down",      9216,   2560),
    ("qkv / attn.q",  2560,   8192),
    ("out_proj",      4096,   2560),
    ("gdn.in_proj_z", 2560,   4096),
    ("lm_head",       2560, 248320),
]
WORKING_SET = 512 << 20      # past the 96 MB L2, or this measures cache


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
    r = measure_graphed(cycle(b, fn), inner=32, reps=20)
    del b
    torch.cuda.empty_cache()
    return r.median_s * 1e6


for M in (16,):
    print(f"\n=== M={M}   us per call, HBM-resident")
    print(f"{'shape':<15}{'K':>7}{'N':>8}{'bf16':>8}{'fp8 full':>10}{'fp8 pre-q':>11}"
          f"{'fp8 tensor':>12}{'quant us':>10}{'kernel x':>10}")
    for name, K, N in SHAPES:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        xq, sx = quantize_act(x)
        one = torch.tensor(1.0, device="cuda")

        mk16 = lambda: torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        mk8 = lambda: quantize_weight(
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
        mk8raw = lambda: torch.randn(N, K, device="cuda").to(FP8)

        t16 = timed(mk16, N * K * 2, lambda t: F.linear(x, t))
        tfull = timed(mk8, N * K, lambda t: fp8_linear(x, t))
        tpre = timed(mk8, N * K, lambda t: torch._scaled_mm(
            xq, t.data.t(), scale_a=sx, scale_b=t.scale, out_dtype=torch.bfloat16))
        ttensor = timed(mk8raw, N * K, lambda t: torch._scaled_mm(
            xq, t.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))

        print(f"{name:<15}{K:>7}{N:>8}{t16:>8.1f}{tfull:>10.1f}{tpre:>11.1f}"
              f"{ttensor:>12.1f}{tfull - tpre:>10.1f}{t16 / tpre:>9.2f}x")
        del x, xq, sx
        torch.cuda.empty_cache()
