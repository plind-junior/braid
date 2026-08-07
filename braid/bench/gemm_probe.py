"""Probe the weight-only GEMM options at the batch sizes the MVP cares about.

This is the gating experiment for the MVP design. llama.cpp serves Qwen3.5-4B
from Q8_0 (4.47 GB). Our HF weights are BF16 (8.44 GB) -- 2x the bytes, which
by roofline loses at B<=8 and only wins above B=16. If an INT8 weight-only
path is available and reaches a decent fraction of HBM at M in 1..64, the MVP
can match llama.cpp's byte count and win across the whole curve.

Reports achieved GB/s on the WEIGHT bytes, since decode is weight-bound: the
weight is read once per step regardless of M, so GB/s should stay roughly flat
in M and the win from batching shows up as tokens/s, not as bandwidth.
"""
from __future__ import annotations

import torch

from braid.bench.noise_floor import HostHealthSampler, measure_graphed

# Qwen3.5-4B shapes. (name, K, N) for an [M,K] @ [K,N] linear.
SHAPES = [
    ("mlp.gate/up", 2560, 9216),
    ("mlp.down", 9216, 2560),
    ("gdn.in_proj", 2560, 12352),
    ("gdn.out_proj", 4096, 2560),
    ("attn.q(gated)", 2560, 8192),
    ("attn.o", 4096, 2560),
    ("lm_head", 2560, 248320),
]
BATCHES = [1, 8, 16, 64]


def _bf16(M, K, N):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    return lambda: torch.nn.functional.linear(x, w), w.numel() * 2


def _int8(M, K, N):
    """torch._weight_int8pack_mm: W8A16 weight-only, the torchao int8 path.

    Weight is [N, K] int8 with a per-row fp scale; activation stays bf16.
    """
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
    s = torch.randn(N, device="cuda", dtype=torch.bfloat16).abs() + 0.1
    return lambda: torch._weight_int8pack_mm(x, w, s), w.numel()


def main() -> None:
    print(f"{'shape':>16} {'K':>7} {'N':>7} {'M':>4} "
          f"{'bf16 us':>9} {'bf16 GB/s':>10} {'int8 us':>9} {'int8 GB/s':>10} {'speedup':>8}")
    with HostHealthSampler():
        for name, K, N in SHAPES:
            for M in BATCHES:
                try:
                    f16, b16 = _bf16(M, K, N)
                    r16 = measure_graphed(f16, inner=32, reps=20)
                    g16 = b16 / r16.median_s / 1e9
                except Exception as e:
                    print(f"{name:>16} bf16 FAILED: {type(e).__name__}: {str(e)[:60]}")
                    continue
                try:
                    f8, b8 = _int8(M, K, N)
                    r8 = measure_graphed(f8, inner=32, reps=20)
                    g8 = b8 / r8.median_s / 1e9
                    sp = r16.median_s / r8.median_s
                    i8s = f"{r8.median_s * 1e6:>9.1f}"
                    i8g = f"{g8:>10.0f}"
                    sps = f"{sp:>7.2f}x"
                except Exception as e:
                    i8s, i8g, sps = f"{'n/a':>9}", f"{'n/a':>10}", f"  {type(e).__name__[:6]}"
                print(f"{name:>16} {K:>7} {N:>7} {M:>4} "
                      f"{r16.median_s * 1e6:>9.1f} {g16:>10.0f} {i8s} {i8g} {sps}")


if __name__ == "__main__":
    main()
