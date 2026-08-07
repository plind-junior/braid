"""Is there a usable reduced-byte weight path on sm_120 for M in 1..64?

Context: `torch._weight_int8pack_mm` measured 5-50x SLOWER than bf16 here and
its cost grows linearly in M, so it is doing M separate GEMVs. That removes
the cheap INT8 option. Decode is weight-bandwidth-bound, so halving weight
bytes is worth ~2x at low batch -- exactly where BF16 braid would otherwise
lose to llama.cpp's Q8_0.

Candidates, cheapest first:
  1. torch._scaled_mm with float8_e4m3fn weights (native Blackwell FP8)
  2. bf16 baseline, for reference

Weights are cycled over several copies so the measurement streams from DRAM
instead of sitting in the 96 MB L2 -- the same trap that made the scan
microbenchmark report 4300 GB/s.
"""
from __future__ import annotations

import torch

from braid.bench.noise_floor import measure_graphed

SHAPES = [("mlp.gate/up", 2560, 9216), ("lm_head", 2560, 248320)]
BATCHES = [1, 8, 16, 64]
_L2_BYTES = 96 << 20


def _copies(nbytes: int) -> int:
    """Enough distinct weight copies to overflow L2 several times over."""
    return max(2, min(8, (4 * _L2_BYTES) // max(nbytes, 1) + 1))


def _cycled(make, nbytes):
    ws = [make() for _ in range(_copies(nbytes))]
    c = {"i": 0}

    def pick():
        w = ws[c["i"] % len(ws)]
        c["i"] += 1
        return w

    return pick


def probe_bf16(M, K, N):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    nbytes = N * K * 2
    pick = _cycled(lambda: torch.randn(N, K, device="cuda", dtype=torch.bfloat16), nbytes)
    r = measure_graphed(lambda: torch.nn.functional.linear(x, pick()), inner=16, reps=20)
    return r.median_s, nbytes / r.median_s / 1e9


def probe_fp8(M, K, N):
    """torch._scaled_mm, e4m3 weights, bf16-ish activations cast to fp8.

    _scaled_mm wants both operands fp8 and column-major B; it also has
    alignment rules that reject small M on some builds, hence the try.
    """
    x = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    nbytes = N * K
    pick = _cycled(
        lambda: torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).t().contiguous().t(),
        nbytes,
    )
    sa = torch.tensor(1.0, device="cuda")
    sb = torch.tensor(1.0, device="cuda")

    def call():
        return torch._scaled_mm(x, pick().t(), scale_a=sa, scale_b=sb,
                                out_dtype=torch.bfloat16)

    call()  # surface errors before capture
    r = measure_graphed(call, inner=16, reps=20)
    return r.median_s, nbytes / r.median_s / 1e9


def main() -> None:
    print(f"{'shape':>14} {'M':>4} {'bf16 us':>9} {'bf16 GB/s':>10} "
          f"{'fp8 us':>9} {'fp8 GB/s':>9} {'speedup':>8}")
    for name, K, N in SHAPES:
        for M in BATCHES:
            t16, g16 = probe_bf16(M, K, N)
            try:
                t8, g8 = probe_fp8(M, K, N)
                row = f"{t8 * 1e6:>9.1f} {g8:>9.0f} {t16 / t8:>7.2f}x"
            except Exception as e:
                row = f"{'FAIL':>9} {'-':>9}  {type(e).__name__}: {str(e)[:44]}"
            print(f"{name:>14} {M:>4} {t16 * 1e6:>9.1f} {g16:>10.0f} {row}")


if __name__ == "__main__":
    main()
