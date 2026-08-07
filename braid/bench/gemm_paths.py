"""What can be done about the 8.17 ms of GEMM in the decode step?

`docs/runbooks/decode-profile.md` leaves the B=16 step GEMM-dominated: 8.17 ms
of 12.07, which is **68% of the weight-read roofline** computed against the
1,508 GB/s copy benchmark. The obvious reading is "32% headroom, go get it".

That reading has two problems this probe exists to test.

1. **1,508 GB/s is a copy number, not a per-tensor ceiling.** A GEMM reading a
   `[2560, 9216]` weight does not get to move bytes the way a giant contiguous
   copy does. So every path is measured here against a **streaming-read floor
   taken on the same tensor** -- a full reduction over the weight, which is the
   cheapest thing that touches every byte. That is the number a GEMM can
   actually be held to.

2. **The alternatives may not exist.** `torch._weight_int8pack_mm` is already
   measured at 5-50x slower than bf16 on this box, and `torch._scaled_mm` FP8
   returns `CUBLAS_STATUS_NOT_SUPPORTED` on sm_120. So before "quantize the
   weights" is promoted onto the critical path, something has to actually run.

`ncu` would answer the limiter question directly and **cannot run on this
box**: the driver reports `ERR_NVGPUCTRPERM`, which needs a host-level
`NVreg_RestrictProfilingToAdminUsers=0` and a reboot. This is the measurement
that replaces it -- wall-clock A/B against a floor, which is weaker than a
stall breakdown but is not a guess.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from braid.bench.noise_floor import HostHealthSampler, measure_graphed

# The shapes decode actually issues, with how many launches per step at B=16.
# From `decode_profile.py --mode locate`.
SHAPES = [
    ("mlp.gate/up",    2560,   9216, 64),
    ("mlp.down",       9216,   2560, 32),
    ("qkv / attn.q",   2560,   8192, 32),
    ("out_proj",       4096,   2560, 32),
    ("gdn.in_proj_z",  2560,   4096, 24),
    ("attn.k/v",       2560,   1024, 16),
    ("gdn.a, gdn.b",   2560,     32, 48),
    ("lm_head",        2560, 248320,  1),
]


# A single decode weight is 5-47 MiB and this card has **96 MB of L2**. Timing
# one weight in a loop therefore measures L2, not HBM: the first cut of this
# probe reported a 3,815 GB/s "read floor" against a 1,508 GB/s card and every
# ratio under it was meaningless. Each shape is measured over a rotating bank
# of distinct copies sized past L2 instead, so the bytes come from memory the
# way they do in a real step -- where 8.41 GB sweeps past a 96 MB cache.
WORKING_SET_BYTES = 512 << 20


def _bank(make, nbytes: int) -> list:
    """Enough distinct tensors that the loop cannot stay resident in L2."""
    n = max(2, -(-WORKING_SET_BYTES // max(nbytes, 1)))
    return [make() for _ in range(min(n, 64))]


def _cycle(bank: list, fn):
    """A callable that advances through the bank on every invocation."""
    i = [0]

    def step():
        t = bank[i[0]]
        i[0] = (i[0] + 1) % len(bank)
        return fn(t)

    return step


def _bw(nbytes: int, seconds: float) -> float:
    return nbytes / seconds / 1e9


def run(M: int, peak: float) -> None:
    print(f"\n=== M={M}   weight-byte GB/s, and % of a same-tensor streaming read")
    print(f"{'shape':<16}{'K':>7}{'N':>8}{'n/step':>7}"
          f"{'read floor':>12}{'bf16 mm':>10}{'% floor':>9}{'pad64':>9}{'lt':>9}")

    total_bf16_us = 0.0
    total_floor_us = 0.0
    for name, K, N, per_step in SHAPES:
        nbytes = N * K * 2
        bank = _bank(lambda: torch.randn(N, K, device="cuda", dtype=torch.bfloat16),
                     nbytes)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        # Floor: the cheapest op that reads every weight byte exactly once.
        floor = measure_graphed(_cycle(bank, lambda w: w.sum()), inner=32, reps=20)
        base = measure_graphed(_cycle(bank, lambda w: F.linear(x, w)),
                               inner=32, reps=20)

        # Does padding M give the heuristic a shape it likes better? Same weight
        # bytes, more FLOPs -- if it wins, the kernel choice was the problem.
        xp = torch.randn(max(M, 64), K, device="cuda", dtype=torch.bfloat16)
        pad = measure_graphed(_cycle(bank, lambda w: F.linear(xp, w)),
                              inner=32, reps=20)

        try:
            torch.backends.cuda.preferred_blas_library("cublaslt")
            lt = measure_graphed(_cycle(bank, lambda w: F.linear(x, w)),
                                 inner=32, reps=20)
            lt_s = f"{_bw(nbytes, lt.median_s):>9.0f}"
        except Exception as e:
            lt_s = f"{type(e).__name__[:7]:>9}"
        finally:
            torch.backends.cuda.preferred_blas_library("cublas")

        total_bf16_us += base.median_s * 1e6 * per_step
        total_floor_us += floor.median_s * 1e6 * per_step
        print(f"{name:<16}{K:>7}{N:>8}{per_step:>7}"
              f"{_bw(nbytes, floor.median_s):>12.0f}"
              f"{_bw(nbytes, base.median_s):>10.0f}"
              f"{base.median_s and floor.median_s / base.median_s * 100:>8.0f}%"
              f"{_bw(nbytes, pad.median_s):>9.0f}{lt_s}")
        del bank, x, xp
        torch.cuda.empty_cache()

    print(f"\n  sum over one step: bf16 GEMM {total_bf16_us / 1e3:.3f} ms, "
          f"same-tensor read floor {total_floor_us / 1e3:.3f} ms "
          f"({total_floor_us / total_bf16_us * 100:.0f}% of it)")
    print(f"  for reference, weight bytes / {peak:.0f} GB/s copy peak = "
          f"{sum(K * N * 2 * n for _, K, N, n in SHAPES) / peak / 1e6:.3f} ms")


def quantized() -> None:
    """Do any reduced-byte weight paths actually run on sm_120?"""
    print("\n=== reduced-byte weight paths, [16, 2560] x [2560, 9216]")
    M, K, N = 16, 2560, 9216
    nb16, nb8 = N * K * 2, N * K
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    b16 = _bank(lambda: torch.randn(N, K, device="cuda", dtype=torch.bfloat16), nb16)
    base = measure_graphed(_cycle(b16, lambda w: F.linear(x, w)), inner=32, reps=20)
    print(f"  bf16 baseline            {base.median_s * 1e6:>9.1f} us  "
          f"{_bw(nb16, base.median_s):>6.0f} GB/s")
    del b16
    torch.cuda.empty_cache()

    s8 = torch.randn(N, device="cuda", dtype=torch.bfloat16).abs() + 0.1
    b8 = _bank(lambda: torch.randint(-127, 127, (N, K), device="cuda",
                                     dtype=torch.int8), nb8)
    try:
        r = measure_graphed(_cycle(b8, lambda w: torch._weight_int8pack_mm(x, w, s8)),
                            inner=8, reps=10)
        print(f"  torch._weight_int8pack_mm{r.median_s * 1e6:>9.1f} us  "
              f"{_bw(nb8, r.median_s):>6.0f} GB/s  "
              f"{base.median_s / r.median_s:>5.2f}x vs bf16")
    except Exception as e:
        print(f"  torch._weight_int8pack_mm  FAILED {type(e).__name__}: {str(e)[:50]}")
    del b8
    torch.cuda.empty_cache()

    # NOTE: `_scaled_mm` is **W8A8**, not weight-only -- both operands are fp8.
    # The activation cast is priced in below because a real implementation pays
    # it, and it is the accuracy question, not the speed one: THESIS records one
    # per-tensor fp8 scale over a heterogeneous pack at +4% PPL, per-row flat.
    try:
        sc = torch.tensor(1.0, device="cuda")
        bf8 = _bank(lambda: torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn),
                    nb8)
        r = measure_graphed(
            _cycle(bf8, lambda w: torch._scaled_mm(
                x.to(torch.float8_e4m3fn), w.t(), scale_a=sc, scale_b=sc,
                out_dtype=torch.bfloat16)),
            inner=32, reps=20)
        print(f"  torch._scaled_mm fp8     {r.median_s * 1e6:>9.1f} us  "
              f"{_bw(nb8, r.median_s):>6.0f} GB/s  "
              f"{base.median_s / r.median_s:>5.2f}x vs bf16  (incl. act cast)")
        xf = x.to(torch.float8_e4m3fn)
        r2 = measure_graphed(
            _cycle(bf8, lambda w: torch._scaled_mm(xf, w.t(), scale_a=sc, scale_b=sc,
                                                   out_dtype=torch.bfloat16)),
            inner=32, reps=20)
        print(f"  torch._scaled_mm fp8     {r2.median_s * 1e6:>9.1f} us  "
              f"{_bw(nb8, r2.median_s):>6.0f} GB/s  "
              f"{base.median_s / r2.median_s:>5.2f}x vs bf16  (act pre-cast)")
    except Exception as e:
        print(f"  torch._scaled_mm fp8       FAILED {type(e).__name__}: {str(e)[:50]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", type=int, nargs="+", default=[16])
    p.add_argument("--peak", type=float, default=1508.0)
    args = p.parse_args()
    with HostHealthSampler() as health:
        for M in args.batches:
            run(M, args.peak)
        quantized()
    print(f"\nhost health: {health.report()}")


if __name__ == "__main__":
    main()
