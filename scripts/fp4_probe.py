"""4-bit MLP weights on sm_120: is there a path, and does it beat FP8?

**Why this is a probe and not a project.** braid's decode step is already at
7.40 GiB of weights against llama.cpp Q8_0's derived ~7.87, and the MLP is the
largest block of that, so halving it again is the obvious next lever. But the
refutation ledger already has NVFP4 losing on this card — measured −9% to −20%
on the GDN projections, where tuned FP16 won — and that was a *decode* result on
small-M shapes, which is exactly the regime here. So the prior is that this
loses, and the cheapest honest thing is to measure the GEMM before writing a
quantizer for it.

**The comparison it must not accidentally win.** Dropping braid to 4-bit while
llama.cpp stays at Q8_0 would move braid's weight bytes below a competitor that
did not move, and the resulting ratio would be an artifact of picking different
precisions rather than a result. If a path here wins, the head-to-head has to
move to llama.cpp Q4_K_M at the same time, and perplexity has to be re-priced on
both sides. That is the condition on acting on anything this prints.

Three candidate paths, each feature-detected rather than assumed:

  int4 weight-only   `torch._weight_int4pack_mm` — groupwise scales, weights in
                     4 bits, activations bf16. This is the shape that should
                     win at decode: the step is memory-bound on weights and this
                     is the only path that halves them without touching the
                     activation.
  fp4 (NVFP4)        `torch._scaled_mm` on an e2m1 dtype, if this build has one.
                     Tensor-core 4-bit; needs M large to pay, which decode is
                     not.
  fp8 / bf16         the arms braid actually ships, as the denominators.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.bench.noise_floor import measure_graphed
from braid.model.quant import FP8Weight, fp8_matmul, quantize_act, quantize_weight

# Qwen3.5-9B MLP shapes, plus the head as the one place fp8 already wins big.
SHAPES = [
    ("mlp.gate/up", 4096, 12288),
    ("mlp.down", 12288, 4096),
    ("lm_head", 4096, 248320),
]
BATCHES = [1, 8, 32, 64]
WORKING_SET = 512 << 20  # past the 96 MB L2, or this measures cache


def bank(make, nbytes):
    return [make() for _ in range(max(2, min(64, -(-WORKING_SET // max(nbytes, 1)))))]


def cycle(b, fn):
    i = [0]

    def step():
        t = b[i[0]]
        i[0] = (i[0] + 1) % len(b)
        return fn(t)

    return step


def timed(b, fn) -> float:
    """us per call, graphed, cycling a bank big enough to defeat the 96 MB L2."""
    return measure_graphed(cycle(b, fn), inner=32, reps=20).median_s * 1e6


def have_int4() -> bool:
    return hasattr(torch.ops.aten, "_weight_int4pack_mm")


def fp4_dtype():
    """The e2m1 dtype under whichever name this build uses, or None."""
    for name in ("float4_e2m1fn_x2", "float4_e2m1fn"):
        dt = getattr(torch, name, None)
        if dt is not None:
            return dt
    return None


def try_int4(K: int, N: int, M: int, group: int = 128):
    """`_weight_int4pack_mm(x, packed, group, scales_and_zeros)`.

    Returns a timed step or None. The packing layout is version-specific and
    undocumented; anything that raises here means "no usable path in this
    build", which is a legitimate answer and the one to report.
    """
    if not have_int4():
        return None
    n_groups = K // group
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    szp = torch.randn(n_groups, N, 2, device="cuda", dtype=torch.bfloat16)
    # The packed layout has changed across torch versions — int32 [N, K] on
    # older builds, uint8 [N, K/2] nibble pairs on newer ones. Both are tried
    # rather than assumed, so "no path" means no path and not "I guessed the
    # older signature".
    attempts = [
        ("uint8 [N,K/2]",
         lambda: torch.randint(0, 255, (N, K // 2), device="cuda", dtype=torch.uint8)),
        ("int32 [N,K]",
         lambda: torch.randint(0, 16, (N, K), device="cuda", dtype=torch.int32)),
    ]
    errs = []
    for label, make in attempts:
        try:
            packed = torch.ops.aten._convert_weight_to_int4pack(make(), 8)
            out = torch.ops.aten._weight_int4pack_mm(x, packed, group, szp)
            assert out.shape == (M, N), out.shape
            return lambda: torch.ops.aten._weight_int4pack_mm(x, packed, group, szp)
        except Exception as e:  # noqa: BLE001 - a probe reports, it does not raise
            errs.append(f"{label}: {type(e).__name__} {str(e).splitlines()[0][:44]}")
    return "unavailable: " + " | ".join(errs)


def main() -> None:
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"{torch.cuda.get_device_name(0)}")
    print(f"  _weight_int4pack_mm: {'present' if have_int4() else 'ABSENT'}")
    dt4 = fp4_dtype()
    print(f"  fp4 dtype:           {dt4 if dt4 is not None else 'ABSENT'}")
    print("\nAll times us, median of the harness's own repeats, graphed.\n")
    print(f"{'shape':<14}{'M':>4}{'bf16':>10}{'fp8':>10}{'int4':>10}"
          f"{'int4/bf16':>11}{'int4/fp8':>10}")

    for label, K, N in SHAPES:
        # The weight bank is built once per shape and shared by every M: it is
        # the expensive part, and M changes only the activation.
        wbytes = N * K * 2
        b16 = bank(lambda: torch.randn(N, K, device="cuda", dtype=torch.bfloat16),
                   wbytes)
        # One real quantization, then clones of the fp8 payload. Quantizing 60
        # copies of a [248320, 4096] weight would dominate the probe's runtime
        # and measure the quantizer rather than the GEMM; the scale is shared
        # because a per-tensor scale is a scalar and does not affect traffic.
        w0 = quantize_weight(b16[0])
        b8 = [FP8Weight(data=w0.data.clone(), scale=w0.scale, shape=w0.shape)
              for _ in range(max(2, len(b16) * 2))]
        del w0

        for M in BATCHES:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            xq, sx = quantize_act(x)

            t_bf16 = timed(b16, lambda w: F.linear(x, w))
            t_fp8 = timed(b8, lambda w: fp8_matmul(xq, sx, w))

            r4 = try_int4(K, N, M)
            if callable(r4):
                t_i4 = measure_graphed(r4, inner=32, reps=20).median_s * 1e6
                print(f"{label:<14}{M:>4}{t_bf16:>10.1f}{t_fp8:>10.1f}{t_i4:>10.1f}"
                      f"{t_bf16 / t_i4:>10.2f}x{t_fp8 / t_i4:>9.2f}x")
            else:
                print(f"{label:<14}{M:>4}{t_bf16:>10.1f}{t_fp8:>10.1f}"
                      f"{'-':>10}{'':>11}{'':>10}  {r4 or 'absent'}")
            del x, xq
            torch.cuda.empty_cache()

        del b16, b8
        torch.cuda.empty_cache()

    print("\nRead this against the ledger, not on its own: NVFP4 on the GDN "
          "projections\nalready measured -9% to -20% on this card. A win here "
          "is only actionable if\nthe head-to-head moves to llama.cpp Q4_K_M at "
          "the same time -- 4-bit braid against\nan 8-bit competitor is a "
          "precision choice wearing a speedup's clothes.")


if __name__ == "__main__":
    main()
