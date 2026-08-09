"""Where the decode step's milliseconds actually go, at one batch.

**The question this exists to settle.** braid reaches 54% of the card's measured
1,514 GB/s memory wall at B=1, where llama.cpp reaches 88%. Two very different
things produce that, and they have opposite fixes:

  * **Gaps.** The step is a long chain of small kernels and the GPU is idle
    between them. Fix: fewer, bigger kernels (fusion). Graph replay already
    removes *launch* cost, so any remaining gap is dependency stalls and tail
    effects, not CPU.
  * **Kernels.** The GPU is busy the whole step but the kernels themselves move
    bytes inefficiently. Fix: better kernels, or fewer bytes.

So the headline is not a list of hot kernels — it is `sum(kernel device time)`
against the wall-clock step. If those two agree, there are no gaps and hunting
launches is wasted effort. If they do not, the difference IS the prize, and it
is quoted in milliseconds rather than inferred.

**Two timers, on purpose.** The step time is taken with CUDA events on an
unprofiled run, because CUPTI inflates what it measures. The profiler is used
only for the *decomposition*, and the two are reconciled by reporting the
profiled step time beside the clean one — if profiling has distorted the shape,
that shows up as the two disagreeing by more than a few percent, and the
breakdown should then be read as ratios and not as absolute milliseconds.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from braid.bench.decode_speed import STATE_DTYPES, _seed
from braid.model.engine import Engine
from braid.model.graph import GraphedDecoder
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

# Buckets are matched in order and the first hit wins, so they run specific to
# general. `_OTHER` catching a large share is a result worth seeing, not a bug to
# paper over -- it means the step is dominated by something this list does not
# name, and the fix is to name it rather than to widen an existing bucket.
BUCKETS: list[tuple[str, str]] = [
    ("gdn scan", r"gdn_decode_kernel|gdn_prefill_kernel"),
    ("conv", r"conv1d_decode"),
    # fp8 GEMMs land in cutlass/cublasLt; bf16 GEMMs in the ordinary gemm names.
    ("fp8 gemm", r"scaled_mm|fp8|f8f8|cutlass.*(e4m3|e5m2)"),
    ("gemm", r"gemm|Gemm|GEMM|cutlass|nvjet|ampere_|sm[0-9]+_xmma"),
    # `quantize_act` is amax + clamp + mul + cast per activation. They are
    # elementwise/reduce kernels and are NOT separable from other elementwise
    # work by name alone -- which is exactly why "elementwise" is one bucket and
    # is not labelled "quantization". Calling it that would be asserting an
    # attribution the trace does not support.
    ("reduce", r"reduce_kernel|Reduce|amax|norm"),
    ("elementwise", r"elementwise|vectorized|unrolled_elementwise|CatArrayBatched"),
    ("index/copy", r"index_select|index_copy|indexSelect|copy_|Copy|gather|scatter"),
    ("softmax/attn", r"softmax|attention|flash|sdpa|efficient"),
]


def bucket_of(name: str) -> str:
    for label, pat in BUCKETS:
        if re.search(pat, name):
            return label
    return "_OTHER"


def _time_clean(fn, steps: int, reset) -> float:
    """ms per step, CUDA events around the whole run. No profiler attached."""
    reset()
    for _ in range(min(steps, 64)):
        fn()
    torch.cuda.synchronize()
    reset()
    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    a.record()
    for _ in range(steps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / steps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--quant", default="all")
    p.add_argument("--state-dtype", default="fp16", choices=list(STATE_DTYPES))
    p.add_argument("--top", type=int, default=18, help="hottest kernels to list")
    # The two launch-count levers, off-switchable so each can be priced alone.
    # Both toggles are asserted bit-identical to their fallbacks (test_quant,
    # test_gdn_raw_gates), so these change WHERE work happens, never what.
    p.add_argument("--no-fused-quant", action="store_true",
                   help="torch-spelling activation quantization (9 kernels/call)")
    p.add_argument("--no-raw-gates", action="store_true",
                   help="alpha/beta in torch (~8 launches/layer/step)")
    args = p.parse_args()

    ck = load_checkpoint(MODEL_DIR, device="cuda")
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16,
                                 use_kernels=True, quant=args.quant,
                                 state_dtype=STATE_DTYPES[args.state_dtype])
    del ck
    torch.cuda.empty_cache()
    if args.no_fused_quant:
        from braid.model.quant import warm_fused
        warm_fused(False)
    if args.no_raw_gates:
        eng.set_raw_gates(False)

    B = args.batch
    cache = eng.allocate_cache(args.max_len, max_slots=B)
    _seed(eng, cache, B, args.prompt_len)
    snap = cache.snapshot()
    tokens = torch.full((B, 1), 42, dtype=torch.long, device="cuda")
    slots = torch.arange(B, device="cuda")
    dec = GraphedDecoder(eng, cache, buckets=(B,))
    cache.restore(snap)

    def step():
        dec.step(tokens, slots)

    clean_ms = _time_clean(step, args.steps, lambda: cache.restore(snap))

    n = 32
    cache.restore(snap)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            step()
        torch.cuda.synchronize()

    per: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    hot: list[tuple[float, int, str]] = []
    total_us = 0.0
    total_calls = 0
    for e in prof.key_averages():
        us = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if us <= 0.0:
            continue
        total_us += us
        total_calls += e.count
        per[bucket_of(e.key)] += us
        calls[bucket_of(e.key)] += e.count
        hot.append((us, e.count, e.key))

    busy_ms = total_us / 1e3 / n

    print(f"\nmodel {MODEL_DIR.name}  B={B}  fp8={args.quant or 'none'}  "
          f"state={args.state_dtype}  KV {args.prompt_len}..{args.prompt_len + args.steps}  "
          f"fused-quant={'off' if args.no_fused_quant else 'on'}  "
          f"raw-gates={'off' if args.no_raw_gates else 'on'}")
    print(f"\n  step, CUDA events, no profiler   {clean_ms:8.3f} ms")
    print(f"  sum of kernel device time        {busy_ms:8.3f} ms   "
          f"({busy_ms / clean_ms * 100:.0f}% of the step)")
    print(f"  gap                              {clean_ms - busy_ms:8.3f} ms   "
          f"({(1 - busy_ms / clean_ms) * 100:.0f}% of the step)")
    print(f"  kernels launched per step        {total_calls / n:8.0f}")
    print("\n  A large gap says fuse; a small gap says the kernels themselves are\n"
          "  the cost and fusing will buy nothing.")

    print(f"\n  {'bucket':<14}{'ms/step':>9}{'% busy':>8}{'launches':>10}")
    for label, us in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<14}{us / 1e3 / n:>9.3f}{us / total_us * 100:>7.1f}%"
              f"{calls[label] / n:>10.0f}")

    print(f"\n  hottest {args.top} kernels:")
    for us, c, name in sorted(hot, reverse=True)[:args.top]:
        print(f"  {us / 1e3 / n:>8.3f} ms {c / n:>6.0f}x  {name[:88]}")

    step_bytes = sum(eng.step_bytes().values())
    # Weight bytes only -- the state and KV terms scale with B and are counted
    # by the caller's own model. This is a floor on the step, not a prediction
    # of it, and it is labelled that way wherever it is quoted.
    ideal_ms = step_bytes / 1514e9 * 1e3
    print(f"\n  weight bytes per step {step_bytes / 2 ** 30:.2f} GiB -> "
          f"{ideal_ms:.3f} ms at the measured 1,514 GB/s wall "
          f"({ideal_ms / clean_ms * 100:.0f}% of the step; weights only, so this "
          f"is a floor)")


if __name__ == "__main__":
    main()
