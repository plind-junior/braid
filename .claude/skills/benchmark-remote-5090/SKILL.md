---
name: benchmark-remote-5090
description: Use when timing, profiling, or A/B-testing anything on braid's remote RTX 5090 — kernel microbenchmarks, scan scaling curves, bandwidth or roofline claims, llama.cpp head-to-heads, "is this speedup real", "why did this get slower", ncu, nsys, noise floor, host health. Also use before publishing any number into the README or a runbook. Do NOT use for writing kernel code (sm120-gdn-kernels) or for provisioning and instance lifecycle (remote-gpu-workflow).
---

# Benchmarking on the remote 5090 — braid

Every number braid publishes is labelled **measured** or **projected**, and every measured one
has to survive this. `docs/THESIS.md` §4 is the measurement contract; the README's
"Measurement rules" is its public summary. This skill is the operational version.

## STOP — what is a real signal on this box

1. **A sync-per-rep loop measures the host, not the kernel.** For anything in the tens of µs
   it is not "noisier", it is *measuring the wrong thing*. braid hit this: per-step time barely
   moved from B=1 to B=8, a plain copy reported bandwidth **above the HBM ceiling**, and the
   health sampler caught the GPU **idle at 360 MHz during its own benchmark**. Time a captured
   CUDA graph of back-to-back launches — `measure_graphed()` in
   [noise_floor.py](../../../braid/bench/noise_floor.py).
2. **Size the working set past L2.** 96 MB of L2 makes small state pools fully resident. An
   L2-resident microbenchmark reported the scan at **4,306 GB/s — 282% of this card's real
   bandwidth.** State per sequence is 63.8 MiB, so B=1 may be L2-resident and B=2 may not:
   measure the whole curve, never extrapolate from one point.
3. **The first ~1 s runs at idle clocks.** Clocks need about a second to ramp. Precede timed
   reps with a discarded warmup that keeps the GPU busy >1 s. This is a clock artifact, not
   heat — do not add temperature cooldowns.
4. **Sample clocks during the run, not before it.** `HostHealthSampler` at 0.1 s intervals
   with the first two samples dropped (the card is still ramping); the run must stay busy >1 s
   or there is no verdict. Healthy is `mem ≥ 13801 MHz`, `sm ≥ 2000 MHz`,
   `peak power ≥ 400 W`. A depressed host reads 8–15% low for hours — a cross-day delta with
   no concurrent health sample is not evidence.
5. **The gate must be wider than the floor.** A 2% threshold against a 10% noise floor is a
   coin flip. Measure the floor with `make bench-noise` (currently **1.65% / 0.41%**) and set
   thresholds above it.
6. **One process per measurement.** Back-to-back sweeps in one process read 6–10% low; cuBLAS
   algo state and allocator state carry over. For a gate, run 3 independent processes × 3 reps,
   take the median across processes, and **print the spread** `(max−min)/min×100`.
7. **Nothing else on the GPU.** `nvidia-smi --query-compute-apps=pid` must be empty. A
   forgotten llama.cpp server reads ~−12% and explains a "regression" for free.
8. **Correctness gates before any timing is reported.** Parity first, then numbers. A fast
   wrong kernel is not a result.
9. **Profile with graphs ON.** A graphs-OFF kernel-time sum runs ~1.8× the real step and
   produces a systematically wrong lever list — the launch-latency classes it flags overlap
   away under the real graph.

## Quick reference

| Goal | Command / tool |
|---|---|
| Noise floor + host-health verdict | `make bench-noise` |
| Scan scaling curve, COLD and HOT | `make bench-scaling` |
| Weight-GEMM options at MVP shapes | `./scripts/remote.sh python3 -B -m braid.bench.gemm_probe` |
| FP8 path availability | `./scripts/remote.sh python3 -B -m braid.bench.fp8_probe` |
| Single kernel, wall-clock A/B | `measure_graphed(fn, inner=32, reps=20)` |
| Per-kernel metrics, stalls | `ncu` — `./.claude/skills/benchmark-remote-5090/ncu-basic.sh` |
| Timeline, launch overhead, graph behaviour | `nsys` |
| Competitor baseline | `scripts/llamacpp_baseline.sh` |
| Per-layer parity | `scripts/parity_report.py` |

## Reporting a kernel number

```
Kernel:      <name>, grid=<...> block=<...> smem=<...>
Wall:        <µs>/call  (graphed, inner=<N>, reps=<R>, warmup >1s)
Noise floor: <cv>%      (p10-p90 over median)
DRAM:        <pct>% of 1,514 GB/s measured   [never of 1,792 datasheet]
Occupancy:   <pct>%     regs=<n>/thread
Clocks live: <sm>/<mem> MHz, <W> peak — healthy | DEPRESSED (<reason>)
Bound by:    memory | compute | latency | launch     reason: <top stall>
End-to-end:  <±X%>      [mandatory — a kernel delta alone is not a result]
```

The last line is not optional. The reference engine measured +16.7% on a scan kernel that moved end-to-end
**−0.18%**. A kernel win that does not move the step is a kernel win, not an engine win.

## Roofline

`AI = total_flops / total_bytes` (matmul FLOPs = `2·M·N·K`; bytes from `dram__bytes.sum`).
Use **measured** peaks or every kernel looks falsely bad:

| Peak | Value |
|---|---|
| HBM | **1,508–1,528 GB/s measured** (1,792 datasheet) |
| L2 | 96 MB |
| FP4 `mma.sync` | ≈2,019 TOPS measured (~½ datasheet); **f32-accumulate runs at ¼ rate** |

Ridge points (datasheet FLOP/byte): FP16 468 · FP8 936 · FP4 1873. AI below ridge →
memory-bound. braid's scan is memory-bound by a wide margin; the interesting axis is bytes
moved, not FLOPs.

## ncu and nsys

Both run on the remote box via `scripts/remote.sh`. Verify they are installed before planning
around them — the box is provisioned by `scripts/provision_remote.sh`, which installs torch and
pytest, not the Nsight suite.

```bash
# metrics for one kernel — ALWAYS skip warmup launches
./scripts/remote.sh ncu --kernel-name "regex:gdn_decode.*" --launch-skip 3 --launch-count 10 \
    -o /root/prof python3 -B -m braid.bench.scan_scaling

# timeline; graphs hide captured kernels without --cuda-graph-trace=node
./scripts/remote.sh nsys profile -t cuda,nvtx --cuda-graph-trace=node --stats=true \
    -o /root/timeline --force-overwrite=true python3 -B -m braid.bench.scan_scaling
```

Key metrics: `dram__throughput.avg.pct_of_peak_sustained_elapsed` (>70% = memory-bound) ·
`sm__throughput.avg.pct_of_peak_sustained_elapsed` (>70% = compute-bound) ·
`sm__warps_active.avg.pct_of_peak_sustained_active` (achieved occupancy) ·
`l1tex__t_sector_hit_rate` · `stall_*` (lowest = the bottleneck). Build with `-lineinfo`
(braid already does) and add `--set detailed --import-source yes` for source-correlated stalls.

**`ncu` wall-clock is not real time** — it serializes and replays. Metrics from `ncu`, timing
from `measure_graphed` or `nsys`.

## Red flags — stop and re-run

- Timing loop syncs every rep → measuring the host (STOP #1)
- Bandwidth above ~1,530 GB/s → L2-resident working set, not a result (STOP #2)
- Cold single-shot number → idle clocks (STOP #3)
- Cross-day delta with no concurrent health sample → host drift is 8–15% (STOP #4)
- Delta smaller than the measured noise floor reported as a win
- `cudaMalloc`/`free` inside the timed region → allocate once outside
- Kernel-only speedup reported with no end-to-end number
- Aggregate tok/s and per-stream ITL quoted as independent wins → aggregate is exactly
  N ÷ ITL; treating them as two results is how a 6× error gets made
- "Perf-neutral" claim on a compiler-hint change with no `cuobjdump -sass` diff
- A number published to the README without its method, date, and commit

## Publishing

README numbers and `docs/runbooks/*` (internal, gitignored) carry measured values. When a change moves one, update it
in the **same commit**, keep the **measured/projected** label, and record the command that
produced it. A number copied into prose without its method is a number that will be wrong and
un-rechecked. braid's public claim rests on method — treat the labelling as load-bearing.
