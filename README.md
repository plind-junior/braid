# braid

A single-GPU inference engine for **hybrid** language models — attention interleaved with a
linear-recurrent mixer (Gated DeltaNet) — built to serve them **at concurrency** on one
RTX 5090.

> **Status: braid beats llama.cpp above batch 16, and loses below it.**
> Measured head-to-head on one card in one session, 5 processes per arm:
> **+23.7% at B=32 and +48.6% at B=64**, against −16.1% at B=16 and −47.3% at B=1.
> From B=1 to B=64 braid scales **32.9×** where llama.cpp scales **11.7×**, which is
> the thesis. braid loads `Qwen3.5-4B`, matches HuggingFace to fp32 machine precision
> over all 32 layers, and serves concurrent streams over SSE with continuous batching.
> **170 tests green** on a remote RTX 5090.
>
> The locked MVP target — ≥+25% at B=16 and ≥+100% at B=64 — is **not met**, so this is
> a NO-GO against the target despite the wins. See
> [The head-to-head](#the-head-to-head). Every number is labelled **measured** or
> **projected**.

---

## Why

Hybrid models are the best quality-per-byte checkpoints that fit on a 32 GB card. The
interesting question is not whether they run — several engines run them — but how well they
run **with many concurrent requests**, which is the shape of real agent workloads.

Measured on an RTX 5090, `llama.cpp` serving `Qwen3.5-4B-Q8_0`:

| parallel | aggregate tok/s | ms/step | weight bandwidth | **% of memory wall** |
|---:|---:|---:|---:|---:|
| 1 | 250.11 | 4.00 | 1,117 GB/s | **74%** |
| 8 | 1,418.06 | 5.64 | 792 GB/s | 52% |
| 16 | 1,879.68 | 8.52 | 524 GB/s | **35%** |
| 32 | 2,497.41 | 12.81 | 349 GB/s | 23% |
| 64 | 2,928.34 | 21.85 | 204 GB/s | **13%** |

A dense model reads its weights **once per step** regardless of batch size, so batching
should push *toward* the memory wall, not away from it. llama.cpp goes the other way: 74%
efficient at batch 1, 13% at batch 64.

**braid's thesis, in one line:**

> At concurrency, a GDN hybrid runs at a fraction of the memory wall.
> braid targets the wall.

That is the whole idea. It is a reproducible number, not a claim about anyone's code.
## The head-to-head

**Measured 2026-08-08, both arms in one session, ABBA order, 5 processes per arm per
point, medians across processes.** Decode-only aggregate throughput, KV 128..256 on
both arms (`llama-batched-bench -npp 128 -ntg 128` against braid's graphed decode
seeded to the same depth). Box noise floor 1.66%; host healthy throughout
(2,820–2,865 MHz SM, 13,801 MHz mem, 474–515 W).

| batch | llama.cpp Q8_0 | braid BF16 | braid FP8-MLP | delta | |
|---:|---:|---:|---:|---:|:--|
| 1 | 250.0 | 131.3 (0.53×) | 131.7 (0.53×) | −47.3% | lose |
| 2 | 452.2 | 221.7 (0.49×) | 232.2 (0.51×) | −48.7% | lose |
| 4 | 810.7 | 433.0 (0.53×) | 451.8 (0.56×) | −44.3% | lose |
| 8 | 1,399.8 | 765.2 (0.55×) | 840.9 (0.60×) | −39.9% | lose |
| 16 | 1,874.8 | 1,425.2 (0.76×) | 1,573.7 (0.84×) | −16.1% | lose |
| 32 | 2,454.2 | 2,994.5 (1.22×) | **3,035.9 (1.24×)** | **+23.7%** | **win** |
| 64 | 2,915.9 | 4,169.9 (1.43×) | **4,333.4 (1.49×)** | **+48.6%** | **win** |

Per-arm spreads are 0.0–0.4%, so every verdict sits far outside noise. **braid crosses
llama.cpp between B=16 and B=32** and the losing rows are published unchanged.

The thesis is about the *shape* of the curve, and the shape holds. From B=1 to B=64
braid scales **32.9×** where llama.cpp scales **11.7×** — a dense model reads its
weights once per step, and braid keeps converting batch into throughput after
llama.cpp has stopped. That is the whole claim, and it is now a measurement.

**The locked MVP target is still not met.** It asks for ≥+25% at B=16 and ≥+100% at
B=64; braid delivers −16.1% and +48.6%. So this is a **NO-GO against the target** even
though braid wins the two largest batches, and the design's falsification clause
applies: re-plan rather than iterate. What the re-plan has to work with is that the
gap at low batch is weight bytes — braid serves **BF16 (8.44 GB)** against llama.cpp's
**Q8_0 (4.47 GB)**, and quantizing the MLP alone (54% of weights) already moves B=16
from 0.76× to 0.84×.

### Measured against projected

The projections that motivated the build, with what braid actually does beside them:

| batch | projected | measured (FP8-MLP) | of projection |
|---:|---:|---:|---:|
| 8 | ~1,300 | 840.9 | 65% |
| 16 | ~2,390 | 1,573.7 | 66% |
| 32 | ~4,110 | 3,035.9 | 74% |
| 64 | ~6,440 | 4,333.4 | 67% |

braid lands at a consistent **~2/3 of its own roofline projection**. The projections
assumed end-to-end the bandwidth efficiency the scan kernel reaches in isolation; the
step is GEMM-dominated and does not get there. They were optimistic by a stable factor
rather than wrong in shape, which is why the crossover still happened — just later and
smaller than predicted.

### Two things this measurement corrected

**The previously published 0.85× at B=16 was not shape-matched.** braid's decode bench
seeded rows with an 8-token prompt and timed at KV 8..256 while llama.cpp decoded at
KV 128..256. Decode attention reads the whole live KV every step, so braid was being
timed on a cheaper step. Shape-matched, BF16 at B=16 is **0.76×**, not 0.85×.

**"Batch buckets stop at 16" was wrong, and it was hiding the win.** That decision held
that c=32 does not fit in VRAM and is throughput-pointless because the linear state
term overtakes the fixed weight sweep at B=14–18. Measured: B=64 peaks at **20.61 GB
of 32.6 GB**, B=16→32 is **1.93×** and B=32→64 is **1.43×**, and at 102 MiB of state
per sequence against 8.44 GB of weights the crossover is near **B≈83**, not 14–18. The
roadmap scoped this head-to-head at c ∈ {1,2,4,8,16} — exactly the range where braid
loses every row. Run as written it would have returned a clean NO-GO with the win
sitting one bucket higher. A planning assumption, never re-measured, nearly falsified
the project's central claim.

### What is not measured here

This table is **decode only**. braid's end-to-end serving throughput including prefill
is lower and is reported separately below; llama.cpp's S_TG column excludes its prefill
too, so the comparison is like-for-like but neither number is a full serving result.
braid also has no prefix caching, which a multi-turn benchmark would reward llama.cpp
for and which is a scoped gap rather than a defect.

## Serving, end to end

braid's own service, prefill included — c=16, 128-token prompts, 64 new tokens, medians
of 3 processes, spread ≤ 0.8%:

| | aggregate tok/s | prefill tok/s | prefill share of wall | ITL p99 |
|---|---:|---:|---:|---:|
| before ragged batched prefill | 122.1 | 270 | 90% | 487.6 ms |
| **now** | **850.1** | **4,301** | 40% | **15.3 ms** |

Prefill used to run one sequence per forward, so sixteen concurrent streams prefilled
sixteen times in sequence at a flat 270 tok/s. The scan's loop costs one iteration per
*column*, not per token, so a batch of sixteen rows costs what one row costs: prefill
throughput is now **exactly `270 × rows-per-forward`** to three significant figures. No
arithmetic got faster.

## What is built and verified

| Component | Status |
|---|---|
| Checkpoint loader (VLM-filtered, nested `text_config`, `A = −exp(A_log)` by tensor name) | **done** |
| Attention (gated GQA, per-head `[q\|gate]` split, partial rope), MLP, sampler | **done**, bit-exact vs HF in fp32 |
| Gated DeltaNet layer, full 32-layer forward | **done** — fp32 rel L2 **6.4e-7** vs HF |
| Batched GDN decode kernel, `(batch × n_heads)` grid, device-resident slot indirection | **done**, matches HF reference at B=1/4/8 |
| Batched slotted causal conv1d + fused SiLU | **done**, 8-step rotating-slot parity |
| Slot-pooled KV / conv / recurrent caches, batched decode B=2…16 | **done**, fp32 token identity |
| CUDA-graph buckets over `(batch, kv_len)` | **done** — replay **bit-identical** to eager, 10.3 µs across slot reassignment |
| Chunked prefill · **ragged batched prefill** | **done** — padding provably inert, bitwise |
| FP8 W8A8 on the MLP (`quant_mlp=True`, opt-in) | **done** — +10.7% at B=16 for +0.50% perplexity |
| Continuous-batching scheduler, SSE server, per-row sampling, release on disconnect | **done** |
| Perplexity harness · noise-floor + host-health harness · llama.cpp baseline | **done** |
| Paged KV blocks | deferred, with arithmetic — the MVP runs two orders below the threshold |
| Prefix caching · preemption · chunkwise prefill scan | **not started** |

**170 tests, all green**, run on a remote RTX 5090 via `make test-remote`.

**Correctness, all measured:** perplexity **8.2361** vs HF 8.2393 (**0.021%**); fp32
end-to-end 6.4e-7; bf16 greedy token identity with HF; fp32 token identity at B=2/4/8;
graph replay bit-identical at `rtol=0, atol=0`; ragged batched prefill bit-identical under
a changed pad token id.

Measured scan behaviour (`Qwen3.5-4B` / `Qwen3.6-35B-A3B` share GDN dimensions):

| batch | µs/step | rows/s | vs b=1 | % of HBM |
|---:|---:|---:|---:|---:|
| 1 | 6.78 | 147,558 | 1.00× | 40% |
| 8 | 21.04 | 380,174 | **2.58×** | **104%** |
| 32 | 92.50 | 345,950 | 2.34× | 95% |

The scan **saturates memory bandwidth at B≈4**. It is at the wall and cannot be made faster,
only smaller — which is why fp16 recurrent state is the top open question.

## Quickstart

Everything runs on a remote GPU box; nothing runs locally. A local `pytest` that dies at
import means you forgot `-remote`, not that the code is broken.

```bash
export BRAID_SSH_KEY=~/.ssh/rtx5090 BRAID_SSH_HOST=root@host BRAID_SSH_PORT=22

make provision        # torch, pytest, numpy, ninja, ccache
make test-remote      # 170 tests
make lint             # ruff — this one runs locally and costs nothing
make bench-noise      # measured noise floor + host-health verdict
make bench-scaling    # the scan scaling curve, COLD and HOT

# Serving throughput, ITL and TTFT
./scripts/remote.sh 'python3 -B -m braid.bench.serve_bench --concurrency 1 2 4 8 16'

# Competitor baseline
./scripts/remote.sh 'bash scripts/provision_llamacpp.sh'
./scripts/remote.sh 'bash scripts/stage_model.sh'
./scripts/remote.sh 'MODEL=/root/models/Qwen3.5-4B-Q8_0.gguf bash scripts/llamacpp_baseline.sh'
```

```python
from braid.model.engine import Engine

eng = Engine.from_pretrained("/root/models/Qwen3.5-4B", use_kernels=True)
print(eng.generate_batch([[1, 2, 3], [4, 5]], max_new_tokens=32))
```

## Measurement rules

Adopted from the reference engine's published methodology so a head-to-head cannot be argued
on method:

- **Correctness gates before any timing is reported.**
- Time a **captured CUDA graph** of back-to-back launches, never a sync-per-rep loop. A
  sync-per-rep harness measured the *host*, reported a copy above the HBM ceiling, and
  showed the GPU **idle at 360 MHz during its own benchmark**.
- Size the state pool **past L2**. An L2-resident microbenchmark reported the scan at
  4,306 GB/s — 282% of this card's real bandwidth.
- **One process per measurement**, 3 processes, median across them, and print the spread.
  Back-to-back sweeps in one process read 6–10% low.
- Sample clocks/power **during** the run; a depressed host invalidates the number. Current
  floors: **1.70% / 0.45%** (`make bench-noise`, re-measured per session). A gate must be
  wider than the floor.
- Use `tg` for A/B, never `pp512` — it varies up to 2.6× from cuBLAS autotuning.
- Report aggregate throughput **and** per-stream ITL. Aggregate is exactly N ÷ ITL, so they
  are one measurement seen twice, not two results.
- **Publish losing rows unchanged.** The c=1 row above loses; hiding it would make the whole
  curve untrustworthy.

## Non-goals

Each excluded on someone's published measurement, not on taste:

- **Batch-1 decode records.** Not our axis; we lose there and publish it.
- **Making the single-sequence scan faster.** A +16.7% kernel win measured **−0.18%**
  end-to-end.
- **WY / SSD / *tensor-core* chunkwise scan variants.** Every tensor-core variant loses to a
  plain chunk-cached scalar loop. Note this rules out the tensor-core ladder, **not chunking
  itself** — a chunk-cached scalar scan over prefill is the next open lever.
- **NVFP4 on GDN projections.** Measured −9% to −20% decode; tuned FP16 wins on those shapes.
- **`torch._weight_int8pack_mm` as an INT8 weight path.** 5–50× *slower* than bf16 here —
  it is doing M separate GEMVs. Re-confirmed 2026-08-08 at the real decode shapes: 0.06×.
- **Multi-GPU, tensor parallelism, CPU offload.** One card is the premise.

> **Correction, 2026-08-08.** This list previously also held *"`torch._scaled_mm` FP8 returns
> `CUBLAS_STATUS_NOT_SUPPORTED` on sm_120"*. That is **no longer true** on this box
> (torch 2.11.0+cu128, CUDA 12.8.61): `_scaled_mm` runs, and at the MLP decode shape from an
> HBM-resident working set it is **1.95× bf16** (1.81× including the activation cast). It is
> W8A8, not weight-only, so the open question is accuracy rather than availability — see
> `braid/bench/gemm_paths.py`. **Now implemented** for the MLP projections
> (`Engine(quant_mlp=True)`, opt-in): +10.7% throughput at B=16 for +0.50% perplexity.
>
> Two things did fall out of it as refuted. **Rowwise `_scaled_mm` is a slower cuBLAS path
> on sm_120** — on `mlp.down`, `out_proj` and `gdn.in_proj_z` it is slower than not
> quantizing at all — and **mixed scale modes raise `RuntimeError`**, so per-tensor
> activations with per-channel weights, the accurate-and-fast combination, does not exist
> here.

## Layout

```
braid/model/      the engine — layers, cache, graph capture, quantisation
braid/kernels/    CUDA sources + the JIT loader
braid/serve/      scheduling and serving
braid/bench/      benchmarks that produce publishable numbers
braid/reference/  slow, obviously-correct implementations the tests trust
scripts/          one-off diagnostics and remote/provisioning shell
tests/            the gate; all but two files need a real GPU
```

## Hardware

| | RTX 5090 (GB202, `sm_120a`) |
|---|---|
| VRAM | 32,607 MiB GDDR7 |
| Bandwidth | 1,792 GB/s datasheet — **1,514–1,528 GB/s measured** |
| L2 | 96 MB |
| Toolchain | CUDA 12.8.61, torch 2.11.0+cu128, driver 580.126.09 |

All bandwidth-efficiency figures in this repo use the **measured** number, not the datasheet.

## License

MIT — see [`LICENSE`](LICENSE).
