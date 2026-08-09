# braid

A single-GPU inference engine for **hybrid** language models — attention interleaved with a
linear-recurrent mixer (Gated DeltaNet) — built to serve them **at concurrency** on one
RTX 5090.

> **Status: braid beats llama.cpp above batch 16, reaches parity at 16, and loses below it.**
> Measured head-to-head on `Qwen3.5-9B`, one card, one session, 5 processes per arm:
> **+30.1% at B=32 and +57.8% at B=64**, −0.7% at B=16, −34.9% at B=1.
> From B=1 to B=64 braid scales **36.9×** where llama.cpp scales **15.2×**, which is
> the thesis. braid matches HuggingFace to fp32 machine precision over all 32 layers
> and serves concurrent streams over SSE with continuous batching.
> **194 tests green** on a remote RTX 5090, on both the 9B and the 4B.
>
> The locked MVP target — ≥+25% at B=16 and ≥+100% at B=64 — is **still not met**, so
> this remains a NO-GO against the target despite the wins. The re-scoped target the
> falsification clause produced (≥+20% at B=32, ≥+50% at B=64) **is** met. See
> [The head-to-head](#the-head-to-head). Every number is labelled **measured** or
> **projected**.

---

## Why

Hybrid models are the best quality-per-byte checkpoints that fit on a 32 GB card. The
interesting question is not whether they run — several engines run them — but how well they
run **with many concurrent requests**, which is the shape of real agent workloads.

Measured on an RTX 5090, `llama.cpp` serving `Qwen3.5-9B-Q8_0`. The bandwidth column
counts **weight bytes only** — the ~7.87 GiB the decode step sweeps, i.e. the 8.87 GiB
GGUF less its ~1.01 GiB embedding table, which is gathered rather than read:

| parallel | aggregate tok/s | ms/step | weight bandwidth | **% of memory wall** |
|---:|---:|---:|---:|---:|
| 1 | 157.4 | 6.35 | 1,330 GB/s | **88%** |
| 8 | 901.1 | 8.88 | 952 GB/s | 63% |
| 16 | 1,369.6 | 11.68 | 723 GB/s | 48% |
| 32 | 1,934.3 | 16.54 | 511 GB/s | 34% |
| 64 | 2,398.9 | 26.68 | 317 GB/s | **21%** |

A dense model reads its weights **once per step** regardless of batch size, so batching
should push *toward* the memory wall, not away from it. llama.cpp goes the other way: 74%
efficient at batch 1, 21% at batch 64.

**braid's thesis, in one line:**

> At concurrency, a GDN hybrid runs at a fraction of the memory wall.
> braid targets the wall.

That is the whole idea. It is a reproducible number, not a claim about anyone's code.

---

## The head-to-head

**Measured 2026-08-09 on `Qwen3.5-9B`. Every arm in one session, rotated order, 5
processes per arm per point, medians across processes.** Decode-only aggregate
throughput, KV 128..256 on all arms (`llama-batched-bench -npp 128 -ntg 128` against
braid's graphed decode seeded to the same depth). Host healthy throughout
(2,850–2,857 MHz SM, 13,801 MHz mem, 526–542 W peak).

| batch | llama.cpp Q8_0 | braid BF16 | braid FP8-MLP | braid FP8-all | best delta | |
|---:|---:|---:|---:|---:|---:|:--|
| 1 | 157.4 | 83.0 (0.53×) | 93.1 (0.59×) | 102.5 (0.65×) | −34.9% | lose |
| 2 | 290.0 | 146.6 (0.51×) | 166.8 (0.58×) | 181.7 (0.63×) | −37.3% | lose |
| 4 | 532.4 | 282.4 (0.53×) | 326.7 (0.61×) | 348.4 (0.65×) | −34.6% | lose |
| 8 | 901.1 | 545.8 (0.61×) | 625.3 (0.69×) | 662.4 (0.74×) | −26.5% | lose |
| 16 | 1,369.6 | 1,092.5 (0.80×) | 1,280.6 (0.93×) | 1,360.0 (0.99×) | −0.7% | parity |
| 32 | 1,934.3 | 2,040.5 (1.05×) | 2,348.8 (1.21×) | **2,516.3 (1.30×)** | **+30.1%** | **win** |
| 64 | 2,398.9 | 2,744.6 (1.14×) | 3,275.3 (1.37×) | **3,786.0 (1.58×)** | **+57.8%** | **win** |

Spreads over the five processes: llama.cpp 1.37%, braid 0.11–0.25%. Every verdict sits
far outside noise. **braid crosses llama.cpp between B=16 and B=32** and the losing
rows are published unchanged.

The thesis is about the *shape* of the curve, and the shape holds. From B=1 to B=64
braid scales **36.9×** where llama.cpp scales **15.2×** — braid keeps converting batch
into throughput after llama.cpp has stopped. That is the whole claim, and it is a
measurement.

**The locked MVP target is still not met.** It asks for ≥+25% at B=16 and ≥+100% at
B=64; braid delivers −0.7% and +57.8%. So this remains a **NO-GO against the target**.
The re-scoped target that the falsification clause produced after the 4B run — ≥+20% at
B=32 and ≥+50% at B=64 — **is** met, at +30.1% and +57.8%.

### The lever was weight bytes, and it is now spent

The 4B run named the cause of the low-batch deficit: braid read BF16 where llama.cpp
read Q8_0. Extending FP8 W8A8 from the MLP alone to every projection closed it.

| | decode-step weight bytes | vs llama.cpp | B=16 | B=64 |
|---|---:|---:|---:|---:|
| braid BF16 | 14.78 GiB | 1.88× | 0.80× | 1.14× |
| braid FP8-MLP | 10.28 GiB | 1.31× | 0.93× | 1.37× |
| **braid FP8-all** | **7.40 GiB** | **0.94×** | **0.99×** | **1.58×** |

llama.cpp's per-step figure (~7.87 GiB) is **derived**, not measured: the 8.87 GiB GGUF
less its ~1.01 GiB embedding table, which is gathered rather than swept. braid's is
measured by `Engine.step_bytes()`.

So braid now reads *fewer* weight bytes per decode step than llama.cpp does, and the
remaining B≤8 deficit is no longer bytes — it is launch overhead. At B=1 braid runs 265
GEMM launches per step and reaches 54% of the memory wall against llama.cpp's 88%; that
is a fixed per-step cost that batching amortises and quantization cannot touch.

**What FP8 costs, measured on the 9B:** perplexity 7.1272 → 7.0773, **−0.70%**. A
*decrease* is not evidence that quantization helped — it says the cost sits below what a
16k-token corpus can resolve. Two groups are deliberately excluded: `in_proj_a` /
`in_proj_b`, because `a_raw` is exponentiated and fp8's three mantissa bits would land
inside an exponent for 0.4% of a layer's bytes; and `embed_tokens`, which is gathered
rather than swept.

### Why the 9B is a harder target than the 4B

Qwen3.5-9B grew `hidden_size` (2,560 → 4,096) and `intermediate_size` (9,216 → 12,288)
and **nothing else** — the GDN state and KV shapes are byte-for-byte identical to the
4B. So braid's weight deficit against Q8_0 doubled in absolute terms while the
per-sequence state term it wins on stayed fixed. On the 4B, BF16 alone won +23.7% at
B=32; on the 9B the same arm gets +5.5%. The win at B≥32 on this model is bought by
quantization, not inherited.

### Peak VRAM

Measured after construction, so it is what serving needs resident rather than the
load-time transient:

| batch | BF16 | FP8-MLP | FP8-all |
|---:|---:|---:|---:|
| 1 | 16.93 | 12.43 | 9.55 |
| 16 | 19.93 | 15.44 | 12.56 |
| 32 | 23.15 | 18.66 | 15.77 |
| 64 | 29.55 | 25.06 | **22.18** |

Card total 31.36 GiB. BF16 at B=64 leaves 1.8 GiB of headroom; FP8-all leaves 9.2.

### What is not measured here

This table is **decode only**. braid's end-to-end serving throughput including prefill
is lower and is reported separately below; llama.cpp's S_TG column excludes its prefill
too, so the comparison is like-for-like but neither number is a full serving result.
braid also has no prefix caching, which a multi-turn benchmark would reward llama.cpp
for and which is a scoped gap rather than a defect.

## Serving, end to end

braid's own service, prefill included — `Qwen3.5-9B`, 128-token prompts, 64 new tokens,
graphs on, one process per point:

| c | BF16 tok/s | FP8-all tok/s | ITL p50 | ITL p99 | TTFT p50 | VRAM GB | prefill % of wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51.5 | 56.5 | 9.9 ms | 10.2 ms | 519 ms | 10.6 | 45% |
| 8 | 361.4 | 399.6 | 12.2 ms | 14.3 ms | 1,800 ms | 12.7 | 39% |
| 16 | 720.8 | 808.2 | 11.9 ms | 16.0 ms | 1,792 ms | 14.7 | 40% |
| 32 | **997.8** | **1,136.7** | 12.9 ms | 21.0 ms | 2,789 ms | 18.2 | 54% |
| 64 | 959.9 | 1,085.1 | 17.2 ms | 33.2 ms | 6,467 ms | 25.3 | 70% |

(latency and VRAM columns are the FP8-all arm.)

**Served throughput peaks at c=32 and falls at c=64, and the cause is prefill.** Decode
alone keeps climbing to B=64; end to end it does not, because prefill's share of the
wall clock goes 40% → 54% → 70%. The GDN prefill scan is still a Python loop over the
sequence axis, so admitting 64 prompts costs real time that no amount of decode
throughput hides. That is the next lever, and it is named in the roadmap as a
chunk-cached scalar scan over prefill.

The host-health sampler flags c=1 and c=8 as power-depressed (272–386 W against a 400 W
floor). At those concurrencies the GPU is genuinely not saturated — that is the
measurement, not a throttled host. Every point the claim rests on (c ≥ 16) is healthy.

**Where this came from.** Prefill used to run one sequence per forward. The scan's loop
costs one iteration per *column*, not per token, so a batch of sixteen rows costs what
one row costs; batching the rows moved the 4B from 122.1 to 850.1 tok/s at c=16 and the
ITL p99 from 487.6 ms to 15.3 ms at the same time. No arithmetic got faster.

## What is built and verified

| Component | Status |
|---|---|
| Checkpoint loader (VLM-filtered, nested `text_config`, `A = −exp(A_log)` by tensor name) | **done** |
| Attention (gated GQA, per-head `[q\|gate]` split, partial rope), MLP, sampler | **done**, bit-exact vs HF in fp32 |
| Gated DeltaNet layer, full 32-layer forward | **done** — fp32 rel L2 **4.5e-7** vs HF (9B), 6.1e-7 (4B) |
| Batched GDN decode kernel, `(batch × n_heads)` grid, device-resident slot indirection | **done**, matches HF reference at B=1/4/8 |
| Batched slotted causal conv1d + fused SiLU | **done**, 8-step rotating-slot parity |
| Slot-pooled KV / conv / recurrent caches, batched decode B=2…64 | **done**, fp32 token identity |
| CUDA-graph buckets over `(batch, kv_len)`, to B=64 | **done** — replay **bit-identical** to eager, for every FP8 group too |
| Chunked prefill · **ragged batched prefill** | **done** — padding provably inert, bitwise |
| FP8 W8A8 by group — `mlp`, `attn`, `gdn`, `head` (`quant="all"`, opt-in) | **done** — decode-step weights **halved**, 14.78 → 7.40 GiB, for −0.70% perplexity |
| Continuous-batching scheduler, SSE server, per-row sampling, release on disconnect | **done** |
| Perplexity harness · noise-floor + host-health harness · llama.cpp baseline | **done** |
| Paged KV blocks | deferred, with arithmetic — the MVP runs two orders below the threshold |
| Prefix caching · preemption · chunkwise prefill scan | **not started** |

**194 tests across 23 modules, all green** on both `Qwen3.5-9B` and `Qwen3.5-4B`, run on
a remote RTX 5090 via `make test-remote` (or `scripts/test_isolated.sh`, one module per
process — at 9B two module-scoped engines no longer share a card).

**Correctness on the 9B, all measured:** perplexity **7.1272** vs HF **7.1312**
(**0.0564%**); full 32-layer fp32 forward vs HF **4.538e-07**, cosine 1.000000000, tokens
identical; full-stack fp32 decode-vs-prefill **1.478e-06**, tokens identical; fp32 token
identity at B=2/4/8; graph replay bit-identical at `rtol=0, atol=0` for bf16 and for every
FP8 group; ragged batched prefill bit-identical under a changed pad token id.

The two full-stack fp32 gates run on the **CPU** — two fp32 copies of the 9B are 72 GiB,
and truncating the stack would retire the one claim they exist to make, that all 32 layers
are wired to the right mixer. The box has 245 GiB of host RAM and a 16-token fp32 forward
costs ~1.5 s per arm.

Measured scan behaviour (`Qwen3.5-4B`, `Qwen3.5-9B` and `Qwen3.6-35B-A3B` all share GDN
dimensions — the 9B grew only `hidden_size` and `intermediate_size`):

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
make test-remote      # 194 tests
make lint             # ruff — this one runs locally and costs nothing
make bench-noise      # measured noise floor + host-health verdict
make bench-scaling    # the scan scaling curve, COLD and HOT

# Serving throughput, ITL and TTFT
./scripts/remote.sh 'python3 -B -m braid.bench.serve_bench --quant all'

# Competitor baseline
./scripts/remote.sh 'bash scripts/provision_llamacpp.sh'
./scripts/remote.sh 'MODEL=Qwen3.5-9B bash scripts/stage_model.sh'
./scripts/remote.sh 'REPS=5 NPL=1,2,4,8,16,32,64 BATCHES="1 2 4 8 16 32 64" \
  GGUF=/root/models/Qwen3.5-9B-Q8_0.gguf BRAID_MODEL_DIR=/root/models/Qwen3.5-9B \
  bash scripts/head_to_head.sh'
```

```python
from braid.model.engine import Engine

eng = Engine.from_pretrained("/root/models/Qwen3.5-9B", use_kernels=True, quant="all")
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
> `braid/bench/gemm_paths.py`. **Now implemented** for every projection group
> (`Engine(quant="all")`, opt-in): decode-step weight bytes halved for −0.70% perplexity
> on the 9B, which is what takes B=16 from 0.80× to 0.99× and B=64 from 1.14× to 1.58×.
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
