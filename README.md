# braid

A single-GPU inference engine for **hybrid** language models — attention interleaved with a
linear-recurrent mixer (Gated DeltaNet) — built to serve them **at concurrency** on one
RTX 5090.

> **Status: the engine runs, serves, and is slower than llama.cpp on decode.**
> braid loads `Qwen3.5-4B`, matches HuggingFace to fp32 machine precision over all 32
> layers, and serves concurrent streams over SSE with continuous batching. **170 tests
> green** on a remote RTX 5090. The head-to-head that decides whether the thesis holds
> has **not** been run — see [Where braid actually is](#where-braid-actually-is).
> Every number below is labelled **measured** or **projected**.

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

## Where braid actually is

Three different measurements that are easy to conflate. **Decode** is the step with no
prefill in it. **Served** is the whole service, prefill included. llama.cpp's column is its
own serving benchmark. They are not a head-to-head and are not presented as one.

**Decode** — bf16, graphs on, CUDA kernels, median of 3 processes:

| batch | braid measured | llama.cpp measured | ratio |
|---:|---:|---:|---:|
| 1 | 131.2 tok/s | 250.11 | **0.52×** |
| 16 | **1,592.9** tok/s (10.044 ms/step) | 1,879.68 | **0.85×** |

**braid loses both rows, and they are published unchanged.** The c=1 loss is expected and
by design — batch-1 decode is not braid's axis, and braid serves BF16 against llama.cpp's
Q8_0, twice the weight bytes. The B=16 loss is not by design; closing it is the work.

**Served** — c=16, 128-token prompts, 64 new tokens, medians of 3 processes, spread ≤ 0.8%
against a 1.70% box noise floor:

| | aggregate tok/s | prefill tok/s | prefill share of wall | ITL p99 |
|---|---:|---:|---:|---:|
| before ragged batched prefill | 122.1 | 270 | 90% | 487.6 ms |
| **now** | **850.1** | **4,301** | 40% | **15.3 ms** |

Prefill used to run one sequence per forward, so sixteen concurrent streams prefilled
sixteen times in sequence at a flat 270 tok/s. The scan's loop costs one iteration per
*column*, not per token, so a batch of sixteen rows costs what one row costs: prefill
throughput is now **exactly `270 × rows-per-forward`** to three significant figures. No
arithmetic got faster.

**The comparison that decides the project has not been run.** That is ROADMAP Phase 4
items 3–4: reproduce llama.cpp's baseline on the same box in the same session, then sweep
c ∈ {1,2,4,8,16} in ABBA order, 5 processes per arm, speculation off in both, fresh
non-repeating prompts, GIL-free multi-process client. Until then braid has *component*
numbers, not a verdict.

## What we are aiming at

Projections from the roofline for this model plus **measured** state-traffic and
weight-read rates, with what braid actually measures beside them:

| parallel | llama.cpp (measured) | braid (**projected**) | braid (**measured**, decode) |
|---:|---:|---:|---:|
| 1 | 250.11 | — | 131.2 |
| 8 | 1,418.06 | ~1,300 | not re-measured at the current config |
| 16 | 1,879.68 | ~2,390 | **1,592.9 — 33% short of projection** |
| 32 | 2,497.41 | ~4,110 | not attempted |
| 64 | 2,928.34 | ~6,440 | not attempted |

**These are projections, not results**, and the one point where both exist shows the
projection was optimistic. They assumed braid hits end-to-end the bandwidth efficiency its
scan kernel hits in isolation; it does not, because the step is GEMM-dominated and braid
carries twice llama.cpp's weight bytes.

> **Unresolved:** the B=32/64 rows contradict a decision braid already made. Phase 1
> concluded that **batch buckets stop at 16** — c=32 does not fit in VRAM and is
> throughput-pointless, since the linear state term overtakes the fixed weight sweep around
> B=14–18. As scoped, braid cannot attempt half of its own stated target. One of the two has
> to move, and it is tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md) rather than quietly
> dropped.

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
