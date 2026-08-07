# braid

A single-GPU inference engine for **hybrid** language models — attention interleaved with a
linear-recurrent mixer (Gated DeltaNet) — built to serve them **at concurrency** on one
RTX 5090.

> **Status: early. The engine does not exist yet.**
> What exists is a validated batched recurrent-scan kernel, a measurement harness, and a
> measured competitor baseline. Every number below is labelled **measured** or **projected**.
> Nothing here is a performance claim about braid.

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

## What we are aiming at

Using the roofline for this model plus **measured** state-traffic and weight-read rates:

| parallel | llama.cpp (measured) | braid (**projected**) | delta |
|---:|---:|---:|---:|
| 8 | 1,418 | ~1,300 | −9% |
| 16 | 1,880 | ~2,390 | +27% |
| 32 | 2,497 | ~4,110 | +64% |
| 64 | 2,928 | ~6,440 | +120% |

**These are projections, not results.** They assume braid hits end-to-end the bandwidth
efficiency its scan kernel hits in isolation. A real engine will not fully. Treat B=32/64 as
the defensible target and B=16 as a stretch; we expect to **lose** at B≤8 and will publish
that row unchanged.

Note the comparison is deliberately unfavourable to us: braid serves **BF16** against
llama.cpp's **Q8_0** — twice the weight bytes, higher precision — because both reduced-byte
paths available in stock PyTorch are dead on this card (see below).

## What is built and verified

| Component | Status |
|---|---|
| Batched GDN decode kernel, `(batch × n_heads)` grid, device-resident slot indirection | **done**, matches HF reference at B=1/4/8 |
| Batched slotted causal conv1d + fused SiLU | **done**, 8-step rotating-slot parity |
| fp32 PyTorch oracle (naive + vectorized) | done |
| CUDA-graph replay across slot reassignment | done — **10.3 µs/replay** |
| Noise-floor + host-health measurement harness | done — **1.65% / 0.41%** floors |
| llama.cpp baseline on the same card | done |
| Checkpoint loader, attention, MLP, sampler, scheduler, server | **not started** |

**38 tests, all green**, run on a remote RTX 5090 via `make test-remote`.

Measured scan behaviour (`Qwen3.5-4B` / `Qwen3.6-35B-A3B` share GDN dimensions):

| batch | µs/step | rows/s | vs b=1 | % of HBM |
|---:|---:|---:|---:|---:|
| 1 | 6.78 | 147,558 | 1.00× | 40% |
| 8 | 21.04 | 380,174 | **2.58×** | **104%** |
| 32 | 92.50 | 345,950 | 2.34× | 95% |

The scan **saturates memory bandwidth at B≈4**. It is at the wall and cannot be made faster,
only smaller — which is why fp16 recurrent state is the top open question.

## Quickstart

Everything runs on a remote GPU box; nothing runs locally.

```bash
export BRAID_SSH_KEY=~/.ssh/rtx5090 BRAID_SSH_HOST=root@host BRAID_SSH_PORT=22

make provision        # torch, pytest, numpy, ninja, ccache
make test-remote      # 38 tests
make bench-noise      # measured noise floor + host-health verdict
make bench-scaling    # the scan scaling curve, COLD and HOT

# Competitor baseline
./scripts/remote.sh 'bash scripts/provision_llamacpp.sh'
./scripts/remote.sh 'bash scripts/stage_model.sh'
./scripts/remote.sh 'MODEL=/root/models/Qwen3.5-4B-Q8_0.gguf bash scripts/llamacpp_baseline.sh'
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
- Sample clocks/power **during** the run; a depressed host invalidates the number.
- Use `tg` for A/B, never `pp512` — it varies up to 2.6× from cuBLAS autotuning.
- Report aggregate throughput **and** per-stream ITL. Aggregate is exactly N ÷ ITL.

## Things we got wrong, and corrected

Kept deliberately, because they are the reason to trust the rest.

- **The original thesis was wrong.** It claimed no engine batches the recurrent scan.
  **llama.cpp does**, and scales 11.7× to B=64. That claim was true of one engine — the
  reference competitor — not of the field. The batched scan is table stakes; the roofline
  gap is the real opening.
- **A 10× numerical bug, caught before shipping.** We copied the reference engine's clamped
  L2 norm,
  `rsqrt(max(sum_sq, 1e-12))`. HuggingFace — the implementation this checkpoint was trained
  with — uses **additive** `rsqrt(sum_sq + 1e-6)`. Identical on a healthy head, **10× apart
  on a near-zero one.** Now pinned by a test.
- **Two measurement traps** that each produced a flattering, false scaling curve. Both are
  written into the measurement rules above rather than quietly fixed.

## Non-goals

Each excluded on someone's published measurement, not on taste:

- **Batch-1 decode records.** Not our axis; we expect to lose there.
- **Making the single-sequence scan faster.** A +16.7% kernel win measured **−0.18%**
  end-to-end.
- **WY / SSD / tensor-core chunkwise scan variants.** Every tensor-core variant loses to a
  plain chunk-cached scalar loop.
- **NVFP4 on GDN projections.** Measured −9% to −20% decode; tuned FP16 wins on those shapes.
- **`torch._weight_int8pack_mm` as an INT8 weight path.** 5–50× *slower* than bf16 here —
  it is doing M separate GEMVs. Re-confirmed 2026-08-08 at the real decode shapes: 0.06×.

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
- **Multi-GPU, tensor parallelism, CPU offload.** One card is the premise.

## Documentation

| Doc | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | **Start here.** What braid is, where the code lives, the four phases, the layers sublayer by sublayer, the numerics contract, sm_120a landmines, and what is / isn't built yet |
| [`docs/THESIS.md`](docs/THESIS.md) | Why braid exists: the competitive case, the throughput arithmetic, the measurement contract, risks, and the dead-ground ledger of what has been measured and refuted |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phases, gates, kill criteria |

## Hardware

| | RTX 5090 (GB202, `sm_120a`) |
|---|---|
| VRAM | 32,607 MiB GDDR7 |
| Bandwidth | 1,792 GB/s datasheet — **1,514–1,528 GB/s measured** |
| L2 | 96 MB |
| Toolchain | CUDA 12.8.61, torch 2.11.0+cu128, driver 580.126.09 |

All bandwidth-efficiency figures in this repo use the **measured** number, not the datasheet.
