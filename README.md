# braid

A single-GPU inference engine for **hybrid** language models — attention interleaved with a
linear-recurrent mixer (Gated DeltaNet) — built to serve them **at concurrency** on one
RTX 5090.

---

## Why

Hybrid models are the best quality-per-byte checkpoints that fit on a 32 GB card. The
interesting question is not whether they run — several engines run them — but how well they
run **with many concurrent requests**, which is the shape of real agent workloads.

A dense model reads its weights once per decode step regardless of batch size, so batching
should push an engine *toward* the memory wall. Measured on this card, `llama.cpp` goes the
other way: 88% of the wall at batch 1, 12% at batch 128, nearly flat from B=64 up. And a
GDN hybrid makes concurrency even cheaper than a dense model does: the recurrent state is
fixed-size per sequence, where a KV cache grows with context.

> **braid's thesis, in one line:** at concurrency, a GDN hybrid runs at a fraction of the
> memory wall. braid targets the wall.

## Results

All numbers **measured** on one RTX 5090 (`Qwen3.5-9B`, llama.cpp at Q8_0 with its own
default settings, braid in its shipping configuration — FP8 on every projection, fp16
state storage). Decode head-to-head: every arm in one session, rotated order, 5 processes
per arm, medians; spreads llama.cpp 1.73%, braid ≤0.99%.

| batch | llama.cpp tok/s | braid tok/s | delta | |
|---:|---:|---:|---:|:--|
| 1 | 157.1 | 122.5 | −22.0% | lose |
| 8 | 901.9 | 807.1 | −10.5% | lose |
| 16 | 1,368.1 | **1,731.2** | **+26.5%** | win |
| 32 | 1,929.0 | **3,246.2** | **+68.3%** | win |
| 64 | 2,391.8 | **5,232.2** | **+118.8%** | win |
| 96 | 2,564.3 | **6,200.5** | **+141.8%** | win |
| 128 | 2,662.8 | **7,199.2** | **+170.4%** | win |

braid crosses llama.cpp between B=8 and B=16, and the losing rows are published
unchanged. From B=1 to B=128 braid scales **58.8×** where llama.cpp scales **17.0×** —
braid keeps converting batch into throughput long after llama.cpp has stopped. The same
sweep at 4× the prompt depth (npp=512) holds the shape: +26.3% at B=16, +106.8% at B=64.

Server against server — braid's HTTP/SSE service racing `llama-server` on the same box,
one client, both sides provisioned for their slot counts, prefill included:

| concurrent streams | llama-server tok/s | braid tok/s | delta | TTFT p50 (llama → braid) |
|---:|---:|---:|---:|---:|
| 1 | 125.4 | 115.9 | −7.6% | 92 → 35 ms |
| 8 | 377.1 | 649.7 | **+72%** | 676 → 84 ms |
| 16 | 379.2 | 1,156.4 | **+205%** | 1,355 → 88 ms |
| 32 | 402.3 | 1,746.1 | **+334%** | 2,295 → 105 ms |
| 64 | 399.2 | 2,288.9 | **+473%** | 3,682 → 124 ms |
| 128 | 47.7 | 2,795.1 | — | 287 s → **158 ms** |

llama-server's time-to-first-token grows linearly at ~57 ms per stream — serialized
prompt processing — while braid's chunked ragged prefill, co-scheduled with decode, holds
TTFT at 84–158 ms across the whole curve. The c=128 delta is left unnumbered because a
four-minute TTFT is a failure mode, not a throughput. In-process, braid serves
**3,633 tok/s at c=128** with ITL p50 of 18.4 ms, in 26 GB of VRAM.

Correctness is measured the same way the speed is: perplexity **7.1272 vs HuggingFace's
7.1312** (0.056%), full 32-layer fp32 forward within 4.5e-7 of HF with identical tokens,
CUDA-graph replay bit-identical to eager, and the prefill scan kernel bit-identical to
the decode kernel applied T times. The gate is 259 tests across 27 modules, green on
both the 9B and the 4B.

## How

The engine is plain PyTorch plus a small set of JIT-compiled CUDA kernels, each one
priced against a measurement before it stayed:

- **FP8 W8A8 on every projection** (`torch._scaled_mm`, per-tensor) — decode-step weight
  bytes halved to 7.40 GiB, fewer than llama.cpp's Q8_0 sweeps, for −0.70% perplexity.
- **fp16 recurrent state pool** — storage only, the scan math stays fp32. Halves the
  per-sequence cost that decides where the batch curve stops; it is what makes B=128 fit.
- **Batched GDN decode kernel** with device-resident slot indirection, and a
  **chunk-cached prefill scan** that is the same recurrence with the loop moved inside —
  asserted bit-identical to the decode kernel, which no tensor-core reformulation could be.
- **CUDA-graph buckets** over `(batch, kv_len)` — replay bit-identical to eager.
- **A launch diet** — fused activation quantization and in-kernel gates, −43% kernel
  launches per decode step, bit-identical to the torch spelling they replaced.
- **Continuous batching** over an SSE server: chunked ragged prefill co-scheduled with
  decode, per-row sampling, slot release on disconnect.

## Quickstart

Everything runs on a remote GPU box; nothing needs a local GPU. A local `pytest` that
dies at import means you forgot `-remote`, not that the code is broken.

```bash
export BRAID_SSH_KEY=~/.ssh/rtx5090 BRAID_SSH_HOST=root@host BRAID_SSH_PORT=22

make provision        # torch, pytest, numpy, ninja, ccache
make test-remote      # the suite, on the box
make lint             # ruff — runs locally, costs nothing

# Serving throughput, ITL and TTFT
./scripts/remote.sh 'python3 -B -m braid.bench.serve_bench --quant all --state-dtype fp16'

# The llama.cpp head-to-head
./scripts/remote.sh 'bash scripts/provision_llamacpp.sh'
./scripts/remote.sh 'MODEL=Qwen3.5-9B bash scripts/stage_model.sh'
./scripts/remote.sh 'bash scripts/head_to_head.sh'
```

```python
import torch
from braid.model.engine import Engine

# The shipping configuration: fp8 on every projection, fp16 state storage.
eng = Engine.from_pretrained("/root/models/Qwen3.5-9B", use_kernels=True,
                             quant="all", state_dtype=torch.float16)
print(eng.generate_batch([[1, 2, 3], [4, 5]], max_new_tokens=32))
```

## Measurement rules

The part of the project most easily damaged by carelessness, so it is short and strict:

- Correctness gates before any timing is reported.
- Every published number is labelled **measured** or **projected**; losing rows are
  published unchanged.
- One process per measurement, medians across processes, spreads printed. A/B claims are
  same-session only — cross-session deltas under ~2% are noise on this box.
- Clocks and power are sampled *during* the run; a depressed host invalidates the number.
- Time captured CUDA graphs, never a sync-per-rep loop, from working sets sized past L2.

## Non-goals

Each excluded on a measurement, not on taste: batch-1 decode records (not our axis; we
lose there and publish it) · tensor-core chunkwise scan variants (every one measured
loses to the chunk-cached scalar scan) · 4-bit weights (wins exactly where braid loses,
collapses where braid wins) · multi-GPU, tensor parallelism, CPU offload (one card is
the premise) · prefix caching and paged KV are deferred, not rejected.

## Layout

```
braid/model/      the engine — layers, cache, graph capture, quantisation
braid/kernels/    CUDA sources + the JIT loader
braid/serve/      scheduling and serving
braid/bench/      benchmarks that produce publishable numbers
braid/reference/  slow, obviously-correct implementations the tests trust
scripts/          diagnostics, remote/provisioning shell, the PR eval bot
tests/            the gate; nearly all files need a real GPU
```

## Hardware

RTX 5090 (GB202, `sm_120a`) · 32 GB GDDR7 · 1,792 GB/s datasheet, **1,514–1,528 GB/s
measured** — all bandwidth-efficiency figures use the measured number · CUDA 12.8,
torch 2.11.

## Contributing

PRs are reviewed automatically and speedup claims are **measured, not trusted**: an eval
bot checks out your merged tree, runs the suite on a real RTX 5090, benches it against
`main` in the same session, and posts the table on your PR. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the rules and the workflow.

Every runtime PR is measured on a real 5090 by the eval bot; the verdict is
re-derived inside an Intel TDX machine and lands as an `eval:*` label, a
required `braid/eval` commit status, and an auditable receipt under
`refs/notes/braid-eval`. The full contract is in `CONTRIBUTING.md`.

## License

MIT — see [`LICENSE`](LICENSE).
