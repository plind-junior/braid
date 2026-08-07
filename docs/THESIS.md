# braid — the thesis

**Status:** the case for building braid, and the ledger of what has been ruled out
**Date:** 2026-08-07
**Companion to:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (what braid *is*) and
[`ROADMAP.md`](ROADMAP.md) (what gets built when)

This document is an argument, not a map. It says why the project is worth doing, what
would prove it wrong, and which paths have already been measured and closed. Read it once
before starting a phase and once before publishing a number. **Nothing here tells you where
the code lives** — that is `ARCHITECTURE.md`.

Every claim is cited against either braid's own measurements on the remote 5090 or the
reference engine's committed source and measurement data at `d8aabf88`.

**Reference competitor:** a mature single-GPU C++/CUDA engine @ `d8aabf88` — ~135k lines,
447 source files.
**Target hardware:** one NVIDIA RTX 5090 (GB202, `sm_120a`), 32 GB GDDR7,
1,792 GB/s datasheet / **1,508 GB/s measured**.

---

## 1. The thesis

> **2026-08-07, measured.** §2 below argued that the reference engine's inability to batch
> the recurrent scan is the opening. That is true *of that engine* and **false as a claim
> about the field**.
> **llama.cpp already batches GDN recurrent decode** and scales 11.7× to B=64 on
> Qwen3.5-4B, measured on this box 2026-08-07.
> The batched scan is table stakes, not IP.

The replacement thesis is stronger, because it is a reproducible number rather than a claim
about someone else's kernel signatures:

> At concurrency, llama.cpp runs a GDN hybrid at **35% of the memory wall at B=16 and
> 13% at B=64**, against 74% at batch 1. The reference engine does not compete on that axis
> at all — every llama.cpp comparison it publishes is single-stream.
> **braid targets the wall.**

| parallel | llama.cpp | roofline | ms/step | % of roofline |
|---:|---:|---:|---:|---:|
| 1 | 250 | ~340 | 4.00 | 74% |
| 16 | 1,880 | ~3,900 | 8.52 | 35% |
| 64 | 2,928 | ~8,400 | 21.85 | 13% |

**Target: beat 1,880 tok/s at B=16 and 2,928 at B=64 on Qwen3.5-4B**, publishing the
batch-1 row unchanged even where we lose it.

**Load-bearing consequence:** BF16 weights are 2× the bytes of llama.cpp's Q8_0, which loses
at B≤8 and wins only above B=16. The MVP needs INT8/Q8_0-class weight-only quantization or
the claim is dismissible. This is the gating design decision, and it is what
`braid/bench/gemm_probe.py` and `braid/bench/fp8_probe.py` exist to answer.

---

## 2. The engine-specific case

Accurate about **the reference engine** and about the mechanics of the scan. Read it as the
engine-specific case, not the market case — §1 supersedes it as the competitive claim.

The original claim was: *the reference engine cannot batch the recurrent scan, and the scan
is the bottleneck.* The first half is confirmed. The second half is false, and repeating it
in public would be refuted in one profiler run.

**The GDN scan is 6.72% of hybrid decode time.** From the reference engine's own committed
nsys extract (`tools/roofline/history/raw/cf1b382a_20260711_193211/nvfp4-hybrid_tg256_r0.nsys_extract.json`,
class `gdn_scan`): `gdn_scan_fused_kernel` 4.87% + `ssm_conv1d_decode_f32_silu` 1.85%.
Per step that is ~0.20 ms of a 3.06 ms decode step.

So the scan is not a bottleneck. **It is a lock.**

| Kernel class | Share of hybrid decode | Batches? |
|---|---:|---|
| `gemv_nvfp4` | 34.0% | yes |
| `gemv_fp` (FP8 SSM-projection sidecar) | 22.7% | yes |
| `attn_decode_paged` | 14.0% | yes |
| `moe_routing` | 8.5% | yes |
| **`gdn_scan` + `ssm_conv1d`** | **6.7%** | **no — this is the lock** |
| `rmsnorm` | 3.7% | yes |
| `gdn_norm_gate`, `rope` | 2.2% | yes |

*Source: `docs/audit/roofline_2026_07_11.md:35-41` plus the raw extract for the two rows
their own report omits.*

93% of the step batches perfectly well. The reference engine proves it: on non-recurrent
models it takes Qwen3-Coder-30B-A3B-FP4 from **396 tok/s single-stream to 1,173 aggregate
at c=16** (2.96×, `docs/BENCHMARKS.md:269`) — above vLLM's published 1,157 on the same model
class. That machinery already exists there and works.

But a 6.7% kernel with no batch axis forces `decode_batch.resize(1)` on the whole engine
(`src/runtime/engine_scheduler.cpp:1475`), so none of the other 93% ever gets to batch on a
hybrid. **The tail wags the dog.** Removing the lock costs us the scan scaling linearly in
sequence count; it buys the entire weight sweep being amortised across the batch.

### The arithmetic

At B=1, the reference engine's hybrid step is 3.06 ms of kernel time (783.4 ms / 256 steps),
which implies 327 tok/s against 311–320 measured — so hybrid decode is **GPU-busy-bound, not
host-bound**. Do not plan a win around Python overhead; there isn't one to reclaim.

Decompose that step:

```
scan + conv (linear in B)   ≈ 0.20 ms
everything else (amortises) ≈ 2.86 ms
```

At B=8 the state traffic is the only term that grows: 8 sequences × 125.8 MB of h_state
read+write per step = 1.006 GB, which at the **measured** 1,508 GB/s is ~667 µs. The weight
sweep is read once regardless of B. Per-sequence attention and MoE-expert divergence add
real cost, so this is a ceiling, not a forecast.

Recomputed properly against their per-class data, the split is **~66–69% fixed** (weight
reads: `gemv_nvfp4` 34.31 + `gemv_fp` 22.66 + `moe_routing` 8.62 + `rmsnorm` 3.73) and
**~28% linear** (`gdn_scan` 6.72 + `attn_decode_paged` 13.27 + elementwise/unclassified
7.74). That gives a two-parameter model:

```
step(B) = fixed + linear·B  ms

central      2.0 + 0.86·B   ⇒ c=8:   901 tok/s   c=16: 1,014   asymptote 1,163
pessimistic  3.0 + 1.00·B   ⇒ c=8:   727         c=16:   842
optimistic   2.0 + 0.55·B   ⇒ c=8: 1,250         c=16: 1,481
```

against a **flat ~317** at any concurrency. **Even the pessimistic floor is 2.3×.**
The central case converges with their own independently measured dense batching factor
(2.96× at c=16, and ~3× at c=15 in `docs/audit/PERF_LOG.md:604-619`), which is a good sign
that the model is not fantasy.

Two consequences that must go into the plan:

- **The target is an absolute number, not a multiplier.** "3× the reference engine" hides
  the real risk, which is that `braid_c1` will be below their 320 — they run the scan at
  47% of HBM peak and have six refuted GEMV tuning campaigns behind them.
  **Gate: ≥800 tok/s at c=8 and ≥1,000 at c=16.**
- **The curve saturates near c=16.** At B×133.7 MB of state traffic per step, the linear
  term overtakes the fixed weight sweep around B=14–18. c=32 is pointless on throughput
  grounds and does not fit on VRAM grounds (§3). **Cap the graph buckets at 16.**

There is unmodelled upside: `attn_decode_paged` runs at **26.8 GB/s = 1.5% of roofline** on
the hybrid — purely latency-bound at M=1, and batching is the textbook fix for that. The
scan itself achieves 843 GB/s = 47% of peak *from 19% of the SMs*, so it should get more
efficient per sequence as B grows, until the L2 cliff.

### Cross-check: bottom-up byte arithmetic

The class-share model above is top-down. A bottom-up count of bytes moved per decode step at
B=8, 4k context, on the published 35B checkpoint disagrees in an instructive way:

| Tensor class | Bytes/step @ B=8 | Amortises with B? |
|---|---:|---|
| Routed experts, NVFP4, ~56 unique of 256 | 3.50 GB | **barely** |
| LM head, BF16 `[248320, 2048]` | 1.02 GB | yes |
| GDN in/out projections, BF16, 30 layers | 1.01 GB | yes |
| GDN recurrent state, read+write | 1.02 GB | **no — linear in B** |
| KV read, 20 KiB/tok × 4096 × 8 | 0.64 GB | **no — linear in B** |
| Attention projections, NVFP4, 10 layers | 0.14 GB | yes |
| Shared experts, NVFP4, 40 layers | 0.06 GB | yes |
| **Total** | **≈ 7.4 GB** | |

**The routed-expert term is the whole game, and it is the one that does not amortise.** With
top-8 of 256 experts, B=1 touches 8 experts but B=8 touches
`E[unique] = 256·(1−(1−1/256)^64) ≈ 56` — 7× the weights for 8× the tokens, i.e. only ~12%
saved per token. An A3B MoE is structurally worse at batching than a dense model, and the
model must not pretend otherwise.

At 1,508 GB/s measured and the reference engine's own observed efficiency band (30–41% of
roofline for MoE-shaped decode), a B=8 step lands at 8.8–12.0 ms → **660–910 aggregate
tok/s**. The same arithmetic at B=1 gives ≈2.9 GB/step → 520 tok/s at 100% efficiency,
against their measured 320 = **62%**, which calibrates the band.

**Take the bottom-up number as the planning figure: 660–910 tok/s at c=8 — 2.1–2.8× the
reference engine.**
It is more conservative than the class-share model and it is built from bytes rather than
from percentages. The gate stays at ≥800 at c=8 — inside the band, so it is a real gate.

### Two hazards this arithmetic hides

1. ~~**The L2 cliff.**~~ **Resolved 2026-08-07 — there is no cliff.** The premise confused a
   sequence's *per-layer* slab (2 MiB) with its whole-model footprint (63.8 MiB on the 35B,
   51 MiB on the 4B), and the layers are never live in one scan call. Per-row cost *improves*
   from B=1 to B=2, 6.78 → 3.97 µs. What remains is a 2.4× gap between an L2-resident
   microbenchmark and a production-realistic one — a *measurement* trap, now controlled for
   in `braid/bench/scan_scaling.py`.
2. **The reference engine's 320 is not a fixed target.** Their own ranked lever list has two
   unbuilt hybrid decode levers: `gemv_nvfp4` @ +21.4% and `attn_decode_paged` @ +13.7%
   (`docs/audit/roofline_2026_07_11.md:157,161`). Their hybrid c=1 could be ~430 without any
   batching work. Do not build a moat on 320.

### The second, larger, undefended target

**`gdn_scan_chunkwise_kernel` is 40.13% of hybrid *prefill*** — 122.3 ms of a 304.8 ms
pp512 window, the single largest kernel class in that cell. The scan class totals 43.96%.
Their published roofline table for that cell lists five kernels summing to 46.1% and
**does not include it at all** (`docs/audit/roofline_2026_07_11.md:30-34`).

The reason is structural: their ncu capture regex is
`"gemv|nvjet|device_kernel|paged_attention|rmsnorm|rope|qknorm|write_kv|argmax|topk|softmax|apply_|residual"`
(`tools/roofline/config.json:176`), and report rows are built only from ncu-captured
kernels. **`gdn_scan_*` matches nothing in it.** The scan has never had an arithmetic
intensity, an achieved GB/s, a %-roofline, or an occupancy number computed for it — in any
of their 12 committed runs.

And it runs on a grid of `n_heads` = **32 blocks on a 170-SM card** (`src/compute/gdn.cu:21`,
`gdn_scan.cu:508`), at 8.33% occupancy, serialising over tokens inside one block per head.
81% of the GPU is idle during 44% of prefill.

Prefill is TTFT. An agent session pays TTFT on every one of 20–100 tool calls. This is a
bigger and better-defended win than decode, and it is invisible to their own instruments.
It is Phase 5, not because it is smaller, but because decode is where the head-to-head
number lives.

---

## 3. The Phase 5+ target model and its capacity arithmetic

braid's MVP target is `Qwen3.5-4B` — see [`ARCHITECTURE.md`](ARCHITECTURE.md) for its
shapes. The *competitive* target, which returns in Phase 5+, is
`mmangkad/Qwen3.6-35B-A3B-NVFP4`, 18 GB, arch `QWEN36_MOE`. The reference engine decodes it
at 320 tok/s. Every constant below is confirmed against its loader and kernels.

| Property | Value | Source |
|---|---|---|
| Layers | 40 = **30 GDN + 10 gated attention**, period 4 | `executor_workspace_config.cu:305-314` |
| Layer pattern | `3 × (GDN → MoE-FFN) → 1 × (gated attn → MoE-FFN)` | dispatched per-layer by weight presence, not by index |
| `d_model` | 2048 | `hf_config_loader.cpp:416-490` |
| Attention | `n_heads=16`, `head_dim=256`, `n_kv_heads=2` | — |
| GDN heads / groups | `n_heads=32`, `n_groups=16` (2 heads per group) | `linear_num_value_heads` / `linear_num_key_heads` |
| GDN dims | `head_dim=128`, `state_size=128`, `inner=4096` | `linear_value_head_dim` / `linear_key_head_dim` |
| Conv | `conv_channels=8192`, `conv_kernel=4` | `4096 + 2×16×128` |
| MoE | 256 experts, per-expert intermediate 512, plus an always-on shared expert | `executor_workspace_config.cu:78-93` |

**Its GDN block is dimensionally identical to the 4B's**, which is why the 4B is the MVP
target: the decode kernel transfers unchanged while the MoE and NVFP4 paths stay off the
critical path.

**Per-sequence recurrent state on the 35B:**

```
h_state     [32, 128, 128] fp32  = 2 MiB     per GDN layer
conv_state  [8192, 4]      fp32  = 128 KiB   per GDN layer
                                   ─────────
            × 30 GDN layers, each sub-block 256B-aligned  =  63.8 MiB / sequence
```

At c=8 that is 510 MiB, at c=16 it is 1,020 MiB. **Recurrent state is not what caps
concurrency.** KV is cheap here too — only 10 attention layers with `n_kv_heads=2`, so
~20,480 B/token (`src/runtime/vram_budget.cpp:539-540`).

**What actually caps the reference engine's concurrency is a constant.** On every
native-NVFP4 model it adds `phase3_reserve = 10% of card + 1 GiB = 4,284.7 MiB` to the
weight-cache demand (`src/runtime/vram_budget.cpp:371-372,388`). On the 35B, whose entire
distributable budget after weight upload is 6,083 MiB (`src/memory/plan.cpp:101-103`), that
single safety margin is **70% of the budget** — and it is why their KV pool comes out at
4,096 tokens on a first start. A planner that charges exact bytes recovers ~4.2 GiB, which
is the difference between c=8 and c=16 being reachable. **Exact capacity planning is a
component of this architecture, not an optimisation.**

---

## 4. Measurement contract

Adopted wholesale from the reference engine's own `benchmark-cuda` skill and
`scripts/verify.sh`, so a head-to-head cannot be disputed on methodology.

- **Precondition:** no other GPU process. `nvidia-smi --query-compute-apps=pid` empty.
- **Environment:** `CUBLAS_WORKSPACE_CONFIG=:4096:8`, warm clocks >1 s before timing.
- **Gate measurement:** 3 independent processes × 3 reps, median across processes, and
  **print the spread** `(max−min)/min×100`.
- **Host-health classifier**, sampled at 1 Hz concurrently with every timed run, first 2
  samples dropped: depressed if `mem_med < 13801 MHz` or `pwr_max < 400 W` or
  `sm_med < 2000 MHz`. Numbers taken on a depressed host are reported as such or discarded.
- **Use tg256, never pp512, for A/B.** pp512 varies up to **2.6×** across container restarts
  from cuBLAS autotuning.
- **Always profile with CUDA graphs ON.** Graphs-OFF kernel-time sums run ~1.8× the real
  step and produce a systematically wrong lever list.
- **Correctness gates before any timing is reported.** Per-layer HF parity, plus a
  degeneration suite: no token run > 6, no 4-gram repeated ≥ 4× consecutively,
  `unique_ratio > 0.25`, clean stderr.
- **Concurrency sweeps at c ∈ {1,2,4,8,16,32} report aggregate tok/s *and* per-stream ITL
  p50/p90/p99.** Aggregate is exactly N ÷ per-stream ITL; quoting them as independent wins
  is how a 6× error gets made.
- **The head-to-head runs on the reference engine's own harness**,
  `tools/agent_bench.py --concurrency 1,4,16`, driving both engines through the same
  OpenAI-compatible endpoint. It is their scorer.

The significance threshold is **measured, not chosen** — `braid/bench/noise_floor.py`
exists because a 2% gate against a 10% noise floor is a coin flip.

---

## 5. Risks

**Our c=1 is the whole risk.** Hybrid decode is GPU-busy-bound at 3.06 ms/step, so there is
no host overhead to reclaim, and their NVFP4 GEMVs are tuned. If braid's c=1 lands near 150
and we scale 3×, we finish at 450 — a win over their flat 320, but a narrow one that their
two unbuilt levers (+21.4%, +13.7%) would erase. **Mitigation:** measure c=1 at the end of
Phase 3 and re-plan if it is below 120.

**The reference engine closing it is the DOMINANT risk, and it is a schedule risk, not a
technical one.**
Three independent adversarial verifiers sized their fix at 1–2 weeks, 2–4 weeks and 3–6
weeks. The decode half is: add `const int* ssm_seq_slots` to `InferenceState`, give
`gdn_scan_fused_kernel` a `dim3(n_heads, n_seqs)` grid and a slot-indexed `h_state` offset,
re-key the graph pool. **Their data layout is already pre-shaped for it** —
`ssm_state_->init(n_ssm, config_.max_batch_size, …)` (`engine_kv_cache_init.cpp:565`), a
uniform `per_seq_bytes_` stride, and a live `recurrent_slot_of_` map. **They merge ~73
PRs/week.** Braid Phase 0→4 is 3–6 months.

**The only moat is doctrinal, and it is explicit in their own docs.** `docs/GOAL.md:31`
declares concurrency *"a secondary metric: it may never be bought by regressing
single-stream decode"*, and `docs/roadmap.md:104` lists continuous batching as
explicitly-not-a-gap. They are not blocked; they have decided this is not the race.

The strategic consequence has to be stated plainly: **publishing braid's curve is
simultaneously our strongest asset and the trigger that removes the moat.** When to publish
is a real decision with a real cost, and it belongs to whoever owns the project, not to this
document. The engineering answer is the only one available to us — be fast, and pick targets
(prefill, quantizer) whose fixes are *not* pre-shaped in their tree.

**The expensive parts for them, which is where our durable lead lives:** the prefill scan
(§2), the MoE combine kernel (`moe_weighted_sum_residual` has *no token dimension at all*,
`moe_routing_permute.cu:224-241`), and the batch-1-gated spec-verify path.

**Silent correctness hazards we inherit if we copy carelessly.** The reference engine's
L2-norm block reduction has a **data race** — after the k-reduction's `__syncthreads()`,
every thread reads `s_reduce[0]` and immediately writes `s_reduce[d]` with no barrier
between (`gdn.cu:129-131` and three sibling sites). Their recurrent-slot allocator has an
aliasing fallback that silently puts two live sequences on the same 63.8 MiB slab
(`engine_sampling_stop.cpp:256-262`). Ours must not — see the parity contract in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 6. Non-goals

Each excluded because the reference engine measured it and the ground is taken.

- **Single-stream batch-1 decode tok/s.** They are at ~80% of the weight-bandwidth wall.
- **A hand-written sm_120a SASS stack.** They surveyed and refuted, with measurements,
  NVFP4 GEMV tuning (6 approaches), FMHA rewrites, cuTile, ptxas autotuning, BitDecoding
  and FFN contextual sparsity. See §6.1.

  > **Correction to the original spec.** It claimed *"2:4 sparse FP4 does not exist on
  > consumer Blackwell — `ptxas` refuses `kind::mxf4nvf4` with `.sp`."* **This is false.**
  > `tools/analysis/ptx_mma_survey.sh:198-199` tests
  > `kind::f8f6f4.sp::ordered_metadata.m16n8k64.row.col.f32.e2m1.e2m1.f32` — FP4 2:4
  > sparse — and their archived survey records that *and* `sparse mxf4nvf4 4X K=128 ue4m3`
  > as ptxas-**accepted** on `sm_120a`. The spec conflated block-scaled `mxf4nvf4` with
  > plain `f8f6f4`. Do not repeat the claim in public. It remains out of scope for us, but
  > on effort grounds, not availability grounds.
- **Speculative decoding as the headline.** Four drafters already ship, including a trained
  MTP head at 85%+ accept. Re-measured 2026-07-27 as still net-negative (−7% on reasoning).
- **Beating NVIDIA ModelOpt at general-purpose quantization.** Their in-tree quantizer
  already does (9.9252 vs 10.0301 PPL on Qwen3-14B).
- **Multi-GPU, tensor parallelism, CPU offload.** One card is the whole premise.
- **Prefix caching, vision, LoRA, constrained decoding, model swap.** All shipped in the
  reference engine, none on the critical path to the number we are chasing.

### 6.1 Dead ground — measured and refuted, do not re-attempt

Each of these is something a reasonable person would try. The reference engine tried it and
published the number. Re-attempting any of them is pure schedule loss.

| Refuted | Measurement |
|---|---|
| **Making the single-sequence scan faster** | +16.7% kernel microbench → **−0.18% / −0.11% / −0.35% end-to-end**. Any milestone that measures scan kernel time without measuring end-to-end is measuring nothing. |
| WY-representation and Tensor-Core chunkwise scan (Phase 2a/2b/2c) | Every TC variant **loses** to a plain chunk-cached scalar loop: sequential 1.567 µs/tok, chunk-cached 1.343 (the winner). The original spec's "keep WY/SSD as a later optimization" is dead ground. |
| Split-K / cross-block parallel scan **at decode** | Cross-block sync 4 × 7 µs = 28 µs against a 5.9 µs decode kernel = **475% overhead**. |
| NVFP4 on GDN `in_proj`/`out_proj` | **−9% (Nemotron) to −20% (Qwen3.6)** decode. Tuned FP16 GEMV hits 70–81% of HBM on wide GDN-output shapes and beats the NVFP4 one. |
| One per-tensor FP8 scale over the fused GDN input pack | **+4% PPL** — the pack is heterogeneous (q/k/v/gate/beta row groups) and one amax is dominated by the largest. Per-**row** scales are PPL-flat (8.021 → 8.012). |
| FP4/NVFP4 inside attention math (QK^T or PV) | Format-intrinsic, refuted 4×: e4m3-QK PPL **5722** vs 6.12; MXFP4 blockscale 5.0× slower than FA2. |
| NVFP4 decode GEMV micro-optimization | 6 approaches; runs at 64–73% of HBM peak. One attempt caused **−41%** decode (157.71 → 92.28 tok/s) by blowing L1. |
| Launch-latency / kernel-fusion levers under graphs+PDL | Refuted **by class**: a bit-identical fusion removing a 6.9%-share launch measured **0%** e2e. |
| cuTile / CUDA Tile for attention | Correct autotuned cuTile FA2 = 26.5 eff-TFLOPS = **3.2% of roofline**. |
| ptxas / compiler autotuning | Search space is flat on these hotspots; all sweep points within **±0.4%**. |
| Tensor-core KV decode attention (BitDecoding-class) | WMMA 118.4 µs vs scalar 119.0 µs at 16k ctx — identical. Decode is weight-bound. |
| FFN contextual sparsity | Real sparsity 25–52%, **+0–1% e2e** — wallclock is set by the slowest warp. |
| Host-RAM / KV spill tier | Scoped 2026-08-01, verdict "do not build it": no reproducible trigger, **6.5× bandwidth cliff** (1,531 → 237 GB/s). |
| `torch._weight_int8pack_mm` as the cheap INT8 weight path | Measured **5–50× slower than bf16** on this box, and its cost grows linearly in M — it is doing M separate GEMVs. Recorded in `braid/bench/gemm_probe.py`. |

---

## 7. Open questions that gate work

Ranked by how much they move the plan. Each names the cheapest experiment.

1. ~~**What is the batched scan's achieved DRAM bandwidth at B = 1…32?**~~ **ANSWERED
   2026-08-07.** The scan reaches **40% of HBM at B=1 and 104% at B=8**, saturating the
   roofline at B≈4. Aggregate rows/s peaks at B=8 (2.58× B=1) and eases ~9% by B=32.
   **The linear term is now measured, not assumed: 83 µs per sequence per decode step**
   (30 layers × 2 MiB × 2 ÷ 1,528 GB/s), i.e. 665 µs at B=8 — which independently
   reproduces the 667 µs estimated from their data. Consequence: **the scan is at the wall
   and cannot be made faster, only smaller.** That promotes question 5 to the top.
2. ~~**Does the L2 cliff bite between B=1 and B=2?**~~ **ANSWERED — no.** Per-row cost
   *improves* 6.78 → 3.97 µs. See §2's hazard 1.
3. **Does a PyTorch runtime fit?** The c=16 target has ~114 MiB of slack; PyTorch's context,
   caching-allocator fragmentation and FlashInfer/cuBLAS workspaces plausibly cost 1–2 GiB
   more resident than their bare C++. **Experiment:** load the weights into torch with
   CUTLASS/FlashInfer imported, run one warm forward, read `torch.cuda.memory_reserved()`.
   Half a day. **If this fails, the language choice is wrong and we need to know in week 1.**
4. **What is the reference engine's *actual* hybrid aggregate at c=8 and c=16?** The ~317
   figure is derived (single-stream minus rotation tax); they have never published it.
   **Experiment:** run their server on the 35B with a **GIL-free multi-process** client at
   c=1..16, sweeping `hybrid_decode_quantum` ∈ {8, 32, 128}. One day. Yields the exact
   number we must beat *and* the fairness/throughput tradeoff curve.
5. **Does FP16 h_state hold quality on the delta rule?** **Promoted to the top open question
   by the Phase 1 result.** The scan is bandwidth-saturated, so the only remaining lever on
   the linear term is halving its bytes: FP16 state is worth **~332 µs/step at B=8** and
   510 MiB at c=16. **Not refuted** — the reference engine's FP16-state failure was a buffer
   overrun (`CHANGELOG.md:2891-2894`), and they run FP16 h_state on Mamba2 models today.
   (FP8 E4M3 state *is* refuted: `engine_kv_cache_init.cpp:322-326`.)
6. **Does FP8 KV hold on Qwen3.5/3.6?** Worth 640 MiB at c=16 — the difference between a
   razor-thin c=16 and a comfortable one. They ship it default-on for four other families at
   +0.83–1.07% PPL and blocked this one only because the family declares no FP8 hint and the
   per-family gate is **unrun**. **Experiment:** run *the reference engine itself* with
   `--kv-fp8` on the 35B.
7. **How does the market score entries — aggregate, per-stream ITL, or batch-1?** Unresolved,
   and it is the largest non-technical risk. If batch-1 is the axis, the reference engine is
   at ~89–94% of the real bandwidth wall and this is the wrong race entirely.
   **Resolve before Phase 4.**
