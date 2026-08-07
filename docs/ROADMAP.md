# braid — roadmap

**Date:** 2026-08-07
**Companion to:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
**Goal:** beat the reference engine on aggregate throughput serving a Gated-DeltaNet hybrid
at concurrency, on one RTX 5090.

---

## The number

**Gate: ≥800 tok/s aggregate at c=8, ≥1,000 at c=16**, on `Qwen3.6-35B-A3B-NVFP4`, against
the reference engine at its own default settings on the same box.

Absolute, not a multiplier — because our c=1 will be *below* their 320 and a multiplier
framing hides that. Bottom-up byte arithmetic puts a correct batched engine at
**660–910 tok/s at c=8**, so 800 sits inside the predicted band and is a real gate rather
than a formality. Their hybrid is **flat at ~317** at every concurrency, by construction.

---

## The schedule risk, stated first

This is the thing most likely to make the whole plan worthless, and it is not technical.

- The reference engine's fix for batched hybrid decode was independently sized by three
  adversarial reviewers at **1–2 weeks, 2–4 weeks, and 3–6 weeks**.
- Their data layout is **already pre-shaped for it**: `ssm_state_->init(n_ssm,
  config_.max_batch_size, …)`, a uniform `per_seq_bytes_` stride, a live `recurrent_slot_of_`
  map.
- **They merge 74 PRs/week** over the last 28 days (295 PRs since 2026-07-10); 65/week over
  56 days; 45/week lifetime (1,069 PRs since 2026-02-23). *Velocity is increasing, not
  decaying.*
- Braid Phase 0 → 4 is **14–17 weeks.**

The only reason the gap persists is doctrinal: `docs/GOAL.md:31` calls concurrency *"a
secondary metric: it may never be bought by regressing single-stream decode"*, and
`docs/roadmap.md:104` lists continuous batching as explicitly-not-a-gap. **They are not
blocked. They have decided this is not their race.**

Three consequences that shape every phase below:

1. **Publishing our curve is both the asset and the trigger.** When to publish is an
   ownership decision, not an engineering one, but the plan must not assume the gap survives
   contact.
2. **Prefer targets whose fix is not pre-shaped in their tree.** The prefill scan and the
   MoE-expert quantizer are months of work for them; the decode scan is weeks.
3. **Front-load the kill-tests.** Phase 0 and Phase 1 exist to spend two weeks, not four
   months, discovering the plan is wrong.

---

## Phase 0 — De-risk (week 1, six parallel spikes)

**No braid engine code is written in this phase.** Each spike can invalidate the plan, and
each is cheap. Run them concurrently.

| # | Spike | Cost | Kills the plan if… |
|---|---|---|---|
| 0.1 | ~~**NVFP4 on CUDA 12.8.**~~ **PARTLY RESOLVED 2026-08-07.** Phase 1's kernel builds and runs at `arch = sm_120a` under CUDA 12.8.61 — confirmed via `cuobjdump --dump-sass`. **The existing plan's "NVFP4 needs CUDA 13.x, which this box does not have" is wrong about the arch target.** Still to confirm: the `mxf4nvf4.block_scale` mma and `.b8`-routed `cvt.rn.satfinite.e2m1x2` specifically (expect `QMMA`, `F2FP.SATFINITE.E2M1`). | 1 h | …the FP4 instructions specifically are absent. The arch-conditional target is no longer in doubt. |
| 0.2 | **NVFP4 MoE grouped GEMM.** `pip install vllm` (cu128); probe `cutlass_scaled_fp4_mm` and `cutlass_moe_fp4` at M ∈ {1,8,16,32}, 256 experts. | 1 d | …nothing off-the-shelf works. This is **47% of the B=8 step's bytes**, and cuBLASLt has **zero** algorithms for grouped GEMM on sm_120. Borrowed = 200 lines; custom = ~900 lines and 3–4 weeks. **This single answer is the difference between a 10-week and a 16-week MVP.** |
| 0.3 | **The reference engine buildable on this box.** Sideload CUDA 13.3 + gcc-14 + CMake 3.31 on the Ubuntu 22.04 instance (no Docker available), build it, run its `--bench` mode. | 1 afternoon | …it cannot be built. Then the head-to-head is braid-on-native-Linux vs their published WSL2 numbers, which is disputable and undermines Phase 4 entirely. Fallback: provision a 24.04/26.04 instance. |
| 0.4 | **PyTorch VRAM overhead.** Load the 18 GB checkpoint into torch with CUTLASS/FlashInfer imported, one warm forward, read `torch.cuda.memory_reserved()` vs `nvidia-smi`. | 0.5 d | …PyTorch costs 1–2 GiB more resident than the reference engine's bare C++. c=16 has ~114 MiB of slack. **If this fails, the language choice is wrong and week 1 is when to know.** |
| 0.5 | **FlashInfer at `head_dim=256`, GQA 16/2, sm_120.** Build `BatchDecodeWithPagedKVCacheWrapper`, compare against gather+SDPA at B=4, ctx=1024. | 2 h | …nothing kills the plan; a gather+SDPA shim exists. But the answer sizes the CUDA budget. |
| 0.6 | **The reference engine's *actual* hybrid aggregate.** Their server on the 35B, **GIL-free multi-process** client, c ∈ {1,2,4,8,16}, sweeping `hybrid_decode_quantum` ∈ {8,32,128}. | 1 d | …nothing. But ~317 is *derived*, never published. This produces the real number we must beat **and** the fairness/throughput curve that is its own attack vector (§ Phase 5). |

Plus the environment gate and noise floor from the existing plan (Tasks 1–2), extended with
a 1 Hz clock/power sampler and the depressed-host classifier.

**Exit:** every spike answered and written down. 0.1 and 0.2 are hard blockers for the MVP's
schedule; 0.3 is a hard blocker for Phase 4.

---

## Phase 1 — MVP: the batched scan, standalone — ✅ **COMPLETE 2026-08-07**

> **This is the MVP.** The smallest artifact that can *kill the thesis*, built before any
> engine exists. It either proves aggregate scan throughput grows with batch or it does not.

**Result: gate PASSED on all four conditions.** 31 tests green on the remote 5090.
Full data in the scan-scaling and noise-floor runbooks.

| Gate | Required | Measured |
|---|---|---|
| Aggregate rows/s at B=8 vs B=1 | > 2× | **2.58×** |
| Achieved bandwidth vs roofline | within 20% | **104% of HBM at B=8** |
| No collapse past saturation | B=32 > 80% of peak | 91% |
| L2 cliff at B=2 | < 1.5× per-row cost | **0.59×** (improves) |
| Graph replay across slot reassignment | correct, < 1 ms | **10.3 µs**, vs their 10–20 ms re-capture |

**What it changed.** The scan **saturates HBM at B≈4** rather than scaling indefinitely. It
is no longer a serialization gate; it is a fixed bandwidth tax of **83 µs per sequence per
decode step**, which independently reproduces the 667 µs at B=8 estimated from their data.
The scan is at the wall and **cannot be made faster, only smaller** — which promotes FP16
`h_state` to the top open question. There is **no L2 cliff**; that premise confused a
per-layer slab (2 MiB) with a whole-sequence footprint (63.8 MiB).

Essentially the existing batched-GDN-decode-kernel plan, Tasks 1–6, with four corrections
and one addition:

1. `__launch_bounds__(128, 1)`, **never `(128, 2)`** — ptxas miscompiles that exact shape and
   produces garbage with correct math.
2. A `__syncthreads()` between the k-reduction's read of `s_reduce[0]` and the q-reduction's
   write of `s_reduce[d]`. **The reference engine ships this race.** (The plan's deterministic
   tree reduction already fixed the related `atomicAdd` issue.)
3. No `--use_fast_math` — it turns `rsqrtf` into an approximation and breaks fp32 parity at
   the asserted tolerances.
4. **Batch buckets stop at 16.** c=32 does not fit in VRAM and is throughput-pointless: the
   linear state term overtakes the fixed weight sweep around B=14–18.
5. **Added: the batched slotted conv1d** (`[S_max, 30, 8192, 4]` fp32, SiLU fused). Same
   `slot_idx` contract. Verify over **8 sequential steps with rotating slots** — a single
   step cannot catch a window-orientation error.

**Deliverables:** fp32 oracle (naive + vectorized, agreeing); batched decode kernel; one
captured graph replaying across three different slot assignments; the scan-scaling runbook.

**Gate — both conditions, not just the first:**

- Aggregate rows/s at B=8 **> 2× B=1**, still rising at B=16.
- Achieved DRAM bandwidth **within 20% of a raw `state.copy_()` of the same bytes.**

> The second condition matters more. The first only says we beat a bad baseline; the second
> says whether the scan is at its ceiling. It also answers the two questions that set the
> *only* free parameter in the throughput model: **what is the linear term** (it spans a 1.7×
> range = the difference between 727 and 1,250 tok/s at c=8), and **does the L2 cliff bite at
> B=2** (one sequence's 63.75 MiB state fits the 96 MB L2; two do not).

**Kill criterion:** if aggregate does not clear 2× at B=8, the scan was never the lock and
Phase 2 does not start.

---

## Phase 2 — A correct engine at B=1 (weeks 5–9)

Correctness only. **No speed claim is made or measured in this phase.**

> **Rescoped 2026-08-07 to the 4B.** This phase was written against
> `Qwen3.6-35B-A3B-NVFP4` — 40 layers, MoE, NVFP4, PPL ~6.8. The MVP target is
> `Qwen3.5-4B`: **32 layers (24 GDN + 8 gated attention), dense MLP, BF16, tied
> embeddings.** That deletes roughly half of item 1 (the whole NVFP4 quantiser
> contract) and all of the MoE work in item 3, and it makes the PPL figure in item 4
> a number to *measure*, not to expect. The 35B figures are retained in Phase 5+ where
> that checkpoint returns.

1. **Checkpoint loader.** ✅ **DONE 2026-08-07** — `braid/model/{config,loader}.py`,
   12 tests in `tests/test_loader.py`.

   The NVFP4 transforms this item was written around **do not exist on a BF16
   checkpoint**: no llm-compressor suffix renames, and no
   `tensor_scale = 1/weight_global_scale` reciprocal-direction trap (llm-compressor
   divides, Modelopt multiplies). They return with the 35B in Phase 5+.

   What replaced them, all specific to this checkpoint:

   - **It is a vision-language checkpoint. 738 tensors, of which braid runs 426** —
     the rest are a 24-block visual tower (297) and an MTP head (15), neither of
     which is in llama.cpp's GGUF. A loader that walks the index naively does not
     just waste VRAM, it makes the head-to-head compare two different models. The
     loader filters by prefix and *reports* what it dropped.
   - **`text_config` is nested.** Every shape lives under it; the top level carries
     the visual tower's `hidden_size: 1024`.
   - **`head_dim` is 256 with 16 heads over `hidden_size` 2560.** 4096 ≠ 2560 and
     256 ≠ 2560/16. `from_dict` refuses to infer it. On *this* checkpoint the
     inferred 160 happens to raise (8192 % 320 ≠ 0), but that is a divisibility
     accident, not a safety property.
   - **`A = −exp(A_log)` cannot be decided by value.** §5 of ARCHITECTURE.md
     prescribes "any element ≥ 0 ⇒ raw HF". **Measured: every one of layer 0's 32
     `A_log` entries is negative (−4.22 … −0.96)**, so that test reads
     "already transformed", skips the exp, and leaves the decay ~40× too fast. It
     collapses the state silently rather than diverging to the NaN the doc describes
     as the tell. braid keys the transform on the **source tensor name**, which is
     unambiguous for a safetensors load, and range-checks the result.
   - The `γ = 1 + W` offset survives unchanged, folded in fp32 at load, on the plain
     norms only — not on the gated `linear_attn.norm`.

2. **Single-layer HF parity.** ✅ **DONE 2026-08-07** for attention and MLP —
   `braid/model/{attention,mlp,norm}.py`, 10 tests in `tests/test_attention_parity.py`.
   The GDN layer was already closed by `tests/test_hf_parity.py`.

   Gate held at **rel L2 ≤ 5e-3, cosine ≥ 0.99999**. Measured, on real weights:

   | arm | rel L2 | cosine |
   |---|---:|---:|
   | attention, fp32 vs HF eager | **0.0** (bit-exact) | 1.0 |
   | attention, bf16 vs HF sdpa | 1.23e-4 | 0.999999992 |
   | MLP, both arms | **0.0** | 1.0 |

   Compare each dtype against the HF implementation that shares its numerics.
   braid-bf16 vs HF-*eager* reads 4.4e-3 / 0.999990 — inside the gate, but with 4e-7
   of margin — and that gap is HF eager taking its softmax in fp32 where SDPA does
   not. Gating on it would be gating on a schedule mismatch and would flake.

   Ablations confirming the gate discriminates (`scripts/parity_report.py`):
   flat-halves `[q|gate]` split **1.12**; rope over all 256 dims instead of 64
   **0.37**; q/k norm missing the `1+W` offset **0.78**.

   **The four open questions this item was to settle are closed**, three of them by
   the earlier GDN parity work: `[Q|K|V]` order (Q first), the `1+W` offset on
   `linear_attn.norm` (**no** offset — it is the gated form), the l2norm form
   (additive 1e-6). The `weight_scale_2` direction is NVFP4-only and moot here. The
   per-head-interleaved Q/gate layout is settled above.

   **New, not in the original plan:** `attn_output_gate: true` means `q_proj` emits
   `2 × n_heads × head_dim` split **per head**, and `partial_rotary_factor: 0.25`
   means only 64 of each 256-wide head rotates. Both are silent if missed.

   **Carried into Phase 3:** at `head_dim = 256` **every fused SDPA backend on this
   box declines** — flash, mem-efficient and cuDNN all report *"head_dim should be no
   more than 128"* and PyTorch falls back to the math backend, which materialises the
   full `[B, H, T, T]` score matrix. This is Phase 0 spike 0.5's question answered
   from the other side: the gather+SDPA shim exists but is not a fused path, so
   FlashInfer at `head_dim=256` is now load-bearing for Phase 3, not optional.

3. **Full forward, B=1, eager, greedy.** **32 layers** on the 24-GDN/8-attention
   period-4 schedule, dense SwiGLU MLP (`mlp_only_layers: []` — no MoE on this
   target), **tied** LM head, sampler.
   Proof of life: `"The capital of France is"` → `" Paris"`, 128 coherent tokens,
   degeneration battery clean at temp 0.7.
4. **Perplexity gate** against an HF bf16 CPU reference over a pinned ≥10k-token
   corpus (the box has 251 GB RAM — this is an overnight one-off). **Measure the
   absolute value; do not expect 6.8, which is the 35B's.** The diagnostic ratio
   still holds: **≈2× the reference PPL means the `1+W` offset is missing on the
   final norm.**

**Gate:** PPL within 20% of the reference, absolute value recorded. **Peak VRAM under
12 GB** — 8.8 GB of BF16 weights plus the recurrent pool — not the 30 GB the NVFP4 35B
was budgeted at, so KV headroom is not a constraint at B=1 on this target.

---

## Phase 3 — The batch axis, everywhere (weeks 10–14)

1. **Batched eager decode at B = 2…16.** The batch axis through the *whole* forward, not just
   the scan, plus the recurrent slot pool.
2. **Graph buckets** {1, 2, 4, 8, 16} with padding, and a no-sync audit.
3. **KV block manager + chunked prefill**, single sequence per forward.

**Gate — greedy token identity.** 8 prompts run as one B=8 batch produce **token-for-token
the same 256 outputs** as 8 sequential B=1 runs.

> This is the strongest correctness test in the build. It catches every batch-leakage bug at
> once: a per-row sampling parameter read from row 0, an expert-combine kernel with no token
> stride (the reference engine's `moe_weighted_sum_residual` has *no token dimension at
> all*), a shared workspace aliased across rows.

Secondary gates: graph replay **bit-identical** to eager for every bucket;
`compute-sanitizer` clean during capture; a deliberately inserted `.item()` makes capture
**fail loudly**, proving the audit is real; `graphs_on / graphs_off ≥ 1.3`.

**Re-plan trigger:** measure c=1 here. If it is below 120 tok/s, the fixed term is worse than
modelled and Phase 4's gate needs revisiting before it is run.

---

## Phase 4 — Serve, and the head-to-head (weeks 15–17)

1. **Scheduler + slot lifecycle + SSE server.** Continuous batching, admission-only capacity
   control, no preemption. A client disconnecting mid-generation must release **both** its KV
   blocks and its recurrent slot — the reference engine leaked slots on exactly this path
   until it added a dedicated reset.
2. **Bench harness** emitting aggregate tok/s, per-stream ITL p50/p90/p99, TTFT, own-peak
   VRAM, the clock/power verdict, and every SHA/version field.
3. **Reproduce the reference engine's baseline on this box** — single-stream within ±10% of
   its published 320, `speculative.ngram=false`, graphs on, clocks locked. **If it does not reproduce,
   characterise and publish the delta before running any comparison.** A baseline you cannot
   reproduce is not a baseline.
4. **The sweep:** c ∈ {1, 2, 4, 8, 16}, **ABBA order, 5 processes per arm per point**, on
   the reference engine's own `tools/agent_bench.py` where possible.

**GO / NO-GO.** Braid's median aggregate at c=8 must exceed theirs at c=8 by more than
the combined spread of the two arms, with every correctness pre-gate green — **and the c=1
row published unchanged even though we lose it.**

Fairness conditions that must be honoured or the result is not defensible:

- Speculation **off in both arms** (otherwise the benchmark measures draft accept rate).
- **Fresh, non-repeating prompts** — the reference engine has hybrid prefix caching (worth
  3.5× on multi-turn TTFT) and braid does not. Repeated prompts would unfairly penalise
  braid on TTFT.
- **GIL-free multi-process client.** They measured a threaded Python client inflating c=64
  TTFT by **19×**.

**If it does not clear, stop and re-plan rather than iterate.** The design's own falsification
clause applies.

---

## Phase 5+ — later, deliberately unscoped

Not planned in detail, and not started before Phase 4 clears. Listed in the order the recon
ranks them, which is **not** the order the original spec had.

| | Item | Why it ranks here |
|---|---|---|
| **5** | **Ragged batched prefill over the recurrent scan** | **Promoted from last to first.** `gdn_scan_chunkwise` is **44% of hybrid prefill**, on 32 of 170 SMs, and it is **structurally invisible to the reference engine** — their ncu capture regex matches none of the scan kernel names, so it has never had an arithmetic intensity or a lever entry in any of 12 committed runs. It is also their largest published deficit (1.55–2.17× behind llama.cpp on hybrid prefill), and prefill is the TTFT axis that agent fan-out actually pays. The original spec called prefill "not the metric"; that was wrong. |
| 6 | The fairness-quantum story | Free — it falls out of Phase 4. At c=8 a waiting stream's worst-case first-token delay under the reference engine's default is `(N−1) × 128 × 3.1 ms ≈ 2.8 s`; braid at B=8 gives every stream a token every ~1.1 ms. **Measurable against them today**, before braid exists. |
| 7 | FP8 KV + FP16 h_state | Worth 640 MiB and 510 MiB at c=16 — the difference between a razor-thin c=16 and a comfortable one, and FP16 state halves the term that caps the curve. **Neither is refuted.** FP8 KV's per-family gate on Qwen3.5/3.6 is simply *unrun*; FP16 state's only failure in the reference engine was a buffer overrun, and they run FP16 state on Mamba2 today. |
| 8 | FP8 row-scale GDN projections in the batched path | The reference engine's best hybrid decode lever (+19% at batch 1) is `M==1` only and evaporates at concurrency. ~10% of our batched step. **Per-row scales only** — one per-tensor scale over the fused pack costs +4% PPL. |
| 9 | Quantizer with MoE expert calibration | **Bigger and harder than the original spec states.** "Default to `--calib-groups BD`" is *undefined on our target*: group D matches `mlp.down_proj.weight` by exact name, which does not exist on a MoE layer, and group B is refused by fold-safety. On Qwen3.6-35B-A3B, `BD` resolves to nothing. This is genuinely new per-expert group modelling, ~1,200 lines and 4–6 weeks — and their own diagnosis warns the objective may be wrong ("a checkpoint whose weights each reconstruct better can still be a worse model"). |
| 10 | Pure-SSM (Mamba2 / Nemotron-H) | Highest ratio, lowest strategic value. The reference engine's CUDA graphs are structurally **off** for pure-SSM (`PureSsmLayers`: *"Mamba2 recurrent state is not graph-safe yet"*) stacked on the batch clamp — a 148 tok/s baseline where a comparable 30B reads 338. But it is one model family and they call the deficit arch-limited. |
| 11 | Qwen3.8-27B | Gated on the weights publishing (~2026-08-10) **and** on confirming it inherits the Qwen3.6 hybrid layer pattern. If it ships as pure attention, the thesis does not apply and it is dropped. Their registry tops out at `QWEN36_MOE`, so here we would start level. |

---

## Effort summary

| Phase | Weeks | Output |
|---|---|---|
| 0 — De-risk | 1 | Six answered spikes; the schedule fork resolved |
| **1 — MVP: batched scan** | **2–4** | **~1,100 lines. The thesis proved or killed.** |
| 2 — Correct engine, B=1 | 5–9 | ~2,500 lines. Parity and PPL. |
| 3 — Batch axis everywhere | 10–14 | ~2,000 lines. Greedy token identity. |
| 4 — Serve + head-to-head | 15–17 | ~1,900 lines. The number. |

**MVP engine total: ~7,500 lines if the MoE GEMM is borrowed, ~11,000 if it must be written.**
That fork is spike 0.2 and it is worth resolving on day one.

## What must not be attempted

See [`ARCHITECTURE.md` §9.1](ARCHITECTURE.md) for the full refutation ledger with
measurements. The three most likely to tempt us:

- **Making the single-sequence scan faster.** The reference engine's +16.7% kernel win
  measured **−0.18% end-to-end.** Any milestone that times the scan kernel without timing
  the step is measuring nothing.
- **WY / SSD / Tensor-Core chunkwise scan variants.** They built the whole ladder; every
  tensor-core variant *loses* to a plain chunk-cached scalar loop. The original spec's "keep
  WY/SSD as a later optimization" is dead ground.
- **NVFP4 on the GDN projections.** Measured at −9% to −20% decode. Tuned FP16 wins on those
  shapes.
