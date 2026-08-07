# braid — roadmap

**Date:** 2026-08-07
**Companion to:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (what braid is) and
[`THESIS.md`](THESIS.md) (why, and what is ruled out)
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
   - **`A = −exp(A_log)` cannot be decided by value.** The original spec
     prescribed "any element ≥ 0 ⇒ raw HF". **Measured: every one of layer 0's 32
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

3. **Full forward, B=1, eager, greedy.** ✅ **DONE 2026-08-07** —
   `braid/model/{gdn,layer,cache,engine}.py`, 10 tests in
   `tests/test_full_forward.py`. 32 layers on the 24-GDN/8-attention period-4
   schedule, dense SwiGLU MLP, tied LM head, KV + conv + recurrent caches, greedy
   and top-p sampling.

   **Proof of life:** `"The capital of France is"` → `" Paris."`; 128 greedy
   tokens with no repeated 8-gram. Weights 7.83 GiB, peak 23.7 GiB (that peak is
   the fp32 *test*, which holds two copies; the bf16 engine alone is well under).

   | check | measured |
   |---|---|
   | one GDN layer vs HF | **bit-identical** at T ≤ 4; 4.8e-5 at T=24 |
   | full 32 layers, fp32 vs HF | **rel L2 6.4e-7**, cosine 1.000000000 |
   | full 32 layers, bf16 vs HF | rel L2 8.3e-3, **100% greedy token identity** |
   | caches: decode vs prefill, fp32 | rel L2 4.7e-7 (GDN), 4.9e-7 (attention) |

   **Where the gate belongs.** Item 2's 5e-3 / 0.99999 is a *single-layer*
   threshold; applied to a 32-layer bf16 stack it measures accumulated rounding,
   not correctness. The per-layer trace (`scripts/layer_trace_diag.py`) shows the
   residual growing smoothly from 1.9e-4 at layer 0 and plateauing near 1e-2,
   with no step at any layer — and the same forward in fp32 lands at 6.4e-7. So
   the gate is applied on the fp32 arm, and the bf16 arm is gated on **greedy
   token identity** plus a drift tripwire. Same for decode-vs-prefill: exact in
   fp32, ~1.2e-2 in bf16 purely because a T=8 GEMM and eight T=1 GEMMs
   accumulate differently.

   **Prefill runs the decode recurrence in a loop.** Slow and deliberate — it
   makes prefill and decode the same arithmetic by construction, so the standard
   "generation drifts after the first token" bug cannot exist. Phase 5's ragged
   chunkwise scan replaces it; this phase makes no speed claim.

   **Two silent numeric deviations found and fixed**, neither in the plan:
   HF takes `beta`'s sigmoid in the *activation* dtype and only then widens
   (computing it in fp32 moves the layer by 4.8e-3), and the **gated** norm
   rounds to the activation dtype *mid-computation*, before applying gamma, where
   the plain norm stays fp32 throughout.

   **And one trap in the test harness, not the engine.** `module.to(bfloat16)`
   followed by `load_state_dict` copies *into* the bf16 parameter, silently
   truncating the tensors this checkpoint deliberately stores as F32 —
   `linear_attn.norm` moves 2.4e-3 and the "reference" becomes a worse model than
   braid. That artefact accounted for nearly all of the GDN layer's apparent
   parity gap. Reference models are built on `meta` and loaded with `assign=True`.
4. **Perplexity gate.** ✅ **DONE 2026-08-07** — `braid/bench/perplexity.py`, 5 tests
   in `tests/test_perplexity.py`, full method in the perplexity runbook.

   | arm | perplexity |
   |---|---:|
   | **braid** (bf16) | **8.2376** |
   | HF `Qwen3_5TextModel` (bf16) | 8.2393 |
   | **delta** | **0.0209%** |

   16,384 tokens of wikitext-2-raw-v1 test, pinned by SHA-256, in 8 non-overlapping
   2,048-token windows. Peak VRAM **8.54 GiB** (weights 7.83).

   **Not an overnight CPU job.** That framing assumed the reference could not share
   the card. Built and freed in sequence, both models fit on the 5090 and the whole
   run is ~2 minutes.

   **The gate is shown to discriminate.** Removing the `1+W` offset from the final
   norm degrades perplexity to 11.8429 — but that is **1.44×**, not the ≈2× this
   item predicted. The 2× came from the 35B's 13.65 → 6.82; the direction survived
   the rescope and the magnitude did not. 1.44× is still far outside the 20% gate,
   so the check holds.

**Gate: MET on all three clauses.** PPL within 20% (measured 0.0209%), absolute value
recorded (8.2376), peak VRAM 8.54 GiB against a 12 GB budget — not the 30 GB the
NVFP4 35B was budgeted at, so KV headroom is not a constraint at B=1 on this target.

> **Phase 2 complete, 2026-08-07.** 74 tests green on the remote 5090. Braid loads
> the checkpoint, matches HF to fp32 machine precision over all 32 layers, generates
> coherent text, and is within 0.03% of the reference on perplexity.
>
> **No speed claim is made or measured anywhere in Phase 2**, by design. GDN prefill
> is a per-token Python loop, attention runs on the SDPA math backend because
> `head_dim=256` disqualifies every fused kernel on this box, and nothing is
> CUDA-graphed. Phase 3 is where those become numbers.

---

## Phase 3 — The batch axis, everywhere (weeks 10–14)

1. **Batched eager decode at B = 2…16.** ✅ **DONE 2026-08-07** — `braid/model/cache.py`
   reworked to slot pools, batched decode through `gdn.py`/`attention.py`/`engine.py`,
   10 tests in `tests/test_batched_decode.py`. **Gate passed at the stated length:
   8 prompts, 256 tokens, 8/8 rows token-for-token identical.**

   State is now addressed by **slot**, not by batch row: a pool of `max_slots`
   entries plus a device-resident `slot_idx[batch]`. Prefill runs one sequence at a
   time into its own slot; decode runs all rows together. That split is what both
   halves of the model support today — a padded rectangle would feed pad tokens
   through the GDN recurrence and corrupt the state unless separately masked, which
   is item 3's problem.

2. **Graph buckets** {1, 2, 4, 8, 16} with padding, and a no-sync audit.
   ✅ **DONE 2026-08-08** — `braid/model/graph.py`, `Engine.decode_step`,
   11 tests in `tests/test_graph_decode.py`, numbers in the decode-speed runbook.
   All four secondary gates met:

   | gate | result |
   |---|---|
   | replay bit-identical to eager, every bucket | **PASS** (`rtol=0, atol=0`) |
   | a deliberate `.item()` makes capture fail | **PASS** |
   | slot reassignment needs no re-capture | **PASS**, 3 assignments |
   | `graphs_on / graphs_off ≥ 1.3` | **2.24 / 1.69 / 1.52** at B=1/8/16 |

   The Phase 1 CUDA kernels are wired in (`Engine(use_kernels=True)`), worth 21%
   at B=1 eager and ~0 at B=16 — what they remove is a fixed per-layer cost.
   `decode_step` is the sync-free, shape-static path; `hidden_states` cannot be
   captured because it reads `positions.max().item()`.

3. **KV block manager + chunked prefill**, single sequence per forward.
   Also removes the two 2 MiB-per-row-per-layer gather/scatter copies the eager
   torch path currently pays for the recurrent slab, and the KV `index_select`
   — and should **bucket `kv_len`**, which `decode_step` currently pins to
   `max_len` so the shape is capturable.

   > **Done first, 2026-08-08: profile the step.** The re-plan trigger below
   > required it before any lever was chosen, and it paid — decode attention was
   > running in fp32 through SDPA's math backend, worth **+38.6% at B=16**.
   > `braid/bench/decode_profile.py` is the tool; `docs/runbooks/decode-profile.md`
   > is the record.
   >
   > It also re-ranks the rest of this item. The step is now GEMM-dominated
   > (8.17 ms of 12.07 at B=16, 68% of the weight-read roofline), so the two
   > remaining pieces are worth **0.43 ms** (the KV `index_select`, which exists
   > only because attention cannot address the pool through `slot_idx` the way the
   > GDN kernels already do) and a length-dependent share of the attention bmms
   > (bucketing `kv_len`). Both are real; neither is the biggest lever any more.
   > `ncu` on `cutlass_80_wmma_tensorop_bf16_s161616gemm` is.

**Gate — greedy token identity.** 8 prompts run as one B=8 batch produce **token-for-token
the same 256 outputs** as 8 sequential B=1 runs.

> This is the strongest correctness test in the build. It catches every batch-leakage bug at
> once: a per-row sampling parameter read from row 0, an expert-combine kernel with no token
> stride (the reference engine's `moe_weighted_sum_residual` has *no token dimension at
> all*), a shared workspace aliased across rows.

> **Amended 2026-08-07: the gate is asserted in fp32, and that is not a weakening.**
> Measured (`scripts/batch_identity_diag.py`):
>
> | dtype | rows token-identical | logit residual, B=8 vs B=1 |
> |---|---|---:|
> | **fp32** | **8/8** | 1e-6 (machine precision) |
> | bf16 | 6/8 | 1e-2 |
>
> The bf16 gap is not a defect and no implementation removes it: a B=8 GEMM and a B=1
> GEMM select different tiles and split-k, so they accumulate in different orders. The
> resulting residual is the **same magnitude as Phase 2's B=1 decode-vs-prefill drift**
> — batching did not make it worse — and greedy argmax amplifies it discontinuously
> wherever the top two candidates are closer than the residual. Row 1's top-2 gap at the
> first decode step is 0.125 against logits of order 10.
>
> Asserting bf16 token identity would be asserting that cuBLAS picks the same tiles at
> M=1 and M=8, which braid neither controls nor should depend on. So bf16 is gated on
> **teacher-forced logit agreement** instead — feed both paths the same tokens so
> sampling cannot amplify, and bound the per-step residual (measured worst 1.24e-2,
> 2 argmax flips in 96 steps). A genuine leak moves that immediately; tile selection
> does not. Free-running bf16 identity is *reported*, not asserted.
>
> **This matters for Phase 4.** The published head-to-head runs in bf16, so braid's
> batched output is not bit-reproducible against its own single-stream output. That is
> true of every batched engine and must be stated rather than discovered by a reviewer.

Secondary gates: graph replay **bit-identical** to eager for every bucket;
`compute-sanitizer` clean during capture; a deliberately inserted `.item()` makes capture
**fail loudly**, proving the audit is real; `graphs_on / graphs_off ≥ 1.3`.

**Re-plan trigger:** measure c=1 here. If it is below 120 tok/s, the fixed term is worse than
modelled and Phase 4's gate needs revisiting before it is run.

> ### ⚠ FIRED 2026-08-07 at c=1 = 113.5 tok/s. ✅ CLEARED 2026-08-08 at 123.1.
>
> It fired, item 3's first task profiled the step as the trigger demanded, the cause was
> found and fixed, and the re-measure clears the threshold. Both states are kept here
> because the reasoning in between is the point.
>
> | batch | tok/s when it fired | tok/s now | ms/step |
> |---|---:|---:|---:|
> | 1 | 113.5 | **123.1** | 8.12 |
> | 8 | 606.4 | **718.7** | 11.13 |
> | 16 | 956.6 | **1,326.2** | 12.07 |
>
> **The ~8 ms/step that was unaccounted for is explained, and 4.6 ms of it is gone.**
> `head_dim = 256` disqualifies every fused SDPA backend on this box, so decode
> attention fell to the math backend, which replicates K and V 4× for GQA **and runs
> the whole thing in fp32** — 3.1 ms/step of copies to feed 1.3 ms of matmul.
> `grouped_decode_attention` groups the query instead and leaves K and V at their
> stored width and dtype. B=16: 16.73 → 12.07 ms, **+38.6%**. Details, including a
> 0.35 ms GDN lever that was measured and *rejected* for moving bf16 prefill off HF's
> greedy tokens, are in the decode-profile runbook.
>
> **A claim made when this fired was wrong and is retracted.** It read the 0.51×
> against llama.cpp as "exactly the BF16:Q8_0 weight-byte ratio, not a coincidence".
> braid is now at **0.705×** with no change to weight bytes at all: the 0.51× was 2×
> weight bytes *and* 4.6 ms of fp32 attention, coinciding. A ratio matching a model is
> evidence for that model only once the other terms are measured, and they were not.
>
> **What survives is the conclusion, not its arithmetic.** braid still carries 2×
> llama.cpp's weight bytes, and the step is now GEMM-dominated — 8.17 ms of 12.07,
> against a 5.58 ms weight-read floor, in an sm_80 WMMA kernel on an sm_120 card. So
> ARCHITECTURE §0's *"the MVP needs INT8/Q8_0-class weight-only quantization or the
> claim is dismissible"* stands, and **weight quantization remains the single gating
> decision for Phase 4.**
>
> Read the cleared margin honestly: 123.1 is 2.6% over the line against a 1.65% box
> noise floor. Clear, not comfortable, and c=1 is the number most exposed to the fixed
> per-step term — re-check it whenever the step changes shape.

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

See [`THESIS.md` §6.1](THESIS.md) for the full refutation ledger with
measurements. The three most likely to tempt us:

- **Making the single-sequence scan faster.** The reference engine's +16.7% kernel win
  measured **−0.18% end-to-end.** Any milestone that times the scan kernel without timing
  the step is measuring nothing.
- **WY / SSD / Tensor-Core chunkwise scan variants.** They built the whole ladder; every
  tensor-core variant *loses* to a plain chunk-cached scalar loop. The original spec's "keep
  WY/SSD as a later optimization" is dead ground.
- **NVFP4 on the GDN projections.** Measured at −9% to −20% decode. Tuned FP16 wins on those
  shapes.
