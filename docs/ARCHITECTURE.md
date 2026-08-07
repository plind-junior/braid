# braid — architecture

**Status:** proposed
**Date:** 2026-08-07
**Target hardware:** one NVIDIA RTX 5090 (GB202, `sm_120a`), 32 GB GDDR7, 1,792 GB/s datasheet / **1,508 GB/s measured**
**Reference competitor:** a mature single-GPU C++/CUDA engine @ `d8aabf88` — ~135k lines, 447 source files

This document supersedes the architecture section of the original batched-hybrid-decode
design spec. The thesis there is correct; **its stated mechanism is wrong**, and the correction
makes the case stronger. Everything below is cited against the reference engine's source
and its own committed measurement data at HEAD.

---

## 0. The thesis, replaced — read this before §1

> **2026-08-07, measured.** §1 below argued that the reference engine's inability to batch
> the recurrent scan is the opening. That is true *of that engine* and **false as a claim
> about the field**.
> **llama.cpp already batches GDN recurrent decode** and scales 11.7× to B=64 on
> Qwen3.5-4B, measured on this box 2026-08-07.
> The batched scan is table stakes, not IP.
>
> **The replacement thesis is stronger, because it is a reproducible number rather than a
> claim about someone else's kernel signatures:**
>
> > At concurrency, llama.cpp runs a GDN hybrid at **35% of the memory wall at B=16 and
> > 13% at B=64**, against 74% at batch 1. The reference engine does not compete on that
> > axis at all — every llama.cpp comparison it publishes is single-stream.
> > **braid targets the wall.**
>
> | parallel | llama.cpp | roofline | ms/step | % of roofline |
> |---:|---:|---:|---:|---:|
> | 1 | 250 | ~340 | 4.00 | 74% |
> | 16 | 1,880 | ~3,900 | 8.52 | 35% |
> | 64 | 2,928 | ~8,400 | 21.85 | 13% |
>
> **New target: beat 1,880 tok/s at B=16 and 2,928 at B=64 on Qwen3.5-4B**, publishing the
> batch-1 row unchanged even where we lose it.
>
> **Load-bearing consequence:** BF16 weights are 2× the bytes of llama.cpp's Q8_0, which
> loses at B≤8 and wins only above B=16. The MVP needs INT8/Q8_0-class weight-only
> quantization or the claim is dismissible. This is now the gating design decision.
>
> §1–§9 below remain accurate about **the reference engine** and about the mechanics of the
> scan. Read them as the engine-specific case, not as the market case.

## 1. The thesis, corrected (engine-specific — superseded by §0 as the competitive claim)

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
at c=16** (2.96×, `docs/BENCHMARKS.md:269`) — above vLLM's published 1,157 on the same
model class. That machinery already exists there and works.

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
read+write per step = 1.006 GB, which at the **measured** 1,508 GB/s is ~667 µs. The
weight sweep is read once regardless of B. Per-sequence attention and MoE-expert
divergence add real cost, so this is a ceiling, not a forecast:

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
  grounds and does not fit on VRAM grounds (§2). **Cap the graph buckets at 16.**

There is unmodelled upside: `attn_decode_paged` runs at **26.8 GB/s = 1.5% of roofline** on
the hybrid — purely latency-bound at M=1, and batching is the textbook fix for that. The
scan itself achieves 843 GB/s = 47% of peak *from 19% of the SMs*, so it should get more
efficient per sequence as B grows, until the L2 cliff.

### Cross-check: bottom-up byte arithmetic

The class-share model above is top-down. A bottom-up count of bytes moved per decode step at
B=8, 4k context, on the published checkpoint disagrees in an instructive way:

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

1. **The L2 cliff.** Per-sequence GDN state is 63.8 MiB; the 5090's L2 is 96 MB
   (`docs/sm120.md:17`). Their B=1 scan may be **entirely L2-resident**, and B≥2 is not.
   Our B=1 datapoint may therefore look artificially good and B=2 artificially bad. The
   scaling curve must be measured from B=1 to B=32, not extrapolated from two points.
2. **The reference engine's 320 is not a fixed target.** Their own ranked lever list has two
   unbuilt hybrid decode levers: `gemv_nvfp4` @ +21.4% and `attn_decode_paged` @ +13.7%
   (`docs/audit/roofline_2026_07_11.md:157,161`). Their hybrid c=1 could be ~430 without
   any batching work. Do not build a moat on 320.

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

## 2. Target model — exact shapes

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

**Per-sequence recurrent state:**

```
h_state     [32, 128, 128] fp32  = 2 MiB     per GDN layer   (head_dim fastest-varying)
conv_state  [8192, 4]      fp32  = 128 KiB   per GDN layer   (per-channel window contiguous)
                                   ─────────
            × 30 GDN layers, each sub-block 256B-aligned  =  63.8 MiB / sequence
```

At c=8 that is 510 MiB, at c=16 it is 1,020 MiB. **Recurrent state is not what caps
concurrency.** KV is cheap here too — only 10 attention layers with `n_kv_heads=2`, so
~20,480 B/token (`src/runtime/vram_budget.cpp:539-540`).

**What actually caps the reference engine's concurrency is a constant.** On every
native-NVFP4 model it adds `phase3_reserve = 10% of card + 1 GiB = 4,284.7 MiB` to the
weight-cache demand
(`src/runtime/vram_budget.cpp:371-372,388`). On the 35B, whose entire distributable budget
after weight upload is 6,083 MiB (`src/memory/plan.cpp:101-103`), that single safety margin
is **70% of the budget** — and it is why their KV pool comes out at 4,096 tokens on a first
start. A planner that charges exact bytes recovers ~4.2 GiB, which is the difference
between c=8 and c=16 being reachable. **Exact capacity planning is a component of this
architecture, not an optimisation.**

---

## 3. Components

Six. Each separately testable, separately scoreable, and marked MVP or later.

### 3.1 `braid/kernels` — the batched scan `[MVP]`

The core IP and the only place custom CUDA is unavoidable.

**Decode (`n_tokens == 1`).** Embarrassingly parallel over the batch — each sequence
applies one rank-1 update to its own state, with no cross-sequence dependency. Grid becomes
`(batch × n_heads)` instead of `(n_heads)`; on a 170-SM card that absorbs B≈5 essentially
free in SM terms.

State lives in one pool with a **device-resident indirection**:

```
h_pool    [max_slots, n_heads, state_size, head_dim]  fp32
slot_idx  [batch]                                     int32   ← read inside the kernel
```

Reading `slot_idx` from device memory rather than baking a base pointer is what makes one
captured CUDA graph valid for every assignment of sequences to slots, forever. The reference
engine cannot do this: `engine_scheduler.cpp:1968-1981` re-captures whenever the slot
changes, at a documented ~10–20 ms (`src/runtime/config.h:130-138`).

**Layout decision — layer-major, not sequence-major.** The reference engine indexes
`pool + seq_id*per_seq_bytes + layer*per_layer_bytes` (`src/memory/ssm_state.cu:76`), which
puts one layer's state for 16 sequences 63.75 MiB apart. We index layer-major so a batched
per-layer scan reads contiguously. They are locked out of this change because their snapshot
store's entry unit is exactly the per-sequence slab.

**Prefill.** Ragged chunkwise scan over packed variable-length sequences with a
`cu_seqlens` offset array, and — the actual lever — parallelism over *chunks* so the grid is
not stuck at 32 blocks. Deferred to Phase 5; correctness first.

### 3.2 `braid/model` — the runtime `[MVP]`

Deliberately thin. PyTorch control plane; custom CUDA only for the scan and the decode-step
graph. GEMM leans on CUTLASS via PyTorch, attention on FlashInfer.

Precedent that thin is enough: `naklecha/simple-llm` reaches 4,041 tok/s at batch 64 on an
H100 in ~950 lines of Python, ahead of vLLM's 3,846 on the same box. Python overhead
disappears inside a captured graph, and batch-1 is explicitly not our metric.

**Honest sizing: this is not 950 lines.** A working NVFP4 loader + 40-layer hybrid forward +
MoE + gated attention is ~3,000–4,000 lines of Python and ~800 lines of CUDA before it
serves anything.

### 3.3 `braid/engine` — scheduler and memory `[MVP for the batching path]`

- Paged KV for the 10 attention layers, per-sequence block tables.
- Fixed slot pool for recurrent state — O(1) in context length, so no paging needed.
- Prefill and decode **chunked and interleaved in one step**, not run to completion in phases.
- Decode graphs captured per batch bucket **(1, 2, 4, 8, 16 — not 32)**, replayed without
  re-capture.
- **Exact capacity planning** (§2): charge real bytes, not a 10%+1 GiB margin.

The reference engine's `src/runtime/` is 18,377 lines. We need maybe 1,500 of its equivalent
for the MVP, because we are not carrying LoRA, vision, constrained decoding, speculation,
prefix caching, model swap, suspend/resume, or five lifetime tiers.

### 3.4 `braid/bench` — the evidence harness `[MVP]`

The reference engine has no GPU CI *job*. `docs/audit/AUDIT_ARCH_2026_07_29.md:1348`: *"No
CI job verifies that any kernel produces the right numbers … WON'T FIX (owner decision
2026-08-03: no GPU runner)."* A one-line RMSNorm bug made the flagship NVFP4 hybrid family
8.6× worse in perplexity (65.1275 → 7.5302 after the fix), undetected until 2026-08-07.

**State this narrowly or it gets refuted.** It has 172 test files, 8 GTest binaries, 12
independent-oracle tests with stated tolerances, bit-identical-greedy checks across fresh
processes, a degeneration battery, and a 3%/5% perf gate medianed over 3 processes — all run
locally and human-initiated. And their GPU CI pipeline is *written and dormant*, gated on
one repo variable: `.github/workflows/ci.yml` `if: vars.HAS_GPU_RUNNER == 'true'`. It flips
on the hour a runner appears.

So this is **hygiene, not a moat**: we get automated per-commit per-layer HF parity from day
one because we already have the card and the key. We should not claim they are untested.

### 3.5 `braid/quant` — the quantizer `[later]`

Two gaps that survived refutation:

- **MoE experts are never calibrated in the reference engine** — `awq_plan.cpp:460` emits
  *"MoE experts NOT calibrated … they stay round-to-nearest."* On an A3B model the experts
  *are* the model.
- **The shipped calibration default is harmful at wide GQA.** Their own attribution on
  Qwen3-14B (`n_rep=5`): `--calib-groups ABCD` = **+2.68 PPL**, `BD` = **−0.1330** (their
  best measured configuration).

  **Caveat that kills the obvious plan:** "default to BD" is *undefined on our target*.
  Group D matches `mlp.down_proj.weight` by exact name (`awq_plan.cpp:347-349`), which does
  not exist on a MoE layer; group B is refused by the fold-safety check because the routed
  experts and the router both read `post_attention_layernorm` and neither is scaled. On
  Qwen3.6-35B-A3B, `BD` resolves to **nothing**. The MoE-expert calibration work is
  therefore genuinely new — per-expert group modelling that exists nowhere — not a
  default-flag change. Sized accordingly in the roadmap.

### 3.6 `braid/serve` — OpenAI-compatible HTTP `[MVP, minimal]`

`POST /v1/chat/completions` with SSE, `GET /v1/models`, `GET /health`. Nothing else. It
exists so the reference engine's own benchmark harness can drive us unmodified (§7).

---

## 4. The decode step, sublayer by sublayer

What braid must implement, per layer, at `n_tokens=1` per sequence. Traced from
`src/exec/executor_ssm_gdn.cu:260-637` and `executor_forward.cu:171-953`.

**GDN sublayer** (layers where `gdn_gate` exists — 30 of 40):

```
1. residual  ← hidden
2. hidden    ← RMSNorm(hidden, attn_norm_w, offset=+1.0)
3. packed    ← hidden @ gdn_input_packed          [12352, 2048]
                rows: [ qkv(8192) | gate/z(4096) | alpha(32) | beta(32) ]
4. conv_f32  ← causal_conv1d(packed[:8192], conv_state, W[8192,4], b) then SiLU
                conv_f32 per-token layout: [ Q(2048) | K(2048) | V(4096) ]   ← Q FIRST
5. y         ← gdn_scan(conv_f32, alpha, beta, A, dt_bias, h_state)          ← OUR KERNEL
6. y         ← RMSNormGated(y, ssm_norm_w[128], gate)      normalise THEN gate
7. hidden    ← y @ ssm_out                                  [4096, 2048]
8. hidden    ← hidden + residual
```

**Gated-attention sublayer** (10 of 40): standard paged decode, with one trap — the output
gate is **per-head interleaved**, `[Q_h0(256), Gate_h0(256), Q_h1(256), Gate_h1(256), …]`,
`q_out_dim = 2*nh*hd`. The reference engine records that the feature-concatenated reading
*"breaks all three staged hybrids"* (`executor_attention.cu:306-368`). Gate applied as
`ao *= sigmoid(gate)` after attention, before `o_proj`.

**MoE-FFN sublayer** (all 40): router → top-k → per-expert GEMV → weighted sum → shared
expert with per-token sigmoid gate → residual **last**. At decode the reference engine skips
the sort/permute entirely and indexes `base + expert_indices[k]*stride` directly
(`moe_routing.cu:720`). A batched design needs a permute they never launch — that is new
work, not ported work.

---

## 5. Parity contract

These are the things a reimplementation gets wrong silently. Every one is cited; every one
cost the reference engine a bug.

**The recurrence, exactly** (`src/compute/gdn.cu:148-174`), thread `d` owning column `d`:

```
kv[d]    = Σ_s H[s,d] · k̂[s]                 ← reduction on the UNDECAYED state
δ[d]     = (v[d] − g · kv[d]) · β
H[s,d]   = g · H[s,d] + k̂[s] · δ[d]
y[d]     = (Σ_s H_new[s,d] · q̂[s]) · rsqrt(head_dim)
```

The state update and the `y` accumulation **share one loop**. Splitting into
"update then read" is algebraically identical and not fp32-identical.

**Gates** (`gdn.cu:89-96`):

```
dt = α_raw + dt_bias
dt = (dt > 20.0f) ? dt : logf(1.0f + expf(dt))      ← logf(1+expf(·)), not log1pf
g  = expf(fmaxf(A · dt, −20.0f))                    ← A is ALREADY −exp(A_log_HF)
β  = 1/(1 + expf(−fmaxf(fminf(β_raw, 20), −20)))
```

`A = −exp(A_log_HF)` is applied on the **host at load**.

> **Corrected 2026-08-07, measured.** This paragraph previously said the decision to apply it
> is made from *values* — "any element ≥ 0 ⇒ raw HF" — rather than from dtype, because
> Qwen3.5-4B ships F32 raw `A_log`. The dtype half is right and **the value half is wrong on
> that very checkpoint**: every one of layer 0's 32 `A_log` entries is negative
> (−4.22 … −0.96), so the value test concludes "already transformed", skips the `exp`, and
> leaves `A = −2.7` where it should be `−0.067`. That is a ~40× *over*-fast decay: the state
> collapses toward zero silently, and the absmax tell below never fires. The value heuristic
> belongs to the GGUF path, whose conversion script may have folded the transform already.
> braid reads HF safetensors and keys the transform on the **source tensor name** (`A_log`),
> then range-checks that the result is finite and strictly negative.
> See `braid/model/loader.py` and `tests/test_loader.py`.

Getting the sign wrong in the other direction makes the state grow instead of decay; the
reference engine logged per-token absmax `0.04, 0.06, 0.40, 2.51, 110, 31680, inf`, then NaN,
then one token forever (#1282).

**L2 normalisation is clamped-rsqrt, not additive epsilon** (`gdn.cu:129,138`):

```
k_inv = rsqrtf(fmaxf(Σ k², 1e-12f))
```

`1e-12` is hardcoded and is *not* `rms_norm_eps`. It clamps the **sum of squares**, so the
effective floor on the norm is `1e-6`; `torch.nn.functional.normalize` clamps the *norm* at
`1e-12` — a 10⁶ difference in the degenerate case. The reference engine's comment records
that the additive form produced a 100–1000× wrong scale and broke Qwen3.6 at layer 1 heads
19/20/22/25/29.

**Head→group mapping is layout-dependent** (`gdn.cu:55`):
`g = grouped ? h/(n_heads/n_groups) : h % n_groups`. HF SafeTensors is **grouped** (`g = h/2`
for 32 heads over 16 groups); GGUF is tiled. Both are valid permutations of the same index
range, so a mismatch produces plausible-looking garbage, never a crash.

**Gated RMSNorm order — normalise first, then gate** (`gdn.cu:227-236`):

```
inv_rms = rsqrtf(Σy²/head_dim + eps)     ← eps INSIDE the sqrt, AFTER the mean
out     = y · inv_rms · γ[d] · silu(gate)
```

`γ` is `[head_dim] = [128]`, **shared across all 32 heads** — not a `[4096]` per-inner-dim
gamma. This is the opposite order from the Mamba2 path in the same file.

**The `+1` gamma offset — OPEN, and it is the highest-value open question here.**
Qwen3.5/3.6 SafeTensors store RMSNorm gammas as deltas (`real γ = 1 + W`), and
`arch_norm_offset = 1.0f` is threaded into `attn_norm`, `ffn_norm`, `q_norm`, `k_norm` and
(only since #1289) `out_norm`. `ssm_norm_w`, `ssm_conv1d_w` and `ssm_conv1d_b` go through a
path that hardcodes `weight_offset = 0.0f` (`weight_upload.cu:1385-1392`) — but **whether
that is correct or is the next bug is live over there right now.** Their last four commits are
`#1287`/`#1288`/`#1289`/`#1290`, all about exactly this, and the final-RMSNorm case was
worth **13.65 → 6.82 PPL**. Getting it wrong on `linear_attn.norm` costs a factor of two in
perplexity and nothing else visible. **Resolve empirically by single-layer HF parity, not by
reading their source.**

**`conv_f32` split order — DISPUTED, resolve before writing the loader.** Two independent
readings of the reference engine disagree: the scan kernel's index math says
`[Q(2048) | K(2048) | V(4096)]`,
Q first; another reading of the executor says `[x(4096) | B(2048) | C(2048)]` — but that is
the **Mamba2** path in the same file, not the GDN path. HF's
`torch.split(mixed_qkv, [key_dim, key_dim, value_dim])` supports Q-first. Get this wrong and
the model is **fluent and completely wrong**, with no crash and no obvious tell. It is
settled in an afternoon by single-layer HF parity with an A/B on the ambiguous axis.

**Other layout traps (these are settled):** conv1d weights are `[C, K]` with `K` contiguous
per channel; conv bias is added **after** the dot and **before** SiLU; SiLU is applied to
**all** of Q, K and V, not just V.

**State precision — say this precisely, because the two cases differ.**

- **FP8 E4M3 state is refuted.** From the reference engine: *"FP8 E4M3 (3-bit mantissa)
  amplifies these through
  the delta rule scan, causing degenerate output after ~50 special tokens in multi-turn
  chat"* (`engine_kv_cache_init.cpp:322-326`). Do not attempt it.
- **FP16 state is NOT refuted.** Their FP16 h_state failure was a **buffer overrun** — an
  allocation that halved the stride so the next layer read into it (`CHANGELOG.md:2891-2894`)
  — and they run FP16 h_state on Mamba2 models *today*. It is worth 31.9 MiB/sequence
  (510 MiB at c=16) and it **halves the DRAM traffic of the term that caps our curve.**
  This is one of the highest-value open questions in the whole plan (§10).

fp32 is mandatory for the MVP so that parity is unambiguous. Parity against a PyTorch fp32
oracle at `rtol=2e-5, atol=2e-6`; per-layer parity against HF at every commit.

---

## 6. sm_120a constraints

Absorbed from the reference engine's `.claude/skills/sm120-cuda-expert/`. Each cost them real time.

| Constraint | Consequence for braid |
|---|---|
| **`__launch_bounds__(HD, 2)` at HD=128 is a ptxas MISCOMPILE** — garbage output, correct math | Our scan kernel must use `__launch_bounds__(HD, 1)`. Directly on our path. |
| Opt-in shared memory is **~99 KB**, not H100's 228 KB | Query `sharedMemPerBlockOptin`; design tiles to ~97 KB. |
| **No TMA.** `cp.async.bulk` and `st.async .b128` to global are unavailable (re-probed on CUDA 13.3 / PTX ISA 9.3) | Use `cp.async.ca/cg.shared.global` at 16 B. Do not port Hopper pipelines. |
| `nvcuda::wmma` compiles but lowers to **HMMA**, not the FP8/FP4 pipes | Hand-write `mma.sync` with register-resident fragments, or don't claim tensor cores. |
| A 247-instruction survey across CUDA 13.2→13.3 flipped **0 instructions** | The ISA surface is silicon-fixed. Don't re-probe on every toolkit bump. |
| `cudaMallocAsync` inside a captured graph **crashes**; any D2H inside capture is an **IMA** | Pre-allocate every workspace; keep all args device-side under capture. This is exactly why `slot_idx` must be a device tensor. |
| CUTLASS NVFP4 on sm_120 is **non-deterministic under `cudaGraphExecUpdate`** | If we need bitwise reproducibility, keep NVFP4 GEMMs out of exec-update. |
| WDDM silently spills to host at ~0 MiB free — bandwidth 1,530 → 237 GB/s, decode −7× | Never size a large allocation from an *estimate* of another's future size. Leave ≥1 GiB free. |
| cuBLASLt returns **zero algorithms** for grouped GEMM on sm_120 | MoE grouped GEMM must be CUTLASS block-scaled or hand-rolled. |

---

## 7. Measurement contract

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

---

## 8. Risks

**Our c=1 is the whole risk.** Hybrid decode is GPU-busy-bound at 3.06 ms/step, so there is
no host overhead to reclaim, and their NVFP4 GEMVs are tuned. If braid's c=1 lands near 150
and we scale 3×, we finish at 450 — a win over their flat 320, but a narrow one that their
two unbuilt levers (+21.4%, +13.7%) would erase. **Mitigation:** measure c=1 at the end of
Phase 3 and re-plan if it is below 120.

**The L2 cliff at B≥2.** 63.8 MiB of state per sequence against 96 MB of L2 means B=1 may be
fully cache-resident and B=2 may not. Measure the full curve; do not extrapolate.

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
(§1), the MoE combine kernel (`moe_weighted_sum_residual` has *no token dimension at all*,
`moe_routing_permute.cu:224-241`), and the batch-1-gated spec-verify path.

**Silent correctness hazards we inherit if we copy carelessly.** The reference engine's
L2-norm block
reduction has a **data race** — after the k-reduction's `__syncthreads()`, every thread reads
`s_reduce[0]` and immediately writes `s_reduce[d]` with no barrier between
(`gdn.cu:129-131` and three sibling sites). Their recurrent-slot allocator has an aliasing
fallback that silently puts two live sequences on the same 63.8 MiB slab
(`engine_sampling_stop.cpp:256-262`). Ours must not.

---

## 9. Non-goals

Each excluded because the reference engine measured it and the ground is taken.

- **Single-stream batch-1 decode tok/s.** They are at ~80% of the weight-bandwidth wall.
- **A hand-written sm_120a SASS stack.** They surveyed and refuted, with measurements,
  NVFP4 GEMV tuning (6 approaches), FMHA rewrites, cuTile, ptxas autotuning, BitDecoding
  and FFN contextual sparsity. See §9.1.

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

### 9.1 Dead ground — measured and refuted, do not re-attempt

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

---

## 10. Open questions that gate work

Ranked by how much they move the plan. Each names the cheapest experiment.

1. ~~**What is the batched scan's achieved DRAM bandwidth at B = 1…32?**~~ **ANSWERED
   2026-08-07.** The scan reaches
   **40% of HBM at B=1 and 104% at B=8**, saturating the roofline at B≈4. Aggregate rows/s
   peaks at B=8 (2.58× B=1) and eases ~9% by B=32. **The linear term is now measured, not
   assumed: 83 µs per sequence per decode step** (30 layers × 2 MiB × 2 ÷ 1,528 GB/s),
   i.e. 665 µs at B=8 — which independently reproduces the 667 µs estimated from their data.
   Consequence: **the scan is at the wall and cannot be made faster, only smaller.** That
   promotes question 5 to the top of the list.
2. ~~**Does the L2 cliff bite between B=1 and B=2?**~~ **ANSWERED — no.** Per-row cost
   *improves* 6.78 → 3.97 µs. The premise was wrong: a sequence's *per-layer* slab is 2 MiB,
   not 63.8 MiB. The 63.8 MiB figure is all 30 layers, which are never live in one scan call.
   The real L2 effect is a 2.4× gap between an L2-resident microbenchmark and a
   production-realistic one — which is a *measurement* trap, not a hardware cliff, and is
   now controlled for.
3. **Does a PyTorch runtime fit?** The c=16 target has ~114 MiB of slack; PyTorch's context,
   caching-allocator fragmentation and FlashInfer/cuBLAS workspaces plausibly cost 1–2 GiB
   more resident than their bare C++. **Experiment:** load the weights into torch with
   CUTLASS/FlashInfer imported, run one warm forward, read `torch.cuda.memory_reserved()`.
   Half a day. **If this fails, the language choice is wrong and we need to know in week 1.**
4. **What is the reference engine's *actual* hybrid aggregate at c=8 and c=16?** The ~317
   figure is derived (single-stream minus rotation tax); they have never published it.
   **Experiment:** run their server on the 35B with a **GIL-free multi-process** client at
   c=1..16, sweeping
   `hybrid_decode_quantum` ∈ {8, 32, 128}. One day. Yields the exact number we must beat
   *and* the fairness/throughput tradeoff curve.
5. **Does FP16 h_state hold quality on the delta rule?** **Promoted to the top open question
   by the Phase 1 result.** The scan is bandwidth-saturated, so the only remaining lever on
   the linear term is halving its bytes: FP16 state is worth **~332 µs/step at B=8** and
   510 MiB at c=16. Not refuted (§5) — the reference engine's FP16-state failure was a
   buffer overrun and they run FP16 state on Mamba2 today.
6. **Does FP8 KV hold on Qwen3.5/3.6?** Worth 640 MiB at c=16 — the difference between a
   razor-thin c=16 and a comfortable one. They ship it default-on for four other families at
   +0.83–1.07% PPL and blocked this one only because the family declares no FP8 hint and the
   per-family gate is **unrun**. **Experiment:** run *the reference engine itself* with
   `--kv-fp8` on the 35B.
7. **How does the market score entries — aggregate, per-stream ITL, or batch-1?** Unresolved,
   and it is the largest non-technical risk. If batch-1 is the axis, the reference engine is
   at ~89–94% of the real bandwidth wall and this is the wrong race entirely.
   **Resolve before Phase 4.**
