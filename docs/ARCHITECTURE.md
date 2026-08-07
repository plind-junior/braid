# braid — architecture

**What this document is:** a map of the code as it exists today. Present tense throughout.
If the code and this document disagree, the code wins — but the disagreement is a bug in
this document and should be fixed.

**What this document is not:** the argument for why braid should exist, what the competition
is doing, or which optimisations have been ruled out. That is [`THESIS.md`](THESIS.md).
The build order and the gates are [`ROADMAP.md`](ROADMAP.md).

Sections open in plain language and then get specific. If a term is unfamiliar, the
[glossary](#glossary) at the end defines it. **Section 9 is the "you are here" —** it lists
what is actually built and what is still a stub.

---

## 1. What braid is, in plain terms

braid is a program that runs a large language model on one graphics card and answers
**many people at once**.

Here is the problem it exists to solve. A language model is a very large table of
numbers — 4 billion of them for the model braid runs, taking up about 8.4 gigabytes.
Producing a single word of an answer means reading essentially that entire table and doing
arithmetic with it. Then reading the whole table again for the next word. And again.

The graphics card can read about 1.5 trillion bytes per second. Divide that by the 8.4
billion bytes of the table and you get roughly **180 words per second, and no amount of
clever programming beats that number** — for one user. The card is not thinking too slowly;
it is *reading* as fast as it physically can.

The way out is that the reading can be *shared*. If eight people ask questions at the same
moment, the card can read the table once and use it to advance all eight answers together.
The reading was the expensive part, and now eight people paid for it instead of one. Do it
for sixteen people and the arithmetic barely changes but the total output roughly
sixteen-folds.

**braid is an engine built so that the shared read is the normal case.** Almost every
inference engine can do this for conventional models. The model braid targets has a
component — the *recurrent scan*, §5.1 — that is much harder to share, and most engines
respond by giving up and serving one person at a time. braid does not.

The one thing that genuinely cannot be shared is each person's own conversation state, which
must be tracked separately per person. Keeping that cost small, and keeping everything
*else* shared, is the whole design.

---

## 2. The model braid runs

The MVP target is **Qwen3.5-4B**, a *hybrid* model. "Hybrid" means its 32 layers are not all
the same kind. Three out of every four are a **Gated DeltaNet** layer, and the fourth is a
conventional **attention** layer.

The difference matters for how memory is used, so it is worth one paragraph in plain terms:

- An **attention** layer remembers by *keeping every past word around* and re-reading them
  all. It is exact, and its memory grows with the length of the conversation.
- A **Gated DeltaNet** layer remembers by *maintaining a fixed-size summary* that it updates
  once per word. It is approximate, and its memory does **not** grow — a 100-word chat and a
  100,000-word chat cost exactly the same to store.

A hybrid gets most of the accuracy of the first and most of the cheapness of the second. It
also means braid has to implement two completely different memory systems and keep them in
step, which is what §4 and §5 are about.

### Exact shapes

Everything below comes out of the checkpoint's own `config.json` and is parsed by
[`ModelConfig`](../braid/model/config.py).

| Property | Value |
|---|---|
| Layers | 32 = **24 Gated DeltaNet + 8 attention**, period 4 (`layer_types`) |
| Layer pattern | `3 × (GDN → MLP) → 1 × (attention → MLP)` |
| `hidden_size` | 2,560 |
| MLP | dense SwiGLU, `intermediate_size` 9,216 — **no mixture-of-experts on this target** |
| Attention | 16 heads, `head_dim` **256**, 4 key/value heads (GQA 4:1), output-gated |
| Rotary | `partial_rotary_factor` 0.25 → only the first **64** of each 256-wide head rotates |
| GDN heads / groups | 32 value heads, 16 key groups (2 heads per group) |
| GDN dims | `head_dim` 128, `state_size` 128, inner 4,096 |
| GDN conv | 8,192 channels, kernel width 4 |
| Vocabulary | 248,320, **tied** — there is no separate `lm_head` tensor in the file |
| Weights | BF16 throughout, **7.83 GiB** (8.4 GB) |

Three of those will bite anyone who assumes the usual:

1. **`head_dim` is 256, and 16 × 256 = 4,096 ≠ 2,560.** The near-universal
   `hidden_size // num_attention_heads` fallback gives 160 here, which is wrong by a factor
   of 1.6. [`ModelConfig.from_dict`](../braid/model/config.py) *refuses* to infer it rather
   than relying on the fact that this particular checkpoint happens to raise on the reshape.
2. **The config is nested.** The published checkpoint is a vision-language model; every
   shape above lives under `text_config`, and the top level carries a `vision_config` whose
   `hidden_size` is 1,024. Reading the top level silently gives you the *visual tower's*
   dimensions.
3. **738 tensors are in the file; braid runs 426.** The other 312 are a 24-block visual
   tower (297) and a multi-token-prediction head (15). Neither is in the GGUF that llama.cpp
   runs, so loading them would not merely waste memory — it would make any head-to-head
   comparison compare two different models. The loader filters by prefix and *reports* what
   it dropped.

**Per-sequence memory**, which is the cost that is *not* shared between users:

```
recurrent state    24 GDN layers × (2 MiB state + 128 KiB conv window)  =  51 MiB per user, fixed
KV cache            8 attention layers × 4 KV heads × 256 × 2 × 2 bytes =  32 KiB per user per token
```

The 51 MiB is constant no matter how long the conversation runs. The 32 KiB/token is not.

The competitive target, `Qwen3.6-35B-A3B-NVFP4`, returns in Phase 5+; its GDN block is
dimensionally identical, which is why the 4B is the MVP. Its shapes are in
[`THESIS.md` §3](THESIS.md).

---

## 3. Where the code lives

```
braid/
  config.py              GDNConfig — the shape parameters the CUDA kernels need
  model/
    config.py            ModelConfig — the whole model's config, parsed from config.json
    loader.py            checkpoint → a flat dict of tensors, with four load-time transforms
    engine.py            Engine — construction, forward(), generate()
    layer.py             DecoderLayer — one layer: norm, mixer, residual, norm, MLP, residual
    gdn.py               GatedDeltaNet — the recurrent mixer
    attention.py         Attention + RotaryEmbedding — the conventional mixer
    mlp.py               MLP — dense SwiGLU
    norm.py              rms_norm and rms_norm_gated
    cache.py             KVCache, RecurrentCache, Cache — per-sequence state
  reference/
    gdn_ref.py           the fp32 oracle for the recurrence: naive and vectorized forms
  kernels/
    loader.py            JIT-compiles the CUDA extension for sm_120a
    csrc/gdn_decode.cu   batched recurrent decode step
    csrc/conv1d_decode.cu  batched slotted causal conv + SiLU
    csrc/bindings.cpp    pybind entry points
  bench/
    noise_floor.py       measures the variance floor, so gates aren't coin flips
    scan_scaling.py      Phase 1 evidence: does the scan scale with batch?
    gemm_probe.py        weight-only GEMM options at M ∈ 1..64
    fp8_probe.py         is there a usable reduced-byte weight path on sm_120?
scripts/
  parity_report.py       parity metrics + ablations proving the gate discriminates
  layer_trace_diag.py    per-LAYER divergence vs HF: smooth growth = rounding, a step = a bug
  gdn_layer_diag.py      per-STAGE divergence inside one GDN layer
  gdn_stage_diag.py      the same, staged differently
  cache_diag.py          is decode ≠ prefill a cache bug or bf16 accumulation?
tests/                   see §8
docs/
  ARCHITECTURE.md        this file
  THESIS.md              why braid exists; what is ruled out
  ROADMAP.md             build order and gates
  runbooks/              recorded measurements: scan scaling, noise floor, llama.cpp baseline
```

Two organising principles, both deliberate:

- **`braid/reference/` is the source of truth for arithmetic, and it is written in plain
  PyTorch.** The CUDA kernels are checked against it; so is the engine. When a fast path and
  the oracle disagree, the oracle is right until proven otherwise.
- **No class inherits from `torch.nn.Module`.** The sublayers are plain objects holding
  tensors. There is no parameter registration, no `state_dict`, no autograd. braid never
  trains anything, and the module machinery would only obscure where the memory is.

---

## 4. How a request flows

Four phases, same shape as any inference engine.

### Phase 1 — Load (once per process)

**Plain version:** read 8.4 GB of numbers off disk onto the graphics card, throw away the
parts of the file braid does not run, and apply a handful of fixed-up-front adjustments so
the rest of the code never has to think about them.

Entry: `load_checkpoint(path, device="cuda")` → `Checkpoint`
([`braid/model/loader.py`](../braid/model/loader.py)).

It reads `config.json` into a [`ModelConfig`](../braid/model/config.py), walks
`model.safetensors.index.json`, keeps only tensors under `model.language_model.`, and
renames them into a flat namespace (`layers.7.self_attn.q_proj`, `embed_tokens`, `norm`).
`Checkpoint.layer(i)` returns one layer's dict; `Checkpoint["embed_tokens"]` returns a named
global. A `LoadReport` records what was dropped.

**Four transforms happen at load time**, each of which is silently wrong if missed:

| Transform | Why it is here and not at use time |
|---|---|
| `A = −exp(A_log)` | The decay rate the recurrence needs. Keyed on the **source tensor name**, not on dtype and not on value — see §6. |
| `gamma = 1 + W` on plain norms | Qwen3.5 stores RMSNorm weights as *deltas*. Folded in fp32 at load so a caller cannot apply it twice — and deliberately **not** applied to `linear_attn.norm`, which is the gated form and stores gamma directly. |
| `conv1d.weight` `[C,1,K]` → `[C,K]` | The kernel wants the 2-D form. |
| `lm_head` ← `embed_tokens` | `tie_word_embeddings` is true; there is no `lm_head` tensor in the file at all. |

`_validate` then asserts every expected tensor is present with the expected shape, so a
missing or misnamed weight fails at load rather than as a confusing reshape error 30 layers
in.

### Phase 2 — Build (once per process)

Entry: `Engine.from_checkpoint(ckpt, device, dtype)` or the one-shot
`Engine.from_pretrained(path)` ([`braid/model/engine.py`](../braid/model/engine.py)).

This constructs 32 [`DecoderLayer`](../braid/model/layer.py) objects — each of which picks
its mixer from `cfg.layer_types[i]`, **not** from index arithmetic — plus the embedding
table, the final norm, the tied head, and a
[`RotaryEmbedding`](../braid/model/attention.py).

No memory is allocated for a conversation here. That is `Engine.allocate_cache(max_len)`,
which builds one [`Cache`](../braid/model/cache.py) holding a `KVCache` for each of the 8
attention layers and a `RecurrentCache` for each of the 24 GDN layers.

> **Dispatching on `layer_types` rather than on `i % 4`** looks like pedantry and is not.
> A checkpoint that broke the period-4 pattern would load cleanly under index arithmetic and
> then mix the wrong sublayer into 24 of 32 layers — fluent output, wrong model.

### Phase 3 — Prefill (once per request)

**Plain version:** read the user's whole question at once and bring the model's memory up to
date, ending with a prediction for the first word of the answer. This is what determines how
long you wait before the first word appears.

Entry: `Engine.forward(input_ids, cache, last_only=True)` with `input_ids` of shape `[1, T]`.

```
embed → for each of 32 layers: DecoderLayer(h, cos, sin, cache) → final RMSNorm → LM head
        └───────────────────── Engine.hidden_states ──────────────────────┘
```

**`hidden_states` and `forward` are separate entry points** because the LM head is the
expensive part to materialise: the vocabulary is 248,320 wide, so logits cost ~1 MB per token
in bf16 and 2 GB for a 2,048-token window in fp32. Generation only ever needs the last row
(`last_only=True`); the perplexity gate applies the head in slices over the hidden states
instead.

### Phase 4 — Decode (once per generated word)

**Plain version:** feed the word just produced back in, advance every layer's memory by one
step, and predict the next word. Repeat until the model says it is finished.

Entry: `Engine.generate(input_ids, max_new_tokens, temperature, top_p, seed, eos_token_id)`.

Each step is the same `forward` call with `T = 1`, which is what makes the caches
load-bearing: the attention layers append one key/value pair each, and the GDN layers advance
their conv window and recurrent state in place. Then `_sample` — argmax at `temperature == 0`,
otherwise temperature + top-p with an explicit `torch.Generator` so a seed reproduces a run.

**Today this is batch-1 only** and `generate` raises `NotImplementedError` on anything wider.
The batch axis is Phase 3 of the roadmap; see §9.

---

## 5. Inside one layer

Every layer, both kinds, is the same six steps
([`braid/model/layer.py`](../braid/model/layer.py)):

```
h = rms_norm(x, input_layernorm)
h = mixer(h, cache)                 ← the only part that differs
x = x + h                           ← residual

h = rms_norm(x, post_attention_layernorm)
x = x + mlp(h)                      ← residual
```

The `mixer` is a `GatedDeltaNet` on 24 layers and an `Attention` on 8. The
[`MLP`](../braid/model/mlp.py) is the same on all 32: dense SwiGLU,
`down(silu(gate(x)) * up(x))`.

### 5.1 The Gated DeltaNet layer

**Plain version.** This layer keeps a fixed-size scratchpad — a 2 MB grid of numbers per
layer per user — and each incoming word does two things to it: it *fades* everything already
written there slightly, and it *writes* a correction. Then it reads an answer back out. The
fade rate and the correction size are themselves computed from the word, which is what
"gated" means. Because the scratchpad never grows, a long conversation costs no more to
remember than a short one.

The catch, and the reason this is the hard part of the whole project: **step *t* cannot start
until step *t−1* has finished**, because it reads the scratchpad that step *t−1* just wrote.
Nearly everything else in a language model can be done for a thousand words simultaneously.
This cannot.

**Exactly** ([`braid/model/gdn.py`](../braid/model/gdn.py)):

```
1. qkv    = x @ in_proj_qkv                     [8192, 2560]
2. qkv    = silu(causal_conv1d(qkv, conv_state, W[8192,4], b))
3. q,k,v  = split(qkv, [2048, 2048, 4096])      ← Q FIRST, see §6
              q,k → [B,T,16,128]   v → [B,T,32,128]
4. beta   = sigmoid(x @ in_proj_b)              step size,  [B,T,32]
   alpha  = exp(A · softplus(x @ in_proj_a + dt_bias))   decay,  [B,T,32]
5. y      = scan(state, l2norm(q), l2norm(k), v, alpha, beta)   ← the recurrence
6. z      = x @ in_proj_z                       [4096, 2560]
   y      = rms_norm_gated(y, z, norm)          normalise THEN gate
7. out    = y @ out_proj                        [2560, 4096]
```

The recurrence in step 5, per head, with thread `d` owning column `d`:

```
kv[d]  = Σ_s H[s,d] · k̂[s]              reduction on the UNDECAYED state
δ[d]   = (v[d] − g · kv[d]) · β
H[s,d] = g · H[s,d] + k̂[s] · δ[d]
y[d]   = (Σ_s H_new[s,d] · q̂[s]) · rsqrt(head_dim)
```

The state update and the `y` accumulation **share one loop**. Splitting them into
"update, then read" is algebraically identical and not fp32-identical.

> **Prefill runs the one-token step in a Python loop.** That is slow and it is deliberate:
> it makes prefill and decode the *same arithmetic by construction* rather than by test, so
> the classic "generation drifts after the first token" bug cannot exist. The ragged
> chunkwise prefill scan that replaces the loop is Phase 5. No speed claim is made about
> this path.

The conv window (step 2) holds the last 4 **pre-convolution** inputs, matching HF's
`causal_conv1d_update` convention. Holding post-conv outputs instead decodes fluently and
wrongly. On a `T ≥ 4` prefill the cache write is `x[:, :, -K:]`; on `T < 4` it is a left-pad.

### 5.2 The attention layer

**Plain version:** the conventional mechanism — every new word compares itself against every
previous word and pulls in a weighted blend of them. Exact, and the stored history grows one
entry per word. There are only 8 of these layers out of 32, which is why the growing cost is
tolerable.

**Exactly** ([`braid/model/attention.py`](../braid/model/attention.py)). Four departures from
a textbook Llama block, all of them silent if missed:

- **`head_dim = 256` with 16 heads over `hidden_size = 2560`.** Every reshape uses the
  configured value; none derives it.
- **Output gating.** `q_proj` emits `2 × n_heads × head_dim = 8192` and the split is
  **per head** — `[q_h0 | gate_h0 | q_h1 | gate_h1 | …]`, not `[all_q | all_gate]`. braid
  views it as `[B, T, H, 2D]` and chunks the last axis. Splitting the flat 8,192 in half
  instead pairs head *h* with the gate of head *h/2*: plausible output, wrong model. The
  gate is applied as `o *= sigmoid(gate)` **after** attention and **before** `o_proj`.
- **q/k RMSNorm over the head dim, before rope**, with the `1 + W` gamma already folded at
  load.
- **Partial rope.** Only the first 64 dims of each 256-wide head rotate; the top 192 pass
  through unchanged. Rotating all 256 is not a shape error — it is a 0.37 relative-L2
  regression that a shape check will never catch (measured in `scripts/parity_report.py`).

`RotaryEmbedding` computes the text case directly. HF's module is MRoPE — it expands
positions into three grids (temporal, height, width) and interleaves their frequencies — but
for text HF broadcasts the same row three times, so the interleave selects between equal
values and is a no-op. The equivalence is *pinned by* `test_rope_matches_hf`, not assumed.

`forward` calls `F.scaled_dot_product_attention` and **refuses** the chunked-prefill case
(`T > 1` onto a non-empty cache with no explicit mask) rather than masking wrongly: SDPA's
`is_causal` aligns its mask top-left, which is only correct when query length equals key
length. Chunked prefill is Phase 3.

> **`head_dim = 256` disqualifies every fused attention backend on this box.** Flash,
> mem-efficient and cuDNN all decline with *"head_dim should be no more than 128"*, and
> PyTorch silently falls back to the math backend, which materialises the full
> `[B, H, T, T]` score matrix. This is fine for correctness work and is *not* fine for
> Phase 3 — which is why FlashInfer at `head_dim = 256` is load-bearing there rather than
> optional.

### 5.3 The two norms

[`braid/model/norm.py`](../braid/model/norm.py) has exactly two functions, and the difference
between them is worth ~2× perplexity:

| | stores | effective gamma | used by |
|---|---|---|---|
| `rms_norm` | a **delta** | `1 + W` (folded at load) | `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` |
| `rms_norm_gated` | gamma **directly** | `W` | `linear_attn.norm` only |

Both compute in fp32 and cast on the way out. `rms_norm_gated` normalises **first, then
gates** — the opposite order from the Mamba2 path in most reference implementations:

```
inv_rms = rsqrt(mean(y²) + eps)          ← eps INSIDE the sqrt, AFTER the mean
out     = y · inv_rms · gamma · silu(gate)
```

`gamma` is `[head_dim] = [128]`, **shared across all 32 heads** — not a `[4096]`
per-inner-dim gamma.

---

## 6. The numerics contract

**Plain version:** braid must produce the same numbers as the reference implementation the
model was trained against, to many decimal places. This is not perfectionism. A language
model that is slightly wrong does not crash and does not produce obvious nonsense — it
produces *fluent, confident, subtly worse text*, and there is no alarm that fires. The only
way to know is to compare against a known-good implementation, on real weights, at every
commit. That is what this section pins down and what §8 enforces.

The reference is **Hugging Face `transformers`**, not the C++ reference engine, because HF
is the implementation the checkpoint was trained with. Always compare against the HF
implementation that shares braid's numerics — bf16 against `sdpa`, fp32 against `eager`.

### Where the gate belongs

**rel L2 ≤ 5e-3 and cosine ≥ 0.99999 is a *single-layer* threshold.** On a 32-layer bf16
stack it measures accumulated rounding, not correctness: `scripts/layer_trace_diag.py` shows
the residual growing smoothly from 1.9e-4 and plateauing near 1e-2 **with no step at any
layer**, while the same forward in fp32 reads 6.4e-7. Gating the bf16 stack on 5e-3 would be
gating on depth.

So the contract is split by arm:

| Arm | Gate | Measured |
|---|---|---|
| single sublayer, either dtype | rel L2 ≤ 5e-3, cosine ≥ 0.99999 | attention fp32 **bit-exact**; MLP **bit-exact**; one GDN layer **bit-identical at T ≤ 4**, 4.8e-5 at T=24 |
| full stack, fp32 | the same strict gate | **rel L2 6.4e-7**, cosine 1.000000000 |
| full stack, bf16 | **greedy token identity** | rel L2 8.3e-3, **100% argmax agreement with HF** |
| caches (decode == prefill), fp32 | the strict gate | 4.7e-7 (GDN), 4.9e-7 (attention) |

Same reasoning for decode-vs-prefill: run the exactness check in fp32, where a cache bug
cannot hide behind rounding, and check the bf16 arm on tokens.

### Settled by measurement

Each of these was an open question, resolved empirically rather than by reading anyone's
source, and each is pinned by a test.

| Item | Resolution | Evidence |
|---|---|---|
| **conv split order** | `[Q \| K \| V]`, **Q first** | `tests/test_hf_parity.py`. Two independent readings of the reference engine disagreed; HF's `torch.split(mixed_qkv, [key_dim, key_dim, value_dim])` supports Q-first. Getting it wrong is fluent and completely wrong, with no crash. |
| **`1 + W` on `linear_attn.norm`** | **No offset** — it is the gated form | `tests/test_hf_parity.py`. Dropping the offset on a *plain* norm is the 13.65 → 6.82 perplexity bug. |
| **l2norm form** | additive `1e-6` (HF), applied in the **activation** dtype and only then widened to fp32 | `braid/model/gdn.py`. The reference engine uses clamped-rsqrt `rsqrtf(fmax(Σk², 1e-12))` instead, which differs by 10⁶ in the degenerate case. |
| **`A = −exp(A_log)`** | keyed on the **source tensor name** | Neither published heuristic works here — see below. |
| **Gate clamps** | **not applied** | The reference engine clamps `A·dt` at −20 and `b_raw` at ±20. Both are no-ops on real activations (`sigmoid(20)` is 1 to within 2e-9) and HF does not clamp, so parity beats the deviation. |
| **fp32 fold of the gamma offset** | fp32, not bf16 | `scripts/parity_report.py`: fp32 fold 0.0 rel-L2; bf16 fold 1.86e-3; no offset at all 8.04e-1. The bf16 fold would still *clear* the gate — fp32 is chosen because it costs nothing (2,560 floats per layer) and buys back three orders of magnitude of headroom. |

**The `A_log` transform deserves its own paragraph**, because both published heuristics are
wrong on this checkpoint. The architecture spec originally prescribed *"any element ≥ 0 ⇒
raw HF"*. Measured: **every one of layer 0's 32 `A_log` entries is negative (−4.22 … −0.96)**,
so the value test concludes "already transformed", skips the `exp`, and leaves `A = −2.7`
where it should be `−0.067`. That is a ~40× *over*-fast decay: the state collapses toward
zero **silently**, and the absmax tell that the reference engine documents
(`0.04, 0.06, 0.40, 2.51, 110, 31680, inf` → NaN) never fires. The dtype heuristic fails too,
because this checkpoint ships `A_log` as F32. braid keys on the tensor name — unambiguous for
a safetensors load — and range-checks that the result is finite and strictly negative.

### Deliberate deviations from HF, and their cost

Two places braid is *cleaner* than HF, both measured and both inside the gate:

- `rms_norm_gated` keeps the normalised value in fp32 where HF rounds it to the activation
  dtype before applying gamma. Worth **3.1e-3 relative** across a whole GDN layer
  (`scripts/gdn_layer_diag.py`).
- **`beta`'s sigmoid is taken in the activation dtype and only then widened**, matching HF's
  asymmetry (`beta = b.sigmoid()` vs `g = −A_log.float().exp() * softplus(...)`). Taking it in
  fp32 instead moves the whole layer output by **4.8e-3 relative** — one bf16 epsilon, flat
  across T and across tokens. `beta` is the delta-rule step size, so its rounding lands
  straight in the output.

> Both are kept, and the full bf16 stack still reaches **100% greedy token identity** with
> HF, which is the gate that matters. They are recorded here because Phase 3's gate is
> greedy token identity *across a batch*, where a 3e-3 deviation either does or does not
> flip an argmax — so they are decisions held on purpose, not inherited by accident.

### Settled layout traps

- conv1d weights are `[C, K]` with `K` contiguous per channel.
- conv bias is added **after** the dot and **before** SiLU.
- SiLU is applied to **all** of Q, K and V, not just V.
- Head→group mapping is `g = h // heads_per_group` — the **grouped** (HF safetensors)
  layout. GGUF uses tiled `g = h % n_groups`. Both are valid permutations of the same index
  range, so a mismatch produces plausible garbage, never a crash.
- Recurrent state is `[B, n_heads, state_size, head_dim]` fp32 with **`head_dim`
  fastest-varying** — the per-slot slab layout the CUDA kernel already indexes.

### State precision

- **FP8 E4M3 state is refuted.** The 3-bit mantissa amplifies through the delta rule and
  degenerates after ~50 special tokens in multi-turn chat. Do not attempt it.
- **FP16 state is NOT refuted** and is a live open question — see [`THESIS.md` §7](THESIS.md).

fp32 is mandatory for the MVP so that parity is unambiguous.

---

## 7. The CUDA kernels

**Plain version:** almost all of braid is ordinary PyTorch. Two small pieces are hand-written
in CUDA, because they are the parts no library does the way braid needs. Both exist and are
tested; **neither is wired into the engine yet** (§9).

[`braid/kernels/csrc/`](../braid/kernels/csrc/), JIT-compiled on first use by
[`braid/kernels/loader.py`](../braid/kernels/loader.py) at `TORCH_CUDA_ARCH_LIST=12.0a` —
the arch-*conditional* target, which exposes instructions plain `sm_120` does not. No
`--use_fast_math`: it turns `rsqrtf` into an approximation and breaks fp32 parity at the
asserted tolerances.

### `gdn_decode` — the batched recurrent step

One block per `(batch row, head)`; one thread per `head_dim` column `d`, holding that state
column in registers. The batch axis on the grid is the entire point — it is what lets eight
users' scans run in one launch instead of eight.

**The state pool is indexed indirectly:**

```
h_pool    [max_slots, n_heads, state_size, head_dim]  fp32
slot_idx  [batch]                                     int32   ← read inside the kernel
```

Reading `slot_idx` from *device* memory rather than baking a base pointer into the launch is
what makes **one captured CUDA graph valid for every assignment of users to slots, forever**.
Measured at 10.3 µs to replay across a slot reassignment, against the 10–20 ms re-capture the
reference engine pays for the same event.

The pool is **layer-major, not sequence-major**, so a batched per-layer scan reads
contiguously.

### `conv1d_decode` — the batched slotted causal conv

One thread per channel: slide the window left by one, append the new value, dot against the
weight row, add bias, apply SiLU. Same `slot_idx` contract.

### sm_120a constraints these kernels are built around

Each cost the reference engine real time; each is absorbed rather than rediscovered.

| Constraint | Consequence |
|---|---|
| **`__launch_bounds__(HD, 2)` at HD=128 is a ptxas MISCOMPILE** — garbage output, correct math | `gdn_decode.cu` uses `__launch_bounds__(HD, 1)`, and the min-1 form is also what lets ptxas give each thread the ~128 registers `S_reg` needs without spilling. |
| Opt-in shared memory is **~99 KB**, not H100's 228 KB | Query `sharedMemPerBlockOptin`; design tiles to ~97 KB. |
| **No TMA** — `cp.async.bulk` and `st.async .b128` to global are unavailable | Use `cp.async.ca/cg.shared.global` at 16 B. Do not port Hopper pipelines. |
| `nvcuda::wmma` compiles but lowers to **HMMA**, not the FP8/FP4 pipes | Hand-write `mma.sync`, or don't claim tensor cores. |
| `cudaMallocAsync` inside a captured graph **crashes**; any device→host copy inside capture is an illegal memory access | Pre-allocate every workspace; keep all args device-side. This is exactly why `slot_idx` is a device tensor — and why the kernel's slot validation runs **only when the stream is not capturing**. |
| CUTLASS NVFP4 on sm_120 is **non-deterministic under `cudaGraphExecUpdate`** | Keep NVFP4 GEMMs out of exec-update if bitwise reproducibility is needed. |
| WDDM silently spills to host at ~0 MiB free — bandwidth 1,530 → 237 GB/s | Never size a large allocation from an *estimate* of another's future size. Leave ≥1 GiB free. |
| cuBLASLt returns **zero algorithms** for grouped GEMM on sm_120 | MoE grouped GEMM (Phase 5+) must be CUTLASS block-scaled or hand-rolled. |
| A 247-instruction survey across CUDA 13.2→13.3 flipped **0 instructions** | The ISA surface is silicon-fixed. Don't re-probe on every toolkit bump. |

---

## 8. How braid is tested

**Plain version:** the only thing that distinguishes a correct engine from a subtly broken
one is a comparison against something known-good, run automatically, on real weights. braid
has a GPU and runs these on every commit.

Four layers, each catching what the one below cannot:

| Layer | Files | Catches |
|---|---|---|
| **Oracle agreement** | `test_gdn_ref.py` | The naive and vectorized fp32 recurrences disagreeing — i.e. a bug in the thing everything else is checked against. |
| **Kernel vs oracle** | `test_gdn_decode_kernel.py`, `test_conv1d_decode.py`, `test_slot_indirection.py` | A CUDA kernel deviating from the fp32 reference; a graph replay breaking when slots are reassigned. The conv is verified over **8 sequential steps with rotating slots** — a single step cannot catch a window-orientation error. |
| **Sublayer vs HF** | `test_hf_parity.py`, `test_attention_parity.py` | Every layout and numerics trap in §6, on real weights. |
| **Whole model** | `test_full_forward.py` | A correct sublayer wired to the wrong layer index; a cache that decodes differently from prefill; degenerate generation. |

**69 tests green on the remote 5090** as of the Phase 2 item 3 commit.

`test_full_forward.py` is the load-bearing one, in four escalating groups: one GDN layer vs
`Qwen3_5GatedDeltaNet`; the full 32-layer stack vs `Qwen3_5TextModel` in **both** fp32 (strict
gate) and bf16 (greedy token identity); **decode == prefill** per sublayer in fp32, so the
blame for any mismatch is unambiguous; and proof of life — `"The capital of France is"` →
`" Paris."`, then 128 greedy tokens with no repeated 8-gram.

**Ablations prove the gate discriminates** rather than passing everything
(`scripts/parity_report.py`): flat-halves `[q|gate]` split reads **1.12**; rope over all 256
dims **0.37**; q/k norm missing the `1+W` offset **0.78** — against a 5e-3 gate.

### Two traps that live in the harness, not the engine

1. **`module.to(bf16)` then `load_state_dict` truncates.** The copy goes *into* the
   already-bf16 parameter, so every tensor this checkpoint stores as F32 gets rounded —
   `linear_attn.norm` moves 2.4e-3 and the "reference" becomes a **worse model than braid**.
   That accounted for nearly all of one GDN layer's apparent parity gap. Reference modules
   are built on `meta` and loaded with `assign=True`.
2. **Two copies of a 4B model do not fit comfortably on a 32 GB card**, and an fp32 arm that
   holds a bf16 copy *and* an fp32 copy is what pushes it over. `load_checkpoint(dtype=…)`
   recasts on the host before transfer so only one copy is ever resident.

Measurement infrastructure lives in `braid/bench/`, and the measurement *rules* — host-health
classifier, 3 processes × 3 reps, print the spread — are in
[`THESIS.md` §4](THESIS.md). Recorded results are in [`docs/runbooks/`](runbooks/).

`tests/test_env.py` and `tests/test_noise_floor.py` gate the environment itself, because a
2% performance gate against a 10% noise floor is a coin flip.

---

## 9. You are here

What exists on disk today, honestly.

| Component | Status |
|---|---|
| Config parsing, `GDNConfig` + `ModelConfig` | **Done**, validated |
| Checkpoint loader with the four transforms | **Done**, 12 tests |
| fp32 oracle for the recurrence | **Done** |
| Batched CUDA `gdn_decode` + `conv1d_decode`, slot indirection, graph replay | **Done and measured** — Phase 1 gate passed on all four conditions: 2.58× aggregate at B=8, 104% of HBM, no L2 cliff, 10.3 µs graph replay across slot reassignment |
| Single-layer HF parity: GDN, attention, MLP, norms, rope | **Done** |
| B=1 eager engine — 32 layers, caches, greedy + top-p sampling | **Done 2026-08-07.** `"The capital of France is"` → `" Paris."`, 128 greedy tokens clean, 100% greedy token identity with HF, fp32 stack at 6.4e-7. 69 tests green. |
| Perplexity gate | **In progress** — Phase 2 item 4. `Engine.hidden_states` was split out of `forward` for it; the gate is PPL within 20% of an HF bf16 CPU reference over a pinned ≥10k-token corpus, absolute value recorded. |
| Batch axis through the whole forward | **Not started** — Phase 3 |
| Graph buckets {1,2,4,8,16}, paged KV, chunked prefill | **Not started** — Phase 3 |
| Scheduler, slot lifecycle, SSE server | **Not started** — Phase 4 |
| Ragged chunkwise prefill scan | **Not started** — Phase 5 |
| MoE, NVFP4, the 35B target | **Not started** — Phase 5+ |

**Two gaps worth stating out loud**, because they are invisible from the file listing:

1. **The engine does not call the CUDA kernels.** `braid/model/gdn.py` imports
   `gdn_decode_vectorized` from `braid/reference/` and runs it in a Python loop. The kernels
   are Phase 1 evidence, verified standalone; wiring them into the forward pass is Phase 3
   work, and doing it before the B=1 forward is proven correct would mean debugging two
   things at once.
2. **There is no batch axis anywhere above the kernels.** `Cache` takes a `batch` argument
   and `Engine.generate` raises on `batch != 1`. The kernels have the axis; the runtime does
   not yet.

Neither is a defect. Both are the Phase 2 → Phase 3 seam, and crossing it before the B=1
forward was proven correct would have meant debugging two things at once.

[`ROADMAP.md`](ROADMAP.md) has the gates and the build order.

---

## Glossary

**Token** — roughly a word, sometimes a word fragment. Models read and write tokens, not
characters. "tok/s" is tokens per second, the throughput number everything is judged on.

**Prefill** — processing the user's prompt, all at once, before any answer is produced.
Determines time-to-first-token.

**Decode** — producing the answer one token at a time. Each step depends on the previous
one, so it cannot be parallelised over time — only over *users*.

**Batch / concurrency (B, c)** — how many users' requests are being advanced in the same
step. The whole point of braid.

**Aggregate vs per-stream** — aggregate tok/s is the total across all users; per-stream ITL
(inter-token latency) is how long one user waits between words. Batching improves the first
and slightly worsens the second. Quoting them as independent wins is how a 6× error gets
made.

**KV cache** — the stored history an attention layer re-reads. Grows one entry per token per
user.

**Recurrent state** — the fixed-size scratchpad a Gated DeltaNet layer maintains. 51 MiB per
user here, and it does not grow.

**Slot** — a numbered place in the pre-allocated pool where one user's recurrent state lives.
Users come and go; slots get reused.

**Roofline / the memory wall** — the hard ceiling set by how fast the card can read its own
memory. If a workload is at the wall, no code change makes it faster; only moving fewer bytes
does.

**Bandwidth-bound / weight-bound** — the workload is limited by reading bytes, not by
arithmetic. Language model decoding is almost always this.

**RMSNorm** — a rescaling step between layers. Cheap, and getting its details wrong is worth
a factor of two in quality.

**Parity** — how closely braid's output matches a known-good reference on identical input.
Measured as relative L2 (how far off, as a fraction) and cosine similarity (how well the
shape matches).

**Perplexity (PPL)** — a quality score for a language model. Lower is better. Doubling it
means something is broken.

**CUDA graph** — a recording of a sequence of GPU operations that can be replayed without
the CPU re-issuing each one. Removes launch overhead, but bakes in memory addresses — which
is why the state pool is indexed through a device-side `slot_idx` rather than a raw pointer.

**Kernel** — one program that runs on the GPU. "Launching" one has fixed overhead, which is
why fusing several into one, or replaying them from a graph, can matter.

**bf16 / fp32** — 16-bit and 32-bit floating point. bf16 halves memory traffic and loses
precision; braid runs weights and activations in bf16 and the recurrent state in fp32.

**sm_120a** — the RTX 5090's GPU architecture target. The trailing `a` is the
*arch-conditional* variant, which exposes instructions plain `sm_120` does not.

**GDN (Gated DeltaNet)** — the recurrent layer type; see §5.1.

**GQA** — grouped-query attention: several query heads share one key/value head, shrinking
the KV cache. 4:1 here.

**SwiGLU** — the feed-forward block used by essentially every modern model.

**MoE (mixture of experts)** — a model whose feed-forward layer routes each token to a few
of many sub-networks. Not on the 4B target; arrives with the 35B in Phase 5+.
