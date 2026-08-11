# braid — architecture

This is braid as it stands, not as it's planned. Where this file and the code disagree the
code is right, and the file needs fixing.

Two companions. [`THESIS.md`](THESIS.md) has the argument for the project: the competitive
case, the throughput arithmetic, and the ledger of things already measured and ruled out.
[`ROADMAP.md`](ROADMAP.md) has the build order and the gates. Section 9 below is the status.

The first half should read without any CUDA background. There's a glossary at the bottom for
the vocabulary.

---

## 1. What braid is

braid runs a language model on one graphics card and serves several requests at the same
time.

The second half is the hard part, so it's worth a paragraph on why. A language model is a
big table of numbers, about 9 billion of them here, 16.68 GiB sitting on the card in BF16. To
produce one word of an answer you read essentially that whole table and do arithmetic on the
way through. Then you read all of it again for the next word.

A 5090 reads its own memory at roughly 1.5 TB/s. Divide, and you get about 85 words a second.
No amount of clever code gets past that; the card isn't thinking slowly, it's reading as fast
as the hardware goes. (braid ships FP8 weights, which halves the bytes and so doubles the
ceiling — but it moves the wall, it doesn't remove it.)

But the reading can be shared. With eight requests in flight you read the table once and
advance all eight with it. At sixteen the arithmetic barely changes while the total output
roughly sixteen-folds.

Most engines do this for ordinary models. The one we care about has a piece that resists it,
the recurrent scan in §5.1, and the common response is to stop batching and serve one request
at a time. That's the gap braid is aimed at.

What genuinely can't be shared is each request's own conversation state. Keeping that cheap
while everything else stays shared is most of the design.

---

## 2. The model

braid runs two checkpoints from the same family. **Qwen3.5-9B is the target** — every
published number, in the README and in [`CHECKPOINT.md`](CHECKPOINT.md), is measured on it.
**Qwen3.5-4B is the parity model**: it is what the test suite loads by default
(`BRAID_MODEL_DIR`), because the oracle gates need headroom the 9B doesn't leave. At 4B a full
32-layer fp32 stack is 16.8 GiB and stays on the card, where the 9B is 35.8 GiB against 31.4
and has to be truncated; and a gate that builds a *second* live engine gets two bf16 copies at
7.8 GiB each at 4B, against 16.7 each at 9B (`tests/conftest.py`). The 4B was the MVP target
through Phase 2–3 and [`ROADMAP.md`](ROADMAP.md) records that history; the engine moved up a
model at the Phase 4 close on 2026-08-09.

Both are hybrids, which here means the 32 layers aren't all one kind: three of every four are
Gated DeltaNet, and the fourth is ordinary attention.

The split is about memory. Attention remembers by keeping every past word around and
re-reading all of them, which is exact and grows with the conversation. Gated DeltaNet keeps
a fixed-size summary and updates it once per word, which is approximate and costs the same
whether the chat is 100 words or 100,000. A hybrid buys most of the first's accuracy at most
of the second's price. The bill is that braid implements two unrelated memory systems and has
to keep them in step, which is what §4 and §5 are largely about.

Read from each checkpoint's `config.json`:

| | **Qwen3.5-9B** (target) | **Qwen3.5-4B** (parity) |
|---|---|---|
| Layers | 32: 24 Gated DeltaNet + 8 attention, period 4 (`layer_types`) | same |
| Pattern | `3 × (GDN → MLP)` then `1 × (attention → MLP)` | same |
| `hidden_size` | 4,096 | 2,560 |
| MLP | dense SwiGLU, width 12,288 | dense SwiGLU, width 9,216 |
| Attention | 16 heads, `head_dim` 256, 4 KV heads (GQA 4:1), output-gated | same |
| Rotary | `partial_rotary_factor` 0.25, so only the first 64 of each 256-wide head rotates | same |
| GDN | 32 value heads, 16 key groups, `head_dim` 128, `state_size` 128, inner 4,096 | same |
| GDN conv | 8,192 channels, kernel width 4 | same |
| Vocab | 248,320, **untied** — a real `lm_head` tensor | 248,320, **tied** — no `lm_head` in the file |
| Weights | BF16 throughout, 16.68 GiB (427 text tensors) | BF16 throughout, 7.83 GiB (426 text tensors) |

Neither has a mixture-of-experts; both MLPs are dense. The whole GDN block — heads, groups,
head dim, state size, conv width — is **identical across the two**, which is why moving up a
model cost no kernel work and why the parity model is still a valid oracle for the scan.

A few of these bite anyone who assumes the usual layout.

`head_dim` is 256 on both, and the common `hidden_size // num_attention_heads` fallback agrees
with that on the 9B (4,096 / 16 = 256) and disagrees on the 4B (2,560 / 16 = 160). That is the
worst possible arrangement: adopt the fallback and it works on the target and reshapes into
garbage on the parity model. [`ModelConfig.from_dict`](../braid/model/config.py) refuses to
infer `head_dim` at all rather than lean on either coincidence.

The config is nested. These are vision-language checkpoints, so every shape above lives under
`text_config`, and the top level carries a `vision_config` whose `hidden_size` is 1,152 on the
9B and 1,024 on the 4B. Read the top level and you silently get the visual tower's dimensions.

braid loads only the text stack, and the file is mostly not that: **9B**, 775 tensors of which
braid loads 427 — the rest is a visual tower (333) and an MTP head (15); **4B**, 738 tensors of
which braid loads 426 — a 24-block visual tower (297) and an MTP head (15). Neither tower is in
the GGUF that llama.cpp runs from, so loading them wouldn't merely waste memory, it would make
any head-to-head compare two different models. The loader filters by prefix and reports what it
dropped.

Per-sequence memory, meaning the part that doesn't get shared between requests. Because the
GDN and attention dimensions are the same on both checkpoints, **this arithmetic is identical
for the 9B and the 4B** — only the shared weights get bigger:

```
recurrent state   24 GDN layers × (2 MiB state + 128 KiB conv window)  =  51 MiB, fixed
KV                 8 attn layers × 4 kv heads × 256 × 2 × 2 bytes      =  32 KiB per token
```

The 51 MiB doesn't move however long the conversation runs. The 32 KiB/token does. (That 51
splits as 48 MiB of `h_state` and 3 MiB of conv window, which is the 48 MiB
`tests/test_state_dtype.py` prices when it halves the storage to fp16.)

The competitive target, `Qwen3.6-35B-A3B-NVFP4`, comes back in Phase 5+. Its GDN block has
identical dimensions to both checkpoints above, which is why this family was chosen to build
on; its shapes are in [`THESIS.md` §3](THESIS.md).

---

## 3. Where things live

```
braid/
  config.py              GDNConfig — the shape parameters the CUDA kernels need
  model/
    config.py            ModelConfig — the whole model's config, from config.json
    loader.py            checkpoint → flat tensor dict, plus four load-time transforms
    engine.py            Engine — construction, hidden_states/forward, decode_step, generate
    graph.py             GraphedDecoder — CUDA-graph capture, one per batch bucket
    layer.py             DecoderLayer — norm, mixer, residual, norm, MLP, residual
    gdn.py               GatedDeltaNet — the recurrent mixer, torch and kernel paths
    attention.py         Attention, RotaryEmbedding, grouped_decode_attention
    mlp.py               MLP — dense SwiGLU
    norm.py              rms_norm, rms_norm_gated
    cache.py             slot-addressed KV and recurrent pools
  reference/
    gdn_ref.py           the fp32 oracle for the recurrence, naive and vectorized
  kernels/
    loader.py            JIT-compiles the extension for sm_120a
    csrc/gdn_decode.cu   batched recurrent decode step
    csrc/conv1d_decode.cu   batched slotted causal conv + SiLU
  bench/
    perplexity.py        the Phase 2 quality gate
    decode_speed.py      tok/s per bucket, graphs on vs off
    decode_profile.py    where the step actually goes
    scan_scaling.py      Phase 1 evidence: does the scan scale with batch?
    noise_floor.py       the variance floor, so gates aren't coin flips
    gemm_probe.py        weight-only GEMM options at M ∈ 1..64
    gemm_paths.py        what can be done about the GEMM-dominated step
    fp8_probe.py         is there a usable reduced-byte weight path on sm_120?
scripts/
  parity_report.py       parity metrics, plus ablations showing the gate discriminates
  layer_trace_diag.py    per-layer divergence vs HF: smooth growth is rounding, a step is a bug
  gdn_layer_diag.py      per-stage divergence inside one GDN layer
  cache_diag.py          is decode ≠ prefill a cache bug or bf16 accumulation?
  kernel_path_diag.py    CUDA path vs torch path, one layer, identical state
  batch_identity_diag.py where a batched run and a sequential run part ways
  decode_attn_diag.py    the attention decode path
  rmsnorm_probe.py       is F.rms_norm faster, and is it the same function?
  sdpa_backend_diag.py   which SDPA backend actually runs, and why
docs/runbooks/           recorded measurements, one file per campaign
```

Two things about the layout that aren't obvious from the tree.

`braid/reference/` holds the arithmetic in plain PyTorch and it's the source of truth. The
CUDA kernels are checked against it and so is the engine. When a fast path disagrees with the
oracle, assume the oracle is right until you've shown otherwise.

Nothing subclasses `torch.nn.Module`. The sublayers are plain objects holding tensors: no
parameter registration, no `state_dict`, no autograd. braid never trains anything, and the
module machinery would mostly hide where the memory went.

---

## 4. How a request moves through

### Load

Read 16.68 GiB off disk onto the card (7.83 on the parity model), skip the parts braid doesn't
run, and apply a few fixups so nothing downstream has to think about them.

`load_checkpoint(path, device, dtype)` → `Checkpoint`, in
[`braid/model/loader.py`](../braid/model/loader.py). It reads `config.json` into a
`ModelConfig`, walks the safetensors index, keeps only what's under
`model.language_model.`, and renames into a flat namespace: `layers.7.self_attn.q_proj`,
`embed_tokens`, `norm`. `Checkpoint.layer(i)` hands back one layer's dict. A `LoadReport`
records what got dropped.

The `dtype=` argument recasts on the host before transfer, so an fp32 run never holds a bf16
and an fp32 copy at the same time. On the 9B that doubling is not survivable — the fp32 copy
alone is 33.4 GiB against a 32 GB card — which is also why the fp32 oracle tests run on the
4B and, above a layer budget, on a truncated stack (`tests/conftest.py`).

Four transforms happen here, each of them silently wrong if missed:

| | |
|---|---|
| `A = −exp(A_log)` | The decay rate the recurrence wants. Keyed on the source tensor name, not on dtype and not on value — see §6. |
| `gamma = 1 + W` on plain norms | Qwen3.5 stores RMSNorm weights as deltas. Folded in fp32 at load so no caller can apply it twice, and deliberately *not* applied to `linear_attn.norm`, which is the gated form and stores gamma directly. |
| `conv1d.weight` `[C,1,K]` → `[C,K]` | The kernel wants the 2-D form. |
| `lm_head` ← `embed_tokens` | `tie_word_embeddings` is true and there is no `lm_head` tensor in the file. |

`_validate` then checks every expected tensor is present at the expected shape, so a missing
or misnamed weight fails at load instead of as a confusing reshape error thirty layers in.

### Build

`Engine.from_checkpoint(ckpt, device, dtype, use_kernels=False)`, or `from_pretrained(path)`
for the one-shot version.

This builds 32 [`DecoderLayer`](../braid/model/layer.py) objects, each picking its mixer from
`cfg.layer_types[i]` rather than from index arithmetic, plus the embedding table, the final
norm, the tied head, and a `RotaryEmbedding`. Dispatching on `layer_types` looks like
pedantry and isn't: a checkpoint that broke the period-4 pattern would load cleanly under
`i % 4` and then mix the wrong sublayer into 24 of 32 layers, which is fluent output from the
wrong model.

Conversation memory comes from `Engine.allocate_cache(max_len, max_slots)`, which is a
separate call and is described in §4.1.

### Prefill

Read the prompt all at once and bring every layer's memory up to date, ending with a
prediction for the first word. This is what sets how long you wait before anything appears.

`Engine.hidden_states(input_ids, cache, last_only)` runs
`embed → 32 × DecoderLayer → final RMSNorm`, and `Engine.forward` is that plus one linear
into the vocabulary. They're split because the LM head is the expensive part to materialise:
248,320 wide is about 1 MB per token in bf16, and 2 GB for a 2,048-token window in fp32.
Generation only needs the last row. Perplexity applies the head in 256-position slices over
the hidden states instead, which is what keeps that run at 250 MB of activations rather than
2 GB.

Prefill is one sequence at a time. A padded rectangle would feed pad tokens through the GDN
recurrence and corrupt the state unless separately masked. Ragged batched prefill is Phase 3
item 3.

### Decode

Feed back the word just produced, advance every layer's memory one step, predict the next.

Two entry points, and the difference matters. `hidden_states` is the general path: it handles
any `T`, sizes the KV slice from the live maximum, and decides whether a mask is needed.
`decode_step(tokens, cache)` is the narrow one, `T = 1` only, with no host sync and static
shapes, which is what lets it be captured into a CUDA graph.

Getting the syncs out cost two trades. `kv_len` is fixed at `cache.max_len` rather than the
live maximum, because a shape that depends on device state can't be a captured shape; masked
keys contribute `exp(−inf) = 0` so the result is unchanged, but the whole KV buffer gets read
every step. And the mask is always built rather than skipped by a data-dependent branch.

`generate_batch(prompts, …)` prefills each prompt on its own, then decodes them all together,
which is what a real continuous-batching engine does and also what these two halves can each
support today.

### 4.1 Slots, and why the cache looks like that

State is addressed by **slot**, not by batch row: a pool of `max_slots` entries, plus a
device-resident `slot_idx[batch]` saying which entry each row of the current batch owns.

The reason is CUDA graphs. Bake a base pointer per sequence and a captured graph is valid for
exactly one assignment of sequences to slots, so every admission or eviction forces a
re-capture — the reference engine documents that at 10–20 ms. Read `slot_idx` from device
memory inside the kernel and one graph is valid for every assignment. braid measured the
replay path at 10.3 µs across three different assignments.

So `Cache.select(slots)` returns a view sharing every pool tensor with only the assignment
changed, and [`GraphedDecoder`](../braid/model/graph.py) reassigns by copying into a static
slot buffer instead of re-capturing. One graph per bucket {1, 2, 4, 8, 16}; a batch of 5
replays the 8-bucket with the spare rows padded.

Padding isn't free of meaning. A padded row is a real sequence as far as the arithmetic is
concerned: it occupies a slot, advances that slot's length, and writes KV. So `step()`
refuses to pad without explicit scratch slots rather than quietly advancing whatever sits at
rows B..bucket.

KV is pooled but not paged. Each slot owns a contiguous `max_len` run, which wastes capacity
on short sequences and is exactly what the Phase 3 item 3 block manager replaces. It isn't a
layout braid intends to serve on.

---

## 5. Inside a layer

Every layer is the same six lines ([`layer.py`](../braid/model/layer.py)):

```
h = rms_norm(x, input_layernorm)
h = mixer(h, cache)                  ← the only part that differs
x = x + h

h = rms_norm(x, post_attention_layernorm)
x = x + mlp(h)
```

The mixer is `GatedDeltaNet` on 24 layers and `Attention` on 8. The MLP is the same on all
32: dense SwiGLU, `down(silu(gate(x)) * up(x))`.

### 5.1 Gated DeltaNet

This layer keeps a fixed-size scratchpad, 2 MB per layer per sequence, and each incoming word
does two things to it. It fades everything already written there slightly, and it writes a
correction. Then it reads an answer back out. Both the fade rate and the correction size are
computed from the word itself, which is what "gated" means. The scratchpad never grows, so a
long conversation costs no more to remember than a short one.

The catch, and it's the reason this project exists: step *t* can't start until step *t−1*
finishes, because it reads the scratchpad *t−1* just wrote. Nearly everything else in a
transformer can be done for a thousand words at once. This can't.

[`braid/model/gdn.py`](../braid/model/gdn.py):

Weight shapes below are the 9B's. The 4B's differ **only** in the `hidden_size` axis (2,560
where this says 4,096); every GDN-internal width is shared, per §2.

```
1. qkv    = x @ in_proj_qkv                        [8192, 4096]
2. qkv    = silu(causal_conv1d(qkv, conv_state, W[8192,4], b))
3. q,k,v  = split(qkv, [2048, 2048, 4096])         ← Q first, see §6
              q,k → [B,T,16,128]   v → [B,T,32,128]
4. beta   = sigmoid(x @ in_proj_b)                 step size,  [B,T,32]
   alpha  = exp(A · softplus(x @ in_proj_a + dt_bias))   decay, [B,T,32]
5. y      = scan(state, l2norm(q), l2norm(k), v, alpha, beta)
6. z      = x @ in_proj_z                          [4096, 4096]
   y      = rms_norm_gated(y, z, norm)             normalise, then gate
7. out    = y @ out_proj                           [4096, 4096]
```

The recurrence in step 5, per head, with thread `d` owning column `d`:

```
kv[d]  = Σ_s H[s,d] · k̂[s]              reduction on the UNDECAYED state
δ[d]   = (v[d] − g · kv[d]) · β
H[s,d] = g · H[s,d] + k̂[s] · δ[d]
y[d]   = (Σ_s H_new[s,d] · q̂[s]) · rsqrt(head_dim)
```

The state update and the `y` accumulation share one loop. Splitting them into "update, then
read" is algebraically identical and not fp32-identical.

Two implementations sit behind this. The default runs `gdn_decode_vectorized` from
`braid/reference/`, in a Python loop over tokens. `Engine(use_kernels=True)` swaps the decode
step for the CUDA kernels; prefill stays on the torch path either way. They're not quite the
same function in bf16 and §7 says why.

Running the one-token step in a loop for prefill is slow and deliberate. It makes prefill and
decode the same arithmetic by construction rather than by test, so "generation drifts after
the first token" can't happen. The ragged chunkwise scan that replaces it is Phase 5, and no
speed claim is made about this path.

The conv window holds the last 4 pre-convolution inputs, matching HF's
`causal_conv1d_update`. Holding post-conv outputs instead decodes fluently and wrongly.

### 5.2 Attention

The conventional mechanism: each new word compares itself against every previous word and
pulls in a weighted blend. Exact, and the stored history grows one entry per word. There are
only 8 of these out of 32, which is why the growth is tolerable.

Four departures from a textbook Llama block, all silent if missed
([`attention.py`](../braid/model/attention.py)):

`head_dim` is 256 with 16 heads, over `hidden_size` 4,096 on the 9B and 2,560 on the 4B. Every
reshape uses the configured value and none derives it — see §2 for why deriving it looks fine
on the target and shreds the parity model.

Output gating. `q_proj` emits `2 × n_heads × head_dim = 8192` and the split is **per head**:
`[q_h0 | gate_h0 | q_h1 | gate_h1 | …]`, not `[all_q | all_gate]`. braid views it as
`[B, T, H, 2D]` and chunks the last axis. Splitting the flat 8,192 down the middle instead
pairs head *h* with the gate of head *h/2*, which produces plausible output from the wrong
model. The gate goes on as `o *= sigmoid(gate)` after attention and before `o_proj`.

q/k RMSNorm over the head dim, before rope, with the `1 + W` gamma already folded at load.

Partial rope: only the first 64 dims of each 256-wide head rotate and the top 192 pass
through. Rotating all 256 isn't a shape error, it's a 0.37 relative-L2 regression that no
shape check will catch.

`RotaryEmbedding` computes the text case directly. HF's module is MRoPE, expanding positions
into temporal/height/width grids and interleaving their frequencies, but for text HF
broadcasts the same row three times so the interleave selects between equal values and does
nothing. `test_rope_matches_hf` pins the equivalence rather than assuming it.

**Decode attention doesn't use SDPA.** `head_dim = 256` disqualifies every fused backend on
this box: flash, mem-efficient and cuDNN all decline with *"head_dim should be no more than
128"* and PyTorch falls back to the math backend. That backend `repeat_interleave`s K and V
4× for GQA and runs in fp32, which at B=16 measured 1.70 ms/step replicating K and V and
1.38 ms scaling the expanded key, against 1.30 ms of actual bmm. `grouped_decode_attention`
groups the *query* instead — head *h* attends KV head `h // groups`, which is what a
`[B, KVH, G, D]` reshape already says — and leaves K and V at their stored width and dtype.
That was worth +38.6% at B=16.

Prefill still calls SDPA, and still refuses the chunked case (`T > 1` onto a non-empty cache
with no explicit mask) rather than masking wrongly. SDPA's `is_causal` aligns top-left, which
is only correct when query length equals key length.

### 5.3 The two norms

[`norm.py`](../braid/model/norm.py) has two functions and the difference between them is
worth a lot of perplexity:

| | stores | effective gamma | used by |
|---|---|---|---|
| `rms_norm` | a delta | `1 + W`, folded at load | `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` |
| `rms_norm_gated` | gamma directly | `W` | `linear_attn.norm` only |

Both reduce in fp32 and cast on the way out. `rms_norm_gated` normalises first and gates
second, the opposite order from the Mamba2 path in most reference implementations:

```
inv_rms = rsqrt(mean(y²) + eps)          ← eps inside the sqrt, after the mean
out     = y · inv_rms · gamma · silu(gate)
```

`gamma` is `[head_dim] = [128]`, shared across all 32 heads, not a `[4096]` per-inner-dim
vector.

---

## 6. Numerics

braid has to reproduce the reference implementation's numbers to a lot of decimal places,
and that isn't fussiness. A model that's slightly wrong doesn't crash and doesn't produce
obvious nonsense. It produces text that reads fine and is quietly worse, and nothing warns
you. The only way to know is to diff against something known-good, on real weights,
automatically, on every commit.

The reference is Hugging Face `transformers`, not the C++ engine, because HF is what the
checkpoint was trained against. Always compare against the HF implementation that shares
braid's numerics: bf16 against `sdpa`, fp32 against `eager`.

### Where the gate belongs

rel L2 ≤ 5e-3 with cosine ≥ 0.99999 is a *single-layer* threshold. On a 32-layer bf16 stack
it measures accumulated rounding rather than correctness: `layer_trace_diag.py` shows the
residual growing smoothly from 1.9e-4 and levelling near 1e-2 with no step at any layer,
while the same forward in fp32 reads 6.4e-7. Gating the bf16 stack at 5e-3 would be gating
on depth.

So the contract splits by arm:

| arm | gate | measured |
|---|---|---|
| single sublayer | rel L2 ≤ 5e-3, cosine ≥ 0.99999 | attention fp32 bit-exact; MLP bit-exact; one GDN layer bit-identical at T ≤ 4, 4.8e-5 at T=24 |
| full stack, fp32 | same strict gate | 6.4e-7, cosine 1.000000000 |
| full stack, bf16 | greedy token identity | rel L2 8.3e-3, 100% argmax agreement |
| decode == prefill, fp32 | strict gate | 4.7e-7 (GDN), 4.9e-7 (attention) |
| perplexity, bf16 | within 20% of HF | **0.0209%** — braid 8.2376, HF 8.2393 |

Same reasoning for decode-vs-prefill: check exactness in fp32, where a cache bug can't hide
behind rounding, and check the bf16 arm on tokens.

### Settled by measurement

Each of these was an open question resolved empirically rather than by reading someone's
source, and each is pinned by a test.

The conv split is `[Q | K | V]`, Q first. Two readings of the reference engine disagreed;
HF's `torch.split(mixed_qkv, [key_dim, key_dim, value_dim])` settles it. Getting it wrong is
fluent and completely wrong, with no crash.

`linear_attn.norm` takes no `1 + W` offset, because it's the gated form. Dropping the offset
on a *plain* norm is the perplexity bug: removing it from the final norm degrades braid from
8.2376 to 11.8429. That's 1.44×, not the ~2× the roadmap predicted, which came from the 35B
and didn't survive the rescope. Still far outside the gate, so the check holds.

l2norm is the additive-`1e-6` form from HF, applied in the activation dtype and only then
widened. The reference engine uses clamped-rsqrt `rsqrtf(fmax(Σk², 1e-12))`, which differs by
a factor of 10⁶ in the degenerate case.

The gate clamps aren't applied. The reference engine clamps `A·dt` at −20 and `b_raw` at ±20;
both are no-ops on real activations (`sigmoid(20)` is 1 to within 2e-9) and HF doesn't clamp,
so parity wins.

The gamma offset is folded in fp32 rather than bf16. From `parity_report.py`: fp32 fold 0.0,
bf16 fold 1.86e-3, no offset at all 8.04e-1. The bf16 fold would clear the gate too. fp32 is
chosen because it costs nothing at one `hidden_size`-wide vector per layer and buys back three
orders of magnitude of headroom.

**`A = −exp(A_log)` deserves its own paragraph** because both published heuristics are wrong
here. The original spec said "any element ≥ 0 ⇒ raw HF". Every one of layer 0's 32 `A_log`
entries is negative, −4.22 to −0.96, so that test concludes "already transformed", skips the
`exp`, and leaves `A = −2.7` where it should be `−0.067`. That's a ~40× over-fast decay: the
state collapses toward zero silently, and the absmax tell the reference engine documents
(`0.04, 0.06, 0.40, 2.51, 110, 31680, inf` → NaN) never fires. The dtype heuristic fails too,
since this checkpoint ships `A_log` as F32. braid keys on the tensor name, which is
unambiguous for a safetensors load, and range-checks that the result is finite and negative.

### Deliberate deviations

Two places braid is *more* precise than HF, both measured, both kept:

`rms_norm_gated` holds the normalised value in fp32 where HF rounds it to the activation
dtype before applying gamma. Worth 3.1e-3 relative across a GDN layer.

`beta`'s sigmoid is taken in the activation dtype and only then widened, matching HF's own
asymmetry against `g`, which is computed entirely in fp32. Taking `beta` in fp32 instead
moves the whole layer by 4.8e-3, flat across T and across tokens. `beta` is the delta-rule
step size, so its rounding lands straight in the output.

The full bf16 stack still reaches 100% greedy token identity with HF, which is the gate that
matters. They're recorded here because they're held on purpose, not inherited by accident.

### Layout traps, settled

conv1d weights are `[C, K]` with `K` contiguous per channel. Conv bias goes on after the dot
and before SiLU. SiLU applies to all of Q, K and V, not just V. Head→group mapping is
`g = h // heads_per_group`, the grouped safetensors layout; GGUF uses tiled `g = h % n_groups`
and both are valid permutations of the same range, so a mismatch gives plausible garbage
rather than a crash. Recurrent state is `[slots, n_heads, state_size, head_dim]` fp32 with
`head_dim` fastest-varying.

### State precision

FP8 E4M3 state is refuted: the 3-bit mantissa amplifies through the delta rule and degenerates
after ~50 special tokens in multi-turn chat. Don't attempt it. FP16 state is *not* refuted and
is a live open question, see [`THESIS.md` §7](THESIS.md). fp32 is mandatory for the MVP so
parity stays unambiguous.

---

## 7. The CUDA kernels

Almost all of braid is ordinary PyTorch. Two pieces are hand-written because no library does
them the way braid needs, and they're the reason the batch axis exists at all.

They live in [`braid/kernels/csrc/`](../braid/kernels/csrc/) and are JIT-compiled on first
use at `TORCH_CUDA_ARCH_LIST=12.0a` — the arch-*conditional* target, which exposes
instructions plain `sm_120` doesn't. No `--use_fast_math`, because it turns `rsqrtf` into an
approximation and breaks fp32 parity at the tolerances the tests assert.

`gdn_decode` runs one block per (batch row, head) and one thread per `head_dim` column,
holding that state column in registers. `conv1d_decode` runs one thread per channel: slide
the window, append, dot against the weight row, add bias, SiLU. Both read the slot from
`slot_idx` on the device, which is what §4.1 is about, and both skip their slot-range
validation while a graph is capturing (the validation is a device→host copy, and that's an
illegal memory access inside a captured region).

They're available behind `Engine(use_kernels=True)` and are **not the default yet**. That
waits on an end-to-end measurement, because a kernel win that doesn't move the step is a
kernel win and not an engine win.

They are also not quite a drop-in, and the difference was measured rather than assumed
(`scripts/kernel_path_diag.py`), one GDN layer from an identical non-zero state:

```
fp32   out 2.665e-07   state 4.514e-08     ← same function
bf16   out 4.948e-03   state 2.390e-04
```

The fp32 row is the real result: the kernel computes what the torch path computes. The bf16
gap is the kernel running the conv and the l2norm in fp32 where the torch path follows HF's
bf16-then-widen order. More precise than the reference, not equal to it. Through 32 layers
that's 1.37e-2 worst case teacher-forced, with 1 argmax flip in 96 — the same magnitude as
the batching noise already characterised.

### sm_120a constraints

Each of these cost the reference engine real time. They're absorbed here rather than
rediscovered.

| | |
|---|---|
| `__launch_bounds__(HD, 2)` at HD=128 is a **ptxas miscompile** — garbage output, correct math | `gdn_decode.cu` uses `(HD, 1)`. The min-1 form is also what lets ptxas give each thread the ~128 registers `S_reg` needs without spilling. |
| Opt-in shared memory is ~99 KB, not H100's 228 KB | Query `sharedMemPerBlockOptin`; size tiles to ~97 KB. |
| No TMA. `cp.async.bulk` and `st.async .b128` to global are unavailable | Use `cp.async.ca/cg.shared.global` at 16 B. Don't port Hopper pipelines. |
| `nvcuda::wmma` compiles but lowers to HMMA, not the FP8/FP4 pipes | Hand-write `mma.sync`, or don't claim tensor cores. |
| `cudaMallocAsync` inside a captured graph crashes; any device→host copy inside capture is an IMA | Pre-allocate every workspace, keep all args device-side. |
| CUTLASS NVFP4 on sm_120 is non-deterministic under `cudaGraphExecUpdate` | Keep NVFP4 GEMMs out of exec-update if you need bitwise reproducibility. |
| WDDM silently spills to host at ~0 MiB free, 1,530 → 237 GB/s | Never size a large allocation from an estimate of another's future size. Leave ≥1 GiB free. |
| cuBLASLt returns zero algorithms for grouped GEMM on sm_120 | MoE grouped GEMM (Phase 5+) has to be CUTLASS block-scaled or hand-rolled. |
| A 247-instruction survey across CUDA 13.2→13.3 flipped 0 instructions | The ISA surface is silicon-fixed. Don't re-probe on every toolkit bump. |

---

## 8. Testing

106 tests, green on the remote 5090. They stack in four layers, each catching what the one
below can't.

`test_gdn_ref.py` checks the naive and vectorized fp32 recurrences against each other, which
is a bug in the thing everything else is measured against.

`test_gdn_decode_kernel.py`, `test_conv1d_decode.py` and `test_slot_indirection.py` check the
kernels against that oracle, and check that a captured graph survives slot reassignment. The
conv runs over 8 sequential steps with rotating slots, because a single step can't catch a
window-orientation error.

`test_hf_parity.py` and `test_attention_parity.py` check each sublayer against HF on real
weights, which is where every trap in §6 gets pinned.

`test_full_forward.py`, `test_perplexity.py`, `test_batched_decode.py` and
`test_graph_decode.py` check the whole thing: the 32-layer stack in fp32 and bf16, decode
against prefill per sublayer, perplexity against HF, 8 prompts as one B=8 batch producing
token-for-token what 8 sequential B=1 runs produce, and graph replay bit-identical to eager
at rtol=0 atol=0.

Two details worth keeping:

A deliberately inserted `.item()` has to make graph capture **fail loudly**, and there's a
test asserting it does. Otherwise the sync audit is decorative.

The ablations in `parity_report.py` show the gate discriminates rather than passing
everything: flat-halves `[q|gate]` reads 1.12, rope over all 256 dims reads 0.37, q/k norm
without the `1+W` offset reads 0.78, against a 5e-3 gate.

### Two traps that live in the harness

`module.to(bf16)` followed by `load_state_dict` truncates. The copy goes *into* the
already-bf16 parameter, so every tensor this checkpoint stores as F32 gets rounded.
`linear_attn.norm` moves 2.4e-3 and the "reference" becomes a measurably worse model than
braid. That accounted for nearly all of one GDN layer's apparent parity gap. Reference modules
are built on `meta` and loaded with `assign=True`.

`Cache.snapshot`/`restore` exist because a decode step advances state, conv, KV *and* lengths.
Rewinding only the lengths leaves two arms starting from different states, which reports as a
99.3% mismatch and is really a bookkeeping error in the test.

Measurement rules — host-health classifier, 3 processes × 3 reps, print the spread — are in
[`THESIS.md` §4](THESIS.md), and results are in [`docs/runbooks/`](runbooks/).
`test_env.py` and `test_noise_floor.py` gate the environment itself, since a 2% performance
gate against a 10% noise floor is a coin flip.

---

## 9. Status

| | |
|---|---|
| Config, loader, oracle | done |
| Batched CUDA kernels, slot indirection | done. Phase 1 gate passed: 2.58× aggregate at B=8, 104% of HBM, no L2 cliff, 10.3 µs replay across reassignment. |
| Single-layer HF parity | done |
| B=1 engine, caches, sampling | done. `"The capital of France is"` → `" Paris."`, 128 greedy tokens clean, 100% token identity with HF. |
| Perplexity gate | done. 8.2376 vs HF's 8.2393, 0.0209% apart, on 16,384 SHA-pinned tokens of wikitext-2. Peak 8.54 GiB against a 12 GB budget. |
| Batched decode over the slot pool | done. 8 prompts as one B=8 batch match 8 sequential B=1 runs, 8/8 rows, 256 tokens each. |
| CUDA-graph capture per bucket | done. Bit-identical replay on every bucket; `graphs_on/off` 2.32 / 1.86 / 1.74 against a 1.30 gate. |
| Kernels in the decode path | done, behind `use_kernels=True`, not default |
| KV block manager, chunked prefill, ragged batched prefill | **in progress** — Phase 3 item 3 |
| Scheduler, slot lifecycle, SSE server, head-to-head | not started — Phase 4 |
| Prefill scan, MoE, NVFP4, the 35B | not started — Phase 5+ |

Decode throughput as measured, median of 3 processes, spreads 0.11–1.31% against a 1.65%
noise floor:

| batch | ms/step | tok/s |
|---:|---:|---:|
| 1 | 8.124 | 123.1 |
| 8 | 11.131 | 718.7 |
| 16 | 12.065 | 1,326.2 |

Two things that follow from those numbers and shape what happens next.

At B=16 braid is at 0.705× llama.cpp's 1,880 tok/s. BF16 weights are exactly 2× the bytes of
its Q8_0 and decode is weight-bandwidth-bound, so weight quantization is the gating decision
for Phase 4. (An earlier runbook read the then-0.51× as *exactly* the weight-byte ratio and
called it a mechanism. It moved to 0.705× with no change in weight bytes, so that was a
coincidence. Retracted.)

The step is now GEMM-dominated: 8.17 ms of the 12.07 at B=16, which is 68% of the weight-read
roofline. Whether the remaining 32% is reachable is what `bench/gemm_paths.py` is for.

---

## Glossary

**Token** — roughly a word, sometimes a fragment. Models read and write tokens. "tok/s" is
tokens per second.

**Prefill** — processing the prompt before any answer comes out. Sets time-to-first-token.

**Decode** — producing the answer one token at a time. Each step needs the previous one, so
it parallelises over users, not over time.

**Batch / concurrency (B, c)** — how many requests advance in the same step.

**Aggregate vs per-stream** — aggregate tok/s is the total across all users; per-stream ITL
is how long one user waits between words. Batching improves the first and slightly worsens
the second, and quoting them as independent wins is how a 6× error gets made.

**KV cache** — the stored history an attention layer re-reads. Grows per token per user.

**Recurrent state** — the fixed-size scratchpad a GDN layer keeps. 51 MiB per user here, and
it doesn't grow.

**Slot** — a numbered place in the pool where one sequence's state lives. Sequences come and
go, slots get reused.

**Roofline / the memory wall** — the ceiling set by how fast the card reads its own memory.
At the wall, only moving fewer bytes helps.

**Bandwidth-bound** — limited by reading bytes rather than by arithmetic. Decode almost
always is.

**RMSNorm** — a rescaling step between layers. Cheap, and getting the details wrong costs
real quality.

**Parity** — how closely braid matches a known-good reference on identical input. Reported as
relative L2 and cosine similarity.

**Perplexity (PPL)** — a quality score. Lower is better. Doubling it means something broke.

**CUDA graph** — a recording of GPU work that replays without the CPU re-issuing each launch.
Fast, but it bakes in addresses, which is why slots are indexed through a device-side
`slot_idx`.

**Kernel** — one program running on the GPU. Launching one has fixed cost, which is why
fusing them or replaying from a graph matters.

**bf16 / fp32** — 16- and 32-bit floating point. Weights and activations are bf16, recurrent
state is fp32.

**sm_120a** — the 5090's architecture target. The trailing `a` is the arch-conditional
variant, which exposes instructions plain `sm_120` doesn't.

**GDN** — Gated DeltaNet, the recurrent layer type. §5.1.

**GQA** — grouped-query attention: several query heads share one KV head. 4:1 here.

**SwiGLU** — the feed-forward block essentially every modern model uses.

**MoE** — mixture of experts, where each token routes to a few of many sub-networks. On
neither Qwen3.5 checkpoint braid runs; arrives with the 35B.
