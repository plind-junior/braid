# braid

A single-GPU inference engine for **hybrid** language models — attention interleaved with a
linear-recurrent mixer (Gated DeltaNet) — built to serve them **at concurrency** on one
RTX 5090.

> **Status: the locked MVP target is MET.** It asked for ≥+25% over llama.cpp at B=16
> and ≥+100% at B=64; braid measures **+26.5% and +118.8%** — every one of the five
> independent processes clears both clauses, worst pairing +25.8%. The same head-to-head
> (`Qwen3.5-9B`, one card, one session, 5 processes per arm) reads **+68.3% at B=32,
> +141.8% at B=96, +170.4% at B=128** — and **−22.0% at B=1**, published unchanged.
> From B=1 to B=128 braid scales **58.8×** where llama.cpp scales **17.0×**, which is
> the thesis. braid matches HuggingFace to fp32 machine precision over all 32 layers and
> serves concurrent streams over SSE with continuous batching, **3,633 tok/s served at
> c=128**, prefill included.
>
> What closed the last gap was not one lever but four, each priced separately below:
> FP8 on every projection, fp16 state storage, a chunk-cached prefill scan kernel
> (14.1× single-stream prefill), and a launch diet (−43% kernel launches per decode
> step) — the last two bit-identical to what they replaced, by test.
>
> This target was published as a NO-GO twice on this page while the margin was built —
> −16.1%/+48.6% on the 4B, then +2.8%/+84.0% here — and the batch it names was never
> moved. Every number is labelled **measured** or **projected**.

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
| 1 | 157.1 | 6.37 | 1,327 GB/s | **88%** |
| 8 | 901.9 | 8.87 | 952 GB/s | 63% |
| 16 | 1,368.1 | 11.70 | 723 GB/s | 48% |
| 32 | 1,929.0 | 16.59 | 509 GB/s | 34% |
| 64 | 2,391.8 | 26.76 | 316 GB/s | 21% |
| 96 | 2,564.3 | 37.44 | 226 GB/s | 15% |
| 128 | 2,662.8 | 48.07 | 176 GB/s | **12%** |

A dense model reads its weights **once per step** regardless of batch size, so batching
should push *toward* the memory wall, not away from it. llama.cpp goes the other way: 88%
efficient at batch 1, 12% at batch 128. Its throughput is nearly flat from B=64 to B=128
(2,392 → 2,663, just +11% for twice the batch) while braid goes 5,232 → 7,199 (+38%).

**braid's thesis, in one line:**

> At concurrency, a GDN hybrid runs at a fraction of the memory wall.
> braid targets the wall.

That is the whole idea. It is a reproducible number, not a claim about anyone's code.

---

## The head-to-head

**Measured 2026-08-09 on `Qwen3.5-9B`. Every arm in one session, rotated order, 5
processes per arm per point, medians across processes.** Decode-only aggregate
throughput, KV 128..256 on all arms (`llama-batched-bench -npp 128 -ntg 128` against
braid's graphed decode seeded to the same depth).

| batch | llama.cpp Q8_0 | braid BF16 | braid FP8-all | braid FP8 + fp16 state | best delta | |
|---:|---:|---:|---:|---:|---:|:--|
| 1 | 157.1 | 84.2 (0.54×) | 122.1 (0.78×) | 122.5 (0.78×) | −22.0% | lose |
| 8 | 901.9 | 552.6 (0.61×) | 790.2 (0.88×) | 807.1 (0.90×) | −10.5% | lose |
| 16 | 1,368.1 | 1,107.6 (0.81×) | 1,656.1 (1.21×) | **1,731.2 (1.27×)** | **+26.5%** | **win** |
| 32 | 1,929.0 | 2,070.6 (1.07×) | 3,003.3 (1.56×) | **3,246.2 (1.68×)** | **+68.3%** | **win** |
| 64 | 2,391.8 | 2,785.2 (1.16×) | 4,377.9 (1.83×) | **5,232.2 (2.19×)** | **+118.8%** | **win** |
| 96 | 2,564.3 | — | 5,093.1 (1.99×) | **6,200.5 (2.42×)** | **+141.8%** | **win** |
| 128 | 2,662.8 | — | — | **7,199.2 (2.70×)** | **+170.4%** | **win** |

Spreads over the five processes: llama.cpp 1.73%, braid 0.25–0.99%. **braid now crosses
llama.cpp between B=8 and B=16** and the losing rows are published unchanged.

The em-dashes are not omissions: those configurations **do not fit the card**. BF16 runs
out above B=64 and FP8 with an fp32 state pool runs out above B=96. Reaching B=128 at all
is a result of the fp16 state pool, not a separate choice of batch.

The thesis is about the *shape* of the curve, and the shape holds. From B=1 to B=128
braid scales **58.8×** where llama.cpp scales **17.0×** — braid keeps converting batch
into throughput long after llama.cpp has stopped. That is the whole claim, and it is a
measurement.

**The locked MVP target is met, and the margin survives the noise.** It asks for ≥+25%
at B=16 and ≥+100% at B=64. At B=16 the margin over the threshold is 1.5pp, so the
verdict is checked per process rather than on medians: pairing each braid rep with its
same-rep llama.cpp run gives +25.9 / +26.6 / +26.5 / +26.3 / +26.5%, and the most
adversarial pairing in the sample (braid's slowest against llama.cpp's fastest) is
**+25.8%**. Every reading clears the clause. B=64 clears its clause by 18.8pp against
sub-2% spreads. This table was published as a NO-GO twice — at −16.1%/+48.6% on the 4B
and +2.8%/+84.0% on this model — while the target stayed pinned to the batches it
names; the earlier re-scoped target (≥+20% at B=32, ≥+50% at B=64) is now academic.

### The win survives longer prompts

Everything above is 128-token prompts. The obvious objection is that real workloads
carry more, and deeper KV is a per-sequence cost that grows with batch. Measured at
**4× the prompt depth** (`-npp 512 -ntg 64`, 3 processes per point, spreads ≤0.94%,
same session, rotated arms):

| batch | llama.cpp Q8_0 | braid FP8 + fp16 state | delta | at npp=128 |
|---:|---:|---:|---:|---:|
| 1 | 155.4 | 122.5 | −21.2% | −22.0% |
| 16 | 1,323.8 | 1,671.5 | **+26.3%** | +26.5% |
| 64 | 2,293.7 | 4,742.9 | **+106.8%** | +118.8% |
| 128 | 2,564.4 | 6,343.0 | **+147.3%** | +170.4% |

The low-batch picture does not move. The high-batch margin compresses 12–23 points —
decode attention reads KV 512..576 deep instead of 128..256, and both engines pay it —
but braid still clears 2.4× at B=128, still scales 51.8× against llama.cpp's 16.5×,
and **both clauses of the locked target hold at the longer prompt too**.

### The lever was weight bytes, and it is now spent

The 4B run named the cause of the low-batch deficit: braid read BF16 where llama.cpp
read Q8_0. Extending FP8 W8A8 from the MLP alone to every projection closed it.

| | decode-step weight bytes | vs llama.cpp | B=16 | B=64 |
|---|---:|---:|---:|---:|
| braid BF16 | 14.78 GiB | 1.88× | 0.81× | 1.16× |
| braid FP8-all | 7.40 GiB | 0.94× | 1.21× | 1.83× |
| **braid FP8-all + fp16 state** | **7.40 GiB** | **0.94×** | **1.27×** | **2.19×** |

(The FP8-MLP intermediate arm — 10.28 GiB, and the row that first localised the deficit
to bytes — was retired from the sweep once FP8-all strictly dominated it.)

llama.cpp's per-step figure (~7.87 GiB) is **derived**, not measured: the 8.87 GiB GGUF
less its ~1.01 GiB embedding table, which is gathered rather than swept. braid's is
measured by `Engine.step_bytes()`.

So braid now reads *fewer* weight bytes per decode step than llama.cpp does, and the
remaining B≤8 deficit is no longer bytes. What it *is* was measured rather than guessed —
see below.

### Where a decode step's milliseconds actually go

`scripts/step_profile.py`, `Qwen3.5-9B`, FP8-all, fp16 state. The headline is not the
hot-kernel list, it is `sum(kernel device time)` against the wall-clock step: a large gap
means fuse, a small gap means the kernels themselves are the cost.

| | B=1 | B=32 |
|---|---:|---:|
| step (CUDA events, unprofiled) | 9.748 ms | 12.474 ms |
| sum of kernel device time | 8.869 ms (**91%**) | 11.078 ms (**89%**) |
| gap | 0.878 ms (9%) | 1.396 ms (11%) |
| kernels per step | 2,496 | 2,801 |

**This refuted the standing hypothesis.** The B=1 deficit had been attributed to launch
overhead; the step is 91% busy, so fusing launches is worth at most 9% and the graphs are
already doing their job. Of the busy time at B=1, GEMMs are 5.646 ms — against a
weights-only floor of 5.24 GiB/1,514 GB/s = **5.24 ms**, so **the GEMMs are running at
about 93% of the memory wall**.

The cost is the 3.2 ms of non-GEMM work wrapped around them:

| bucket | B=1 ms | % of busy | launches |
|---|---:|---:|---:|
| gemm | 5.646 | 63.7% | 233 |
| **elementwise** | **2.195** | **24.7%** | **1,873** |
| reduce | 0.623 | 7.0% | 234 |
| gdn scan | 0.089 | 1.0% | 24 |
| conv · attention · index/copy · other | 0.318 | 3.6% | 132 |

The single largest non-GEMM item is 0.733 ms of plain `direct_copy` over **452 launches**
— casts and `contiguous()` calls, not arithmetic. Dynamic activation quantization is the
next, at ~0.74 ms across ~129 amax/abs/scale/cast groups. Neither is a kernel that needs
to be faster; both are work that should not exist. That is the real lever list at low
batch, and it is nothing like the one this section previously asserted.

### Acting on that list bought 15.9% at B=1 — measured, four arms

Two of the launch sinks were removed by moving the work inside CUDA kernels, and both
changes are **bit-identical to what they replace, by test rather than argument**:

- **Fused activation quantization** (`quant_act.cu`): the 9-kernel torch spelling of
  quantize-one-activation becomes 2 launches, at ~129 calls per step. The bit-identity
  gate caught a real subtlety on its first run: torch divides a tensor by a CPU scalar
  as a *reciprocal multiply*, not a true division, and the kernel's correctly-rounded
  IEEE division came out exactly 1 ulp different on ~60% of inputs. The kernel now
  reproduces torch's approximation, because the contract is "the numbers perplexity was
  measured under", not "the mathematically better rounding". The amax reduction is also
  NaN-propagating by construction — `fmaxf` silently swallows NaN where `torch.amax`
  propagates it, and a quantizer that hides divergence evidence would turn a diverged
  run into plausible-looking fp8.
- **In-kernel gates** (`gdn_decode_raw` / `gdn_prefill_raw`): `alpha`/`beta` — a
  sigmoid, a softplus, an exp and their casts, ~8 launches per GDN layer per step — are
  computed inside the scan kernels from the raw projections. Gate acquisition is a
  policy template over the *same* recurrence lines, so prefill and decode remain one
  function by construction. In prefill, `seq_lens` replaces the caller's `torch.where`
  pad mask, asserted to produce bit-identical states.

| arm (B=1, 9B, fp8-all, fp16 state) | ms/step | launches/step |
|---|---:|---:|
| baseline (previous shipped config) | 9.751 | 2,496 |
| + fused quantization | 8.405 | 1,593 |
| + in-kernel gates | 9.559 | 2,328 |
| **both (shipped)** | **8.199 (−15.9%)** | **1,425 (−43%)** |

The arms compose additively to within 1%, and the baseline arm reproduces the
previously published 9.748 ms to 3 ppm — same harness, same session structure. Since
both levers are asserted bit-identical to their fallbacks, this is a launch-count
change with **no numerics consequence**: perplexity and every parity gate are
unaffected by construction, and the full 26-module suite passes on both models.

**The third cut, and a measurement that nearly lied.** The remaining copies — the conv
input cast and the q/k/v `.contiguous()` copies feeding the scan — were killed by
letting the conv kernel read bf16 directly and the scan kernels read strided
column-slice views (bit-identical by test, like the others; ~96 launches/step
removed). A cross-session comparison first read this as **−2% at B=64/128**, and a
regime-split "fix" was built before the claim was checked the only valid way: a
same-session A/B of views against copies, with B=8 as a shared-code control. The
control agreed to 0.1%; the views **beat** the copies by ~1% at B=64/128. The −2% was
day-to-day session drift wearing a regression's clothes, the split was reverted, and
the views ship everywhere. Cross-session deltas under ~2% are not attributable to
code on this box — that is now written down where it was almost violated.

**What FP8 costs, measured on the 9B:** perplexity 7.1272 → 7.0773, **−0.70%**. A
*decrease* is not evidence that quantization helped — it says the cost sits below what a
16k-token corpus can resolve. Two groups are deliberately excluded: `in_proj_a` /
`in_proj_b`, because `a_raw` is exponentiated and fp8's three mantissa bits would land
inside an exponent for 0.4% of a layer's bytes; and `embed_tokens`, which is gathered
rather than swept.

### The second lever: fp16 recurrent state

Weight bytes are a fixed cost per step. The recurrent state is a **per-sequence** cost,
so it is the term that decides where the batch curve stops. On the 9B it is 48 MiB per
sequence — at B=64 that is 6.14 GiB of traffic against 7.40 GiB of fp8 weights, i.e. 45%
of the step, and it is what runs the card out of memory at B=96.

Storing it in fp16 halves both. **Only the storage narrows:** both the torch path and
the CUDA kernel widen the column to fp32 on load and narrow once on store, so what this
costs is one rounding per step, not a 16-bit scan.

| | per-slot state | B=64 | B=96 | B=128 | perplexity |
|---|---:|---:|---:|---:|---:|
| fp32 pool | 0.047 GiB | 4,377.9 | 5,093.1 | out of memory | 7.1292 |
| **fp16 pool** | **0.023 GiB** | **5,232.2** | **6,200.5** | **7,199.2** | 7.1306 (**+0.02%**) |

Pricing it required fixing the harness first: the perplexity path ran **cacheless**, so
the state pool was never touched and a narrower pool would have scored identically for
the wrong reason. It now feeds each window through a cache in 128-token pieces — sixteen
state round-trips per window — and both arms use that protocol, or the comparison would
be pricing "chunked vs whole" and calling it "fp16 vs fp32".

bf16 was measured too and is 8× worse on the state residual (2.30e-3 against fp16's
2.93e-4), which is what the mantissa counts predict. fp16 is the one that ships.

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

| batch | BF16 | FP8-all | FP8 + fp16 state |
|---:|---:|---:|---:|
| 1 | 16.9 | 9.6 | 9.5 |
| 16 | 20.0 | 12.6 | 11.5 |
| 32 | 23.2 | 15.8 | 13.6 |
| 64 | 29.6 | 22.2 | **17.7** |
| 96 | — | 28.6 | 21.9 |
| 128 | — | — | **26.0** |

Card total 31.36 GiB. BF16 at B=64 leaves 1.8 GiB of headroom; FP8 with an fp16 state
pool leaves 13.7 at the same batch, and still 5.4 at B=128.

### What is not measured here

This table is **decode only**. braid's end-to-end serving throughput including prefill
is lower and is reported separately below; llama.cpp's S_TG column excludes its prefill
too, so the comparison is like-for-like but neither number is a full serving result.
braid also has no prefix caching, which a multi-turn benchmark would reward llama.cpp
for and which is a scoped gap rather than a defect.

## Serving, end to end

braid's own service, prefill included — `Qwen3.5-9B`, 128-token prompts, 64 new tokens,
graphs on, FP8-all with an fp16 state pool. **The two arms are the same binary and the
same weights, differing in one attribute**, so the difference is the chunk scan and
nothing else:

| c | tok/s, loop | tok/s, **chunk scan** | | prefill tok/s | TTFT p50 | prefill % of wall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 56.9 | **97.3** | 1.71× | 255 → **3,585** | 514 → **47 ms** | 45% → 5% |
| 8 | 405.9 | **614.3** | 1.51× | 2,047 → **14,324** | 1,776 → **921 ms** | 40% → 9% |
| 16 | 828.3 | **1,165.9** | 1.41× | 4,073 → **14,045** | 1,755 → **1,040 ms** | 41% → 17% |
| 32 | 1,171.7 | **1,888.4** | 1.61× | 4,253 → **13,601** | 2,732 → **1,406 ms** | 55% → 28% |
| 64 | 1,129.5 | **2,561.0** | 2.27× | 3,079 → **12,913** | 6,318 → **2,265 ms** | 73% → 40% |

**The curve used to fall at c=64 and now it does not.** 1,171.7 → 1,129.5 became
1,888.4 → 2,561.0. That inversion was the whole reason the kernel exists: decode alone
kept climbing to B=64 but end to end it did not, because prefill was taking 73% of the
wall clock.

**The current service, after the launch diet, to c=128** — the A/B above priced the
chunk scan in isolation on the pre-diet binary; this is what ships today:

| c | tok/s | TTFT p50 | ITL p50 | ITL p99 | prefill % | VRAM GB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 116.4 | **35 ms** | 8.29 ms | 8.59 ms | 5% | 10.4 |
| 8 | 729.2 | 779 ms | 10.05 ms | 12.07 ms | 9% | 12.1 |
| 16 | 1,411.9 | 862 ms | 9.41 ms | 13.40 ms | 17% | 13.8 |
| 32 | 2,274.8 | 1,169 ms | 10.06 ms | 17.93 ms | 28% | 16.3 |
| 64 | 3,072.4 | 1,864 ms | 12.68 ms | 28.32 ms | 38% | 21.5 |
| 96 | 3,372.4 | 2,630 ms | 15.95 ms | 39.34 ms | 42% | 28.0 |
| 128 | **3,633.0** | 3,337 ms | 18.39 ms | 49.42 ms | 46% | 25.9 |

Host healthy at every point from c=8 up (2,745–2,827 MHz SM, 416–600 W); c=1 is
power-depressed because a single stream genuinely does not saturate the card — that is
the measurement, not a throttled host. c=96/128 are reachable because the scheduler's
graph-bucket ladder now extends there (and an off-rung capacity captures its own rung
rather than crashing mid-serve).

### Server against server

The decode head-to-head above isolates the step; this races the actual services —
braid's HTTP/SSE server against `llama-server`, same box, same binary flags as the
published decode arm plus capacity provisioning (`-np`, `--threads-http`; braid's
accept backlog got the identical treatment). One stdlib-asyncio client drives both
wire formats with identical random token-id prompts (llama.cpp has prefix caching,
braid does not; fresh prompts keep that lever out), ramped connects, and a per-point
guilt figure: the client publishes its own CPU/wall ratio, and every published point
shows it idle. 3 reps per point, 128-token prompts, 64 generated, medians:

| c | llama-server | braid | delta | TTFT p50 (llama → braid) | ITL p50 (llama → braid) |
|---:|---:|---:|---:|---:|---:|
| 1 | 125.4 | 115.9 | −7.6% | 92 → 35 ms | 6.6 → 8.3 ms |
| 8 | 377.1 | 649.7 | **+72%** | 676 → 84 ms | 10.5 → 10.6 ms |
| 16 | 379.2 | 1,156.4 | **+205%** | 1,355 → 88 ms | 16.4 → 10.8 ms |
| 32 | 402.3 | 1,746.1 | **+334%** | 2,295 → 105 ms | 35.9 → 12.6 ms |
| 64 | 399.2 | 2,288.9 | **+473%** | 3,682 → 124 ms | 61.8 → 17.9 ms |
| 128 | 47.7 | 2,795.1 | — | 287,240 → **158 ms** | 127.0 → 26.9 ms |

Spreads: braid 5.0%, llama-server 28.1%. No request errors, no client-bound points.

**The serving gap is much wider than the decode gap, and the reason is visible in the
TTFT column.** llama-server's time-to-first-token grows *linearly* at ~57 ms per
concurrent stream — the signature of serialized prompt processing — and its aggregate
sits flat near 400 tok/s from c=8 to c=64 while the *same binary* decodes 2,392 tok/s
at B=64 in its own `llama-batched-bench`. Its kernels are not the bottleneck; its
scheduler under request churn is. That is the gap braid was designed around: chunked
ragged prefill co-scheduled with decode holds braid's TTFT at 84–158 ms across the
whole curve. The c=128 row's delta is left unnumbered because llama-server is outside
its working envelope there (four-minute TTFTs are a failure mode, not a throughput);
the defensible headline is c=8–64: **+72% to +473% served**.

The one number braid loses (−7.6% at c=1) and the HTTP tax (2,795 served at c=128
against 3,633 in-process) are printed unchanged. This measurement supersedes the
earlier caveat that no server-vs-server comparison had been run.

**The control is ITL.** The chunk scan touches prefill only, so decode must not move —
and it does not: ITL p50 reads 9.83 / 12.00 / 11.49 / 12.16 / 14.80 ms with the kernel
and 9.83 / 12.00 / 11.49 / 12.18 / 14.81 ms without, identical to three significant
figures at every concurrency. A speedup that had also moved ITL would have been
measuring something other than what it claims.

**What the kernel is.** One launch per layer for a `[B, T]` chunk instead of one Python
iteration per column, with the recurrent state held in registers across the whole chunk.
It is deliberately **not** a WY / UT-transform / tensor-core chunkwise scan: it is the
decode kernel's scalar recurrence line for line, with the loop moved inside. That choice
is what makes the gate possible — the kernel is asserted **bit-identical** to the decode
kernel applied T times, which no chunkwise reformulation could pass, and it is why
prefill and decode remain the same function rather than two implementations that agree
to a tolerance.

The host-health sampler flags c=1 and c=8 as power-depressed (261–348 W against a 400 W
floor) in both arms. At those concurrencies the GPU is genuinely not saturated — that is
the measurement, not a throttled host.

**Where the batching part came from.** Prefill used to run one sequence per forward. The
old loop cost one iteration per *column*, not per token, so a batch of sixteen rows cost
what one row cost; batching the rows moved the 4B from 122.1 to 850.1 tok/s at c=16. That
change made prefill cheaper per row without making any arithmetic faster. This one makes
the arithmetic faster, which is why it also helps at c=1, where batching could not.

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
| **Chunk-cached prefill scan kernel** (`gdn_prefill`) | **done** — **bit-identical** to the decode kernel applied T times; prefill 255 → 3,585 tok/s at c=1 |
| FP8 W8A8 by group — `mlp`, `attn`, `gdn`, `head` (`quant="all"`, opt-in) | **done** — decode-step weights **halved**, 14.78 → 7.40 GiB, for −0.70% perplexity |
| **fp16 recurrent state pool** (storage only; the scan stays fp32) | **done** — per-slot 47 → 23 MiB, +0.02% perplexity, and it is what makes B=128 fit |
| Continuous-batching scheduler, SSE server, per-row sampling, release on disconnect | **done** |
| Perplexity harness · noise-floor + host-health harness · llama.cpp baseline | **done** |
| Paged KV blocks | deferred, with arithmetic — the MVP runs two orders below the threshold |
| Prefix caching · preemption | **not started** |

**252 tests across 26 modules, all green** on both `Qwen3.5-9B` and `Qwen3.5-4B`, run on
a remote RTX 5090 via `make test-remote` (or `scripts/test_isolated.sh`, one module per
process — at 9B two module-scoped engines no longer share a card).

**Correctness on the 9B, all measured:** perplexity **7.1272** vs HF **7.1312**
(**0.0564%**); full 32-layer fp32 forward vs HF **4.538e-07**, cosine 1.000000000, tokens
identical; full-stack fp32 decode-vs-prefill **1.478e-06**, tokens identical; fp32 token
identity at B=2/4/8; graph replay bit-identical at `rtol=0, atol=0` for bf16 and for every
FP8 group; ragged batched prefill bit-identical under a changed pad token id; the chunk
prefill kernel **bit-identical** to the decode kernel applied T times, exactly inert on
pad columns, and invisible to where a chunk boundary falls.

**What the CUDA path costs in perplexity, measured:** 7.1292 → 7.1312, **+0.03%**, on the
same corpus, window and 128-token chunk protocol, with only the implementation changing.
The kernels l2-normalise q and k in fp32 where HF and the torch path normalise in the
activation dtype — a documented deviation the decode kernel already carried. The parity
gates run on the torch path, so this is the one number that prices the kernels, and it is
now a measurement rather than an argument.

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

The scan **saturates memory bandwidth at B≈4**. It is at the wall and cannot be made
faster, only smaller — which is what fp16 state does, and it is now measured rather than
open: half the bytes for +0.02% perplexity, and the difference between stopping at B=64
and reaching B=128.

## Quickstart

Everything runs on a remote GPU box; nothing runs locally. A local `pytest` that dies at
import means you forgot `-remote`, not that the code is broken.

```bash
export BRAID_SSH_KEY=~/.ssh/rtx5090 BRAID_SSH_HOST=root@host BRAID_SSH_PORT=22

make provision        # torch, pytest, numpy, ninja, ccache
make test-remote      # 252 tests
make lint             # ruff — this one runs locally and costs nothing
make bench-noise      # measured noise floor + host-health verdict
make bench-scaling    # the scan scaling curve, COLD and HOT

# Serving throughput, ITL and TTFT
./scripts/remote.sh 'python3 -B -m braid.bench.serve_bench --quant all --state-dtype fp16'

# What the chunk prefill scan is worth: the same binary with it turned off
./scripts/remote.sh 'python3 -B -m braid.bench.serve_bench --quant all --state-dtype fp16 \
  --no-chunk-prefill'

# Where a decode step's milliseconds go: kernels, or the gaps between them
./scripts/remote.sh 'python3 -B scripts/step_profile.py --batch 1'

# Competitor baseline
./scripts/remote.sh 'bash scripts/provision_llamacpp.sh'
./scripts/remote.sh 'MODEL=Qwen3.5-9B bash scripts/stage_model.sh'
./scripts/remote.sh 'REPS=5 NPL=1,8,16,32,64,96,128 BATCHES="1 8 16 32 64 96 128" \
  GGUF=/root/models/Qwen3.5-9B-Q8_0.gguf BRAID_MODEL_DIR=/root/models/Qwen3.5-9B \
  BRAID_ARMS="bf16:|fp8-all:--quant all|fp8-fp16state:--quant all --state-dtype fp16" \
  bash scripts/head_to_head.sh'
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
  plain chunk-cached scalar loop. This rules out the tensor-core ladder, **not chunking
  itself** — and the chunk-cached scalar scan it left open is now built (`gdn_prefill`),
  worth 14.1× on single-stream prefill. Keeping it scalar is also what allows its gate to
  be bit-identity against the decode kernel rather than a tolerance.
- **NVFP4 on GDN projections.** Measured −9% to −20% decode; tuned FP16 wins on those shapes.
- **4-bit weights at the batches braid wins at.** `torch._weight_int4pack_mm` *does* exist
  and is fast where braid is weak — 3.13× over FP8 on the MLP at M=1, 1.66× at M=8 — but it
  is a GEMV kernel and collapses where braid is strong: **0.22× at M=32 and 0.12× at M=64**.
  It wins exactly where braid loses and loses exactly where braid wins. Even at low batch it
  would not be an honest headline without moving the competitor to `Q4_K_M` at the same
  time; 4-bit braid against an 8-bit llama.cpp is a precision choice wearing a speedup's
  clothes. Measured, `scripts/fp4_probe.py`, kept as a record rather than a plan.
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
