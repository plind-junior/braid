# Checkpoint — 2026-08-10 (the serving wave)

Where braid is, what changed, and what to pick up next. Written to be read cold.

**"The 9B" throughout this doc means `Qwen3.5-9B`** — the model braid ships on and the one
every number below is measured against. `Qwen3.5-4B` is the parity model the test suite loads
by default; it was the MVP target through Phase 3 and the engine moved up on 2026-08-09. Shapes
for both: [`ARCHITECTURE.md` §2](ARCHITECTURE.md).

**HEAD:** `5cb3278` — all committed and pushed.
**PR AUTOMATION (late addition, modeled on sparkinfer):** PRs are now reviewed
and *measured* automatically. `.github/`: claude-review.yml (AI first-pass
review, needs the `ANTHROPIC_API_KEY` repo secret — not yet set), pr-gate.yml
(non-maintainer PRs touching braid//tests/ must tick the RTX 5090 box; fails
the check, does not close), PULL_REQUEST_TEMPLATE.md, eval-policy.yml.
`scripts/pr_eval_bot.py` is the teeth: per eligible PR head it starts the box,
runs the suite on the merged tree, benches PR-vs-main **same-session**
(graphed-kvbucket, B=16/64, median of 3, alternated arm order), comments the
measured table, labels eval:pass/noise/reject/error, stops the box. Cron:
`*/30 * * * * scripts/pr_eval_cron.sh` (installed in this machine's crontab,
flock-guarded, log `~/braid-pr-eval.log`). Skips the cycle if the box is
already running — a running box is somebody's session. Null-tested live: no-op
PR #1 → eval:noise at −0.0%, 0.14% spread over 4 interleaved runs, box
auto-stopped. Policy layer unit-tested GPU-free in tests/test_pr_eval_bot.py.
**NEW THIS SESSION, all measured and published in the README:**
  * **Server against server** (the announcement's standing caveat, closed):
    braid's HTTP service vs `llama-server`, one client, both sides provisioned,
    3 reps, client provably idle. **+72% at c=8 → +473% at c=64**; TTFT 84–158 ms
    against 676 ms–4 min. llama-server's TTFT grows linearly ~57 ms/stream
    (serialized prompt processing) while its own batched-bench decodes 6× its
    served rate — the serving gap is its scheduler, which is the thesis. The
    harness fixed BOTH servers first: braid's listen backlog of 5 (358 resets at
    c=128 → zero) and the same httplib default on llama's side (TCP retransmit
    ladders; survives --threads-http, so structural for it).
  * **npp=512 sweep**: +26.3% @ B=16, +106.8% @ B=64, +147.3% @ B=128 — both
    locked-target clauses hold at 4× the published prompt depth.
  * **The copy kill**: conv reads bf16 in-kernel, scan reads strided views;
    ~96 launches/step removed, bit-identical by test. A cross-session −2% at
    B=64/128 was nearly published as a regression; a same-session A/B with a
    shared-code control (agreeing to 0.1%) refuted it — views BEAT copies by
    ~1% there. **Rule, now written in gdn.py: cross-session deltas under ~2%
    are not attributable to code on this box.**
**THE LOCKED MVP TARGET IS MET:** ≥+25% at B=16 and ≥+100% at B=64 measured at
**+26.5% and +118.8%**, every one of 5 independent processes clearing both
clauses (worst adversarial pairing +25.8%). Twice published as a NO-GO on the
way here; the batches the target names were never moved.
**tests:** 259 across 27 modules, green on **both** models, plus 25 GPU-free
eval-bot policy tests (tests/test_pr_eval_bot.py) green locally and in CI.
**GPU:** vast.ai instance `47055458` — **stopped** (verify with `make gpu-status`
before assuming; it bills $0.79/hr when running).

## 0. The evening wave, measured at B=1 on the 9B (4 arms, one session each)

| arm | ms/step | launches/step |
|---|---:|---:|
| baseline (yesterday's shipped config) | 9.751 | 2,496 |
| + fused activation quantization (`quant_act.cu`) | 8.405 | 1,593 |
| + in-kernel gates (`gdn_decode_raw` / `gdn_prefill_raw`) | 9.559 | 2,328 |
| **both** | **8.199 (−15.9%)** | **1,425** |

Additive to within 1%. Baseline reproduces the published 9.748 to 3 ppm.
Implied B=1: 102.6 → ~122 tok/s against llama.cpp's 157.

Both levers are **bit-identical to their fallbacks by test, not argument**
(`test_quant.py` 60/60, `test_gdn_raw_gates.py` 16/16). One bug each found
before shipping: the quantizer's first version used true IEEE division for the
scale where torch's tensor-by-cpu-scalar div is reciprocal-multiply — caught by
the bit gate as an exactly-1-ulp-low scale on ~60% of inputs — and the first
`fmaxf`-based reduction would have SWALLOWED NaN where torch propagates it,
caught by inspection and now pinned by `test_a_nan_activation_survives_quantization`.

---

## 1. State of the engine

Phases 2, 3 and 4 are complete. Phase 5 items 5 (ragged batched prefill) and 7's fp16
half are **done**, and the chunk-cached prefill scan that item 5 left explicitly open is
now built and measured.

| | |
|---|---|
| loads | `Engine.from_pretrained(path)` — 32 layers, 24 GDN + 8 gated attention |
| runs | `forward` (prefill, ragged batched, **chunk-kernel scan**), `decode_step` (sync-free, graph-capturable), `hidden_states` |
| generates | `generate`, `generate_batch` — slot-pooled, per-row sampling |
| accelerates | CUDA GDN decode **and prefill** kernels, conv kernel, CUDA-graph buckets to B=128, **FP8 W8A8 by group**, **fp16 state pool** |
| serves | `braid/serve/` — continuous batching, batched chunked prefill, per-row sampling, SSE, slot release on disconnect |
| does **not** have | paged KV blocks (deferred with arithmetic); prefix caching; preemption (deliberate) |

**The shipping configuration is** `use_kernels=True, quant="all", state_dtype=fp16`.

**Correctness on the 9B, all measured:** perplexity 7.1272 vs HF 7.1312 (0.0564%); full
32-layer fp32 forward vs HF 4.538e-07, cosine 1.000000000, tokens identical; full-stack
fp32 decode-vs-prefill 1.478e-06; fp32 token identity at B=2/4/8; graph replay
bit-identical at `rtol=0, atol=0` for bf16 and every FP8 group; ragged prefill
bit-identical under a changed pad token id; **the chunk prefill kernel bit-identical to
the decode kernel applied T times**.

## 2. Performance — the published curve

The v2 sweep, run after the launch diet landed: 5 processes per point, one
session, rotated order, decode-only aggregate, KV 128..256 on all arms.
Spreads: llama.cpp 1.73%, braid 0.25–0.99%.

| B | llama.cpp Q8_0 | braid BF16 | braid FP8-all | braid FP8 + fp16 state | best |
|---:|---:|---:|---:|---:|---:|
| 1 | 157.1 | 84.2 | 122.1 | 122.5 | −22.0% |
| 8 | 901.9 | 552.6 | 790.2 | 807.1 | −10.5% |
| 16 | 1,368.1 | 1,107.6 | 1,656.1 | **1,731.2** | **+26.5%** |
| 32 | 1,929.0 | 2,070.6 | 3,003.3 | **3,246.2** | **+68.3%** |
| 64 | 2,391.8 | 2,785.2 | 4,377.9 | **5,232.2** | **+118.8%** |
| 96 | 2,564.3 | — | 5,093.1 | **6,200.5** | **+141.8%** |
| 128 | 2,662.8 | — | — | **7,199.2** | **+170.4%** |

Em-dashes are configurations that **do not fit the card**, not missing measurements.
BF16 stops above B=64; FP8 with an fp32 state pool stops above B=96.

Scaling B=1 → B=128: braid **58.8×**, llama.cpp **17.0×**. The crossing moved
from B=16 to between B=8 and B=16.

**Served, prefill included** (9B, 128-token prompts, 64 new, FP8-all + fp16
state, current binary), now to c=128:

| c | tok/s | TTFT p50 | ITL p50 | prefill % |
|---:|---:|---:|---:|---:|
| 1 | 116.4 | 35 ms | 8.29 ms | 5% |
| 16 | 1,411.9 | 862 ms | 9.41 ms | 17% |
| 32 | 2,274.8 | 1,169 ms | 10.06 ms | 28% |
| 64 | 3,072.4 | 1,864 ms | 12.68 ms | 38% |
| 128 | **3,633.0** | 3,337 ms | 18.39 ms | 46% |

Host healthy at every point from c=8 up; c=1 is power-depressed because one
stream genuinely does not saturate the card. The chunk-scan A/B that priced the
prefill kernel in isolation is preserved in the README.

## 3. What changed this session

**fp16 recurrent state.** `StateIO<T>` in the kernel and a `.float()` / `.to(dtype)` pair
in the torch path: storage narrows, arithmetic stays fp32. +0.02% perplexity, per-slot
47 → 23 MiB, and it is the only reason B=128 fits.

**`gdn_prefill`.** One launch per layer per chunk instead of one Python iteration per
column, state resident in registers. Line-for-line the decode kernel's recurrence, which
is what makes its gate bit-identity rather than a tolerance. ptxas: 255 registers, 188 B
spill stores (decode: 76 B), 1024 B smem. Costs +0.03% perplexity for the fp32 l2norm
deviation the decode kernel already carried.

**Two harness bugs that would have produced wrong conclusions.** Perplexity ran
*cacheless*, so a narrower state pool was literally unused and would have scored
identically for the wrong reason — `--chunk` fixes it. And `decode_speed` treated an OOM
at one batch as a crash, so the first arm to run out killed the measurement for every arm
at every batch; a batch that does not fit is now a hole in the table, rendered `-`.

**`Engine.set_chunk_prefill`** returns the number of layers it touched, so an A/B that
matched nothing cannot publish a 0% effect and read as a refutation of the kernel rather
than of the measurement.

## 4. Claims this session corrected

**"The B≤8 deficit is launch overhead."** Wrong, and it was mine. `scripts/step_profile.py`
measures the B=1 step at **91% device-busy** — 8.869 ms of kernel time inside a 9.748 ms
step — so fusing launches is worth at most 9%. The GEMM bucket is 5.646 ms against a
weights-only floor of 5.24 ms, i.e. **the GEMMs already run at ~93% of the memory wall**.
The cost is the 3.2 ms of non-GEMM work around them: 2.195 ms of elementwise over **1,873
launches**, whose largest single item is 0.733 ms of plain `direct_copy` over 452
launches, plus ~0.74 ms of dynamic activation quantization. That is work that should not
exist, not kernels that need to be faster. Corrected in the README in place.

**A test that was passing for the wrong reason.**
`test_graphed_scheduler_matches_the_eager_one` asserted exact bf16 greedy token identity
between schedulers that run the partial batch at B=3 and B=4. Measured: the
eager-vs-graphed logit residual is 0.06–0.19 while the top-2 gap goes as low as 0.0625, so
at several steps the token is decided by GEMM-shape rounding. It flipped when the chunk
kernel landed — a *more* accurate prefill. Replaced with a direct bitwise assertion of the
property its docstring names (a padded row must not advance a live slot) plus an agreement
tripwire. This is the second instance of this exact defect; the first was
`test_decode_matches_prefill`. **Any bf16 gate asserting greedy identity across two paths
with different GEMM batch shapes is measuring tie-breaks.** Look for more.

## 5. Open problems

**The locked MVP target is met — the noise argument matters and is recorded.**
At B=16 the margin over +25% is 1.5pp, so the medians were not trusted alone:
every same-rep pairing of braid against llama.cpp clears the clause (+25.9 to
+26.6%), and the most adversarial pairing in the sample is +25.8%. B=64 clears
by 18.8pp. If a future change shaves B=16, this is the first gate to re-run —
the margin is real but it is not wide.

**4-bit weights are measured and shelved, not unexplored.** `_weight_int4pack_mm` exists
and works on this build: 3.13× over FP8 at M=1, 1.66× at M=8, but **0.22× at M=32 and
0.12× at M=64**. It is a GEMV kernel. It wins exactly where braid loses and loses exactly
where braid wins, so it is not a lever for the thesis. And adopting it at low batch would
require moving llama.cpp to Q4_K_M in the same table, or the comparison is a precision
choice wearing a speedup's clothes.

**Serving now reaches c=128** (3,633 tok/s, prefill included) — the ceiling
item from the last checkpoint is closed. TTFT at c=128 is 3.3 s p50, which is
the number to watch if the serving story goes to OpenRouter: it is prefill
queueing, and prefix caching (unbuilt, deliberately) is the standard answer.

## 6. Next steps, in the order I would take them

1. ~~Amortise activation quantization~~ **done** — `quant_act.cu`, 2 launches for 9,
   −1.35 ms/step at B=1, bit-identical (60 gates).
2. ~~In-kernel gates~~ **done** — `gdn_decode_raw`/`gdn_prefill_raw`, −168 launches/step,
   bit-identical incl the seq_lens pad identity (16 gates).
3. ~~Raise the served ceiling to c=128~~ **done** — `DEFAULT_BUCKETS` to 128, plus the
   Scheduler now appends an off-rung capacity as its own top rung (a capacity of 100
   used to be a mid-serve ValueError the first time all 100 slots decoded in one tick).
4. ~~Audit for bf16 tie-break gates~~ **done** — exactly one more found
   (`test_graph_decode` kv_len arm: 2e-2 tolerance one line, exact argmax the next);
   fixed via the now-shared `conftest.assert_greedy`. `test_chunked_prefill` and
   `test_ragged_prefill` looked at risk but are sound (fp32 + rel<1e-5 guard first).
5. **Kill the remaining copies.** Still the largest single non-GEMM item: `direct_copy`
   launches from `.float()`/`.contiguous()` on projection outputs and norms. The gate
   tensors' share is now gone; what remains is the conv input cast and the q/k/v
   contiguity copies at B>1 — the conv kernel could take bf16 in and widen internally.
6. **FP8 KV**, the other half of Phase 5 item 7. Unrun, not refuted.
7. ~~Re-run the published sweep + serve table on the new code~~ **done** (`95612cc`).
8. ~~Server-vs-server head-to-head~~ **done** (`ee91e0e`, `a2849d2`).
9. ~~Longer prompts~~ **done** — npp=512; npp=1024 remains unrun and is the
   obvious next sweep if anyone asks.
10. ~~Kill the remaining copies~~ **done** (`34971f7`).
11. **FP8 KV** — still the next unrun lever. **Prefix caching** — the answer
   to TTFT at real prompt lengths if serving goes anywhere multi-turn. **The
   35B MoE** — the roadmap's original goal; runs through the MoE quantizer.

## 7. Working notes

- The box bills ~$0.79/hr. `make gpu-stop` as soon as a batch of work is done.
- `scripts/remote.sh` does `rsync -az --delete` before **every** command. **Never run a
  remote command while a benchmark is in flight** — it swaps the code under the running
  measurement. Write outputs to `/root/`, outside the synced tree.
- Long benches: launch in the background and poll a row count, not the box.
- `BRAID_KERNEL_VERBOSE=1` adds `-Xptxas=-v` for registers and spill counts. Changing the
  flags invalidates ninja's cache and forces a rebuild, so it is off by default.
- Scripts under `scripts/` are run by path and need `PYTHONPATH=/root/braid`; modules
  under `braid/` are run with `python3 -m` and do not.
- The 9B suite needs **one module per process**; two module-scoped engines no longer share
  a 31 GB card.
