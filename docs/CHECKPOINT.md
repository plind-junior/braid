# Checkpoint — 2026-08-08

Where braid is, what changed, and what to pick up next. Written to be read cold.

**HEAD:** `f2876da` · **branch:** `main` · **tests:** 132 green on the remote 5090
**GPU:** vast.ai instance `47055458`, **stopped** (`status=exited`). 10.4 h rented.

---

## 1. State of the engine

Phase 2 complete, Phase 3 items 1–2 complete. Item 3: profiling done, **chunked
prefill done (it was silently wrong, not missing)**, `kv_len` bucketing done, the
paged block manager deliberately deferred with arithmetic, ragged batched prefill
still refused.

| | |
|---|---|
| loads | `Engine.from_pretrained(path)` — 32 layers, 24 GDN + 8 gated attention |
| runs | `forward` (prefill), `decode_step` (sync-free, graph-capturable), `hidden_states` |
| generates | `generate`, `generate_batch` — slot-pooled, per-row sampling |
| accelerates | CUDA GDN/conv kernels (`use_kernels=True`), CUDA-graph buckets, FP8 MLP (`quant_mlp=True`) |
| does **not** have | scheduler, continuous batching, SSE server (Phase 4); KV block manager, chunked prefill (Phase 3 item 3) |

**Correctness, all measured:** perplexity 8.2361 vs HF 8.2393 (0.021%); fp32
end-to-end 6.4e-7; bf16 greedy token identity with HF; fp32 token identity at
B=2/4/8; graph replay bit-identical to eager at `rtol=0, atol=0`.

## 2. Performance, and the honest comparison

B=16 decode, median of 3 processes, graphs on, CUDA kernels:

| | ms/step | tok/s | vs llama.cpp |
|---|---:|---:|---:|
| session start (2026-08-07) | 16.727 | 956.6 | 0.51× |
| + grouped decode attention | 12.065 | 1,326.2 | 0.71× |
| + fused RMSNorm | 11.363 | 1,408.0 | 0.75× |
| + FP8 MLP (opt-in) | 10.266 | 1,558.5 | 0.83× |
| + `kv_len` bucketing | **10.044** | **1,592.9** | **0.85×** |

**+66.5% this session.** c=1 is 131.2 tok/s; the Phase 3 re-plan trigger fired at
113.5 and is now cleared, at 8.8% over its 120 threshold.

**braid is still slower than llama.cpp** (1,879.68 at B=16 on this box). The MVP
target is to *beat* it by ≥25% at B=16 and ≥100% at B=64. See §5 — the plan as
written does not obviously get there, and one part of it is self-contradictory.

## 3. What changed this session

Four commits.

- `d9ef187` — docs: THESIS split out of ARCHITECTURE, section references fixed.
- `d7ee96d` — **grouped T=1 GQA decode attention**, +38.6% at B=16. SDPA fell to
  the math backend, which replicates K/V 4× and runs the whole thing in fp32.
- `561593a` — **RMSNorm as one bit-exact kernel** (`F.rms_norm` over fp32 input),
  +6% everywhere. Plus three published-claim corrections (§4).
- `479c034` — **FP8 W8A8 on the MLP**, opt-in, +10.7% at B=16 for +0.50% PPL.
- `f2876da` — **chunked prefill was silently wrong** (1.6e-1 in fp32, greedy token
  still agreeing; the GDN conv ignored the cached window for T>1). Fixed to
  5.9e-7. Plus `kv_len` bucketing, +2.3% at B=16.

New tooling worth knowing about:

| file | what it answers |
|---|---|
| `braid/bench/decode_profile.py` | where the step goes (`--mode attribute` graphed, `--mode locate` eager for op identity, `--mode spin` for a profiler) |
| `braid/bench/gemm_paths.py` | GEMM vs a same-tensor read floor; which reduced-byte paths run |
| `scripts/fp8_scalemode_probe.py` | which `_scaled_mm` scale modes are fast, and which exist |
| `scripts/sdpa_backend_diag.py` | which SDPA backend runs per shape |
| `scripts/rmsnorm_probe.py` | `F.rms_norm` speed and bit-exactness |

## 4. Claims that were wrong and are now corrected

Kept visible because they change what to do next.

1. **"head_dim=256 disqualifies every fused SDPA backend."** Wrong — inferred
   from one error message. Flash accepts head_dim 256 for every shape braid
   issues and the dispatcher picks `flash_fwd_splitkv_kernel` for T=1. What
   forces the math backend is the **explicit additive mask** `decode_step` must
   pass because `kv_len` is pinned to `max_len`. *Consequence: flash-decoding is
   reachable, and the block manager is what unlocks it.*
2. **"GEMMs are at 68% of roofline, so 68→90% is available."** Wrong — that
   divided weight bytes by a *copy* benchmark; a weight GEMM only reads. Against
   a same-tensor read the GEMMs are at **86% aggregate, 99–105% on the dominant
   shapes**. *Consequence: there is no kernel-choice win. `cublaslt` is identical
   and padding M is worse, both tested.*
3. **"braid's 0.51× against llama.cpp is exactly the BF16:Q8_0 weight ratio."**
   Wrong — it is 0.83× now with the weight bytes barely changed. A coincidence
   was read as a mechanism.
4. **"`torch._scaled_mm` FP8 is unsupported on sm_120."** Stale — it runs, and it
   is now shipping.

Also: **`ncu` cannot run on this box.** `ERR_NVGPUCTRPERM` needs a host-level
driver flag and reboot that vast.ai does not give us. The roadmap named "ncu with
graphs on" as item 3's first task; torch.profiler over graph replays replaced it,
reconciled to 96% of the wall clock.

## 5. Two open problems in the plan itself

Neither is a code bug. Both need a decision.

**Quantization is not on the critical path but the target depends on it.** It is
Phase 5+ **item 9**, in a section explicitly "not started before Phase 4 clears",
and scoped there as a MoE-expert quantizer at ~1,200 lines / 4–6 weeks for the
35B. What the 4B needed was ~150 lines and is now done. The remaining candidates
(`lm_head`, attention projections) are similarly small. Consider promoting them.

**The B=64 target contradicts Phase 1.** The locked MVP target is "≥100% at
B=64"; Phase 1 item 4 decided "batch buckets stop at 16 — c=32 does not fit in
VRAM and is throughput-pointless". As scoped, braid cannot attempt half its own
target. One of the two has to move.

## 6. Next steps, in the order I would take them

1. **Phase 4 — scheduler, slot lifecycle, SSE server.** Item 3's remaining pieces
   (paged KV, ragged batched prefill) are both blocked on decisions Phase 4 makes,
   so building them first would be building against a guess. Note the item 3
   runbook's arithmetic: paging is affordable to ~786,000 token-slots and the MVP
   uses 8,192.
2. **`lm_head` FP8** (~0.38 ms, 15% of weights) with its own perplexity gate —
   its error lands directly on the greedy argmax, so gate on token identity.
3. **Attention/GDN-out projections FP8** (~0.5 ms) — feed state that compounds
   across steps, so gate on perplexity *and* a long-generation drift check.
4. **Flash-decoding.** Flash accepts head_dim 256 here; the blocker is the
   additive mask, which needs per-row KV lengths (`_flash_attention_forward` with
   `seqused_k`, or FlashInfer). `kv_len` bucketing alone does **not** unlock it —
   rows still differ in length inside a bucket.
5. **Re-run the llama.cpp head-to-head in one session.** The baseline and
   braid's numbers are from different days; a publishable comparison needs both
   arms with concurrent health sampling.
6. Ragged batched prefill (B>1, T>1), once Phase 4 fixes the admission shape.

## 7. Working notes

- **Always** `make gpu-start` … `make gpu-stop` in the same turn. `vastai stop`,
  never destroy — stopping preserves the 300 GB disk with the staged models.
- One process per measurement, 3 processes, median, print the spread. Noise floor
  is **1.65%**. Time CUDA events around the whole run, never per step.
- The host-health classifier flags `DEPRESSED` on power < 400 W. For FP8 that
  fires at full clocks and is **not** a throttled box — it is the workload moving
  fewer bytes. Clocks are the load-bearing part of that verdict.
- Watch for L2: a single decode weight is 5–47 MiB against 96 MB of L2, so any
  microbenchmark over one tensor measures cache. Rotate a bank past 512 MB.
- `docs/runbooks/` is gitignored — the runbooks are local-only. Anything that
  needs to survive goes in the committed docs.

**Uncommitted and not mine:** `docs/ARCHITECTURE.md` and `docs/THESIS.md` have
in-progress prose edits. `README.md`'s status header is also stale — it still
says "the engine does not exist yet" and "38 tests".
