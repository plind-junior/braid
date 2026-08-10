# sm_120a Known Issues, Dead Ends and Load-Bearing Fixes — braid

Heavy reference for the `sm120-gdn-kernels` skill. Two sources: braid's own measurements, and
the reference engine's own known-issues ledger (its `sm120-cuda-expert` skill) filtered to
what a bf16/fp32 GDN engine can actually hit. Rows sourced from the reference engine are
marked **[ref]** — they are someone else's measurement on the same silicon, credible but not
re-verified here.

**Pre-flight before non-trivial kernel work:** scan the dead ends below. Many obvious
optimizations are proven failures on this chip. Skip pre-flight for parameter tweaks,
signature changes, or fusing two existing kernels.

---

## Hard unavailability (silicon, not toolkit)

A 247-instruction PTX survey at `compute_120a` across CUDA 13.2 → 13.3 (PTX ISA 9.3) flipped
**0 instructions** — none unlocked, none regressed. **[ref]** Do not re-probe on toolkit
bumps; a CUDA release buys tooling and cuBLAS perf, not ISA surface.

| Unavailable | Notes |
|---|---|
| `tcgen05.*`, TMEM, `wgmma`, 2-CTA cluster MMA | SM100 (B200) exclusives |
| TMA — `cp.async.bulk`, `cp.async.bulk.tensor` with `.ignore_oob` | Use `cp.async.ca/cg.shared.global` at 16 B |
| `st.async .b128` to global | PTX 9.2+ only targets `shared::cluster` |
| cuBLASLt grouped GEMM | Returns **zero algorithms** on sm_120. Retry only on a new cuBLAS release. |
| FP8 via `torch._scaled_mm` | `CUBLAS_STATUS_NOT_SUPPORTED` on sm_120 — measured in braid's own `fp8_probe.py`. The reference engine reports the same at non-aligned M and ships FP8 prefill disabled. |

`nvcuda::wmma` compiles but lowers to **HMMA** with a shared-memory round-trip — not async
`wgmma`, not the peak path, and it costs extra SMEM traffic versus hand-written `mma.sync`
with register-resident fragments. **[ref]**

**Correction to an earlier braid claim:** 2:4 sparse FP4 *is* accepted by ptxas on `sm_120a`
(`kind::f8f6f4.sp::ordered_metadata.m16n8k64…`, and `sparse mxf4nvf4 4X K=128 ue4m3`). The
original spec conflated block-scaled `mxf4nvf4` with plain `f8f6f4`. Out of scope for braid on
effort grounds, **not** availability grounds — do not repeat the availability claim publicly.

---

## Produces wrong output silently

These are the expensive class: no crash, no error, plausible-looking tokens.

| Trap | Symptom | Rule |
|---|---|---|
| `__launch_bounds__(HD, 2)` at HD=128 | ptxas **miscompile** — math correct, output garbage **[ref]** | Scan kernel uses `__launch_bounds__(HD, 1)`. Enforced in `gdn_decode.cu:21`. |
| `--use_fast_math` | `rsqrtf` becomes an approximation; fp32 oracle parity fails at `rtol=2e-5` | braid-specific: never add it. Diverges from the reference engine, which uses it in release. |
| Clamped L2 norm `rsqrtf(fmaxf(sum_sq, 1e-12f))` | Identical on a healthy head, **10× apart** on a near-zero one | HF — the implementation the checkpoint was trained with — uses additive `rsqrtf(sum_sq + 1e-6f)`. Pinned by a test; `gdn_decode.cu:86`. |
| Block reduction that reads `s_reduce[0]` then writes `s_reduce[d]` with no barrier between | Data race; intermittent wrong norm | The reference engine's `gdn.cu:129-131` and three sibling sites have this. Do not copy a reduction without checking its barriers. |
| Recurrent-slot allocator with an aliasing fallback | Two live sequences share one state slab | The reference engine's `engine_sampling_stop.cpp:256-262`. Ours must fail loud instead. |
| Pointer advance without `sizeof(T)` | Long-context cliff past 1024 tokens **[ref]** | |
| Pre-dequantizing a tensor without updating its dtype tag | Dispatcher misreads the bytes → state collapse **[ref]** | Relevant once the loader handles mixed storage. |
| Missing `__syncthreads()` after `cp.async.wait_group` | Race on the SMEM read | |
| `reinterpret_cast` on a 34-byte Q8_0 block | Not 4-aligned **[ref]** | Only if braid ever reads GGUF. |

**State precision, stated precisely** (this is braid's top open question,
`ARCHITECTURE.md` §6 and `THESIS.md` §7 — do not let a one-line skill row overwrite the
analysis):

- **FP8 E4M3 state is refuted.** 3-bit mantissa amplifies through the delta-rule scan →
  degenerate output after ~50 special tokens in multi-turn chat. **[ref]** Do not attempt.
- **FP16 state is NOT refuted.** The reference engine's FP16 `h_state` failure was a
  **buffer overrun** (an allocation that halved the stride so the next layer read into it),
  and they run FP16 `h_state` on Mamba2 today. Its own known-issues carries a separate "h_state must be FP32"
  row from a Qwen3.6 NaN — treat that as unexplained, not as a refutation. Worth 31.9
  MiB/sequence and it halves the traffic of the term that caps our curve.
- fp32 is mandatory for the MVP so parity is unambiguous.

---

## CUDA-graph capture

The decode step is graph-captured; anything that breaks capture removes the largest
multiplier on the path.

- `cudaMallocAsync` inside a captured graph **crashes**. Pre-allocate every workspace.
- Any **D2H inside capture is an IMA**. All kernel arguments stay device-side — this is why
  `slot_idx` is a device tensor.
- CUTLASS NVFP4 on sm_120 is **non-deterministic under `cudaGraphExecUpdate`** **[ref]**. If
  bitwise reproducibility is required, keep such GEMMs out of exec-update.
- Measured on braid: CUDA-graph replay across slot reassignment costs **10.3 µs/replay**.

---

## Refuted — do not re-attempt

Each is something a reasonable person would try. All were measured.

| Refuted | Measurement |
|---|---|
| Making the single-sequence scan faster | +16.7% kernel microbench → **−0.18% / −0.11% / −0.35% end-to-end** **[ref]**. The scan is at 104% of HBM at B=8 (braid, measured); it can only get *smaller*, not faster. |
| WY-representation / SSD / tensor-core chunkwise scan | Every TC variant **loses** to a plain chunk-cached scalar loop: 1.567 µs/tok sequential vs 1.343 chunk-cached **[ref]**. |
| NVFP4 on GDN `in_proj`/`out_proj` | **−9% (Nemotron) to −20% (Qwen3.6)** decode **[ref]**. Tuned FP16 GEMV hits 70–81% of HBM on wide GDN-output shapes. |
| FP4/NVFP4 inside attention math (QK^T or PV) | Format-intrinsic, refuted 4×: e4m3-QK PPL **5722** vs 6.12 **[ref]**. |
| One per-tensor FP8 scale over a fused heterogeneous pack | **+4% PPL** — one amax is dominated by the largest row group. Per-**row** scales are PPL-flat **[ref]**. |
| `torch._weight_int8pack_mm` as an INT8 weight-only path | **5–50× slower than bf16** on this card (braid, `gemm_probe.py`). |
| `__launch_bounds__` on regular paths | −4.5% to −20% **[ref]**. |
| Occupancy raises on a path already at its measured ceiling | Refuted repeatedly **[ref]**; on the FA2 hd=256 instance, split-D warp-pairing that halved registers and doubled warps/SM measured **slower everywhere, +10–16%**. |
| Materializing the attention S/P tile in SMEM | Barrier/L1TEX-bound, tensor cores idle **[ref]**. Keep row max/sum and S/P fragments register-resident. |
| Launch-elimination levers under a graphs-ON decode loop | Whole class **[ref]**: a fused kernel with bit-identical output and 2 fewer launches/layer moved e2e **0%**; capping decode split-K regressed −21…−35%. A decode lever must move real **bytes** or critical-path math. |
| C++23 `[[assume]]` as a perf hint | Byte-identical SASS = provably inert **[ref]**. General rule: `cuobjdump -sass` before any "should help" claim on a compiler hint. |
| `__noinline__` on device inner-loop helpers | Spills to local memory (DRAM) **[ref]**. |
| Generic `compute_120` PTX fallback | Lacks FP8 MMA + block-scale **[ref]**. Always pin `120a`. |
| Increasing SMEM past `sharedMemPerBlockOptin` assuming H100's 228 KB | ~99 KB here **[ref]**. |

---

## VRAM and the WDDM cliff

The reference engine's most expensive single bug, and the reason braid's allocator plan is
what it is:

At ~0 MiB free, WDDM/WSL2 **oversubscribes into host memory and keeps returning
`cudaSuccess`** — nothing fails, bandwidth just falls from ~1,530 GB/s to ~237 GB/s and decode
collapses ~7×. **A successful `cudaMalloc` is not evidence of room** (a 28 GiB allocation
succeeded with 22.6 GiB reported free). Bandwidth is the discriminator.

The shipped rule is an **ordering** rule: allocate the tier whose demand is *bounded by the
model* first, and let the elastic tier (KV / state pool) take the **measured** residual.
Sizing one large allocation from an *estimate* of another's future size is what broke it —
the estimate ran ~1.6 GiB low. Leave ≥1 GiB free.

braid's box is native Linux, not WSL2, so the WDDM oversubscription mechanism may not apply
identically — but the ordering rule costs nothing and the failure is silent, so keep it.

---

## Tooling caveats on this hardware

- `compute-sanitizer` does not work under WSL2 (no debugger interface). braid's box is native
  Linux — it should work; verify before relying on it.
- `ncu` serializes and replays kernels: **its wall-clock is not real time.** Use `cudaEvent`
  or a graph replay for timing, `ncu` for metrics only. See `benchmark-remote-5090`.
- CUDA graphs hide captured kernels from `nsys` unless you pass `--cuda-graph-trace=node`.
