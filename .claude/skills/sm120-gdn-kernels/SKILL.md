---
name: sm120-gdn-kernels
description: Use when writing, reviewing, or debugging braid's CUDA kernels for RTX 5090 / GB202 / sm_120a — the GDN decode scan, slotted conv1d, attention or GEMV work in braid/kernels/csrc/. Triggers on CUDA or PTX source, nvcc flags, TORCH_CUDA_ARCH_LIST, shared-memory sizing, __launch_bounds__, occupancy, register pressure, CUDA-graph capture failures, garbage or NaN kernel output, "why is this kernel slow", cp.async, mma.sync, wmma. Do NOT use for timing or profiling a kernel (benchmark-remote-5090) or for getting work onto the GPU box (remote-gpu-workflow).
---

# sm_120a Kernels — braid

RTX 5090 (GB202, **`sm_120a`**, consumer Blackwell). Dead ends, load-bearing fixes and
things that silently produce wrong output → `references/known-issues.md`. **Read it before
proposing a lever** — most obvious ideas on this chip have been measured and refuted.

`docs/ARCHITECTURE.md` §7 (the kernels and their sm_120a constraints) and `docs/THESIS.md`
§6.1 (dead ground) are the in-repo source of truth. This skill is the version that fires
while you are editing a `.cu` file.

## Architecture quick reference

| Spec | Value |
|------|-------|
| SMs · CUDA cores · TC | 170 · 21,760 · 680 (4/SM, 5th gen) |
| L1/SMEM per SM | 128 KB configurable, **~99 KB opt-in** — query `sharedMemPerBlockOptin` |
| L2 | 96 MB unified |
| VRAM | 32,607 MiB GDDR7 — 1,792 GB/s datasheet, **1,508–1,528 GB/s measured** |
| Native MMA | FP16 `m16n8k16`, FP8 `m16n8k32`, FP4 block-scaled `m16n8k64` |
| Toolchain | CUDA 12.8.61, torch 2.11.0+cu128, driver 580.126.09 |

**Use the measured bandwidth for every %-of-wall claim**, never 1,792. All published braid
numbers do.

**sm_120 has NO `tcgen05` / TMEM / `wgmma` / TMA / 2-CTA cluster MMA.** Those are SM100
(B200). `cp.async.bulk` and `st.async .b128`-to-global are unavailable — re-probed under
CUDA 13.3 / PTX ISA 9.3 with **0 of 247 instructions flipped**. The ISA surface is
silicon-fixed; do not re-probe on toolkit bumps, and do not port a Hopper pipeline.

`nvcuda::wmma` *compiles* here but lowers to **HMMA** with a shared-memory round-trip — it is
not the peak path and not a tensor-core claim. Hand-write `mma.sync` with register-resident
fragments or don't claim tensor cores.

## Compile target — braid's flags are not the reference engine's

braid JIT-builds through `torch.utils.cpp_extension.load` in [loader.py](../../../braid/kernels/loader.py):

```python
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")   # NOT "12.0"
extra_cuda_cflags=["-O3", "-lineinfo"]                    # NOT --use_fast_math
```

- **`12.0a`, never `12.0` / `120f`.** The `a` suffix is a superset adding
  `mma.sync.kind::mxf4nvf4.block_scale` and extended `cp.async.bulk.tensor`. torch 2.11's own
  arch list reports only `sm_120`; that constrains torch's prebuilt kernels, not what nvcc
  emits for ours.
- **No `--use_fast_math`.** It turns `rsqrtf` into an approximation and breaks fp32 parity
  against the oracle at the tolerances the tests assert (`rtol=2e-5, atol=2e-6`). The
  reference engine uses it in release builds; braid's parity contract forbids it. Do not
  "restore" it.
- Keep `-lineinfo` — `ncu --set detailed --import-source yes` needs it for source-correlated
  stalls.
- Guard device code with `#if __CUDA_ARCH__ >= 1200`.

## Kernel rules on this chip

1. **The scan is already at the memory wall.** Measured 104% of HBM at B=8. It cannot be made
   faster, only **smaller** — fewer state bytes moved. A kernel-microbench win that does not
   reduce bytes is worth ~0 end-to-end: the reference engine measured +16.7% on its scan kernel → −0.18%
   end-to-end. Any milestone that reports scan kernel time without an end-to-end number is
   reporting nothing.
2. **Occupancy gets you *to* the roofline, not past it.** Keep registers ≤48/thread for 100%
   occupancy (`--ptxas-options=-v` to check). On a path already at its measured ceiling,
   occupancy work is refuted — check the number before spending time there.
3. **Everything in the decode step must be capture-safe.** `cudaMallocAsync` inside a captured
   graph crashes; any D2H inside capture is an IMA. Pre-allocate every workspace and keep all
   arguments device-side — this is exactly why `slot_idx` is a device tensor, not a host int.

## Shared memory

Max opt-in **~99 KB per block** (NOT H100's 228 KB). Design tiles to ≤97 KB and query the
device rather than hardcoding:

```cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
```

Reference tile sizings that fit: head_dim 64 → Bq 128 (~89 KB) · 128 → Bq 64 (~81 KB) ·
256 → Bq 32 (~88 KB, the Qwen3.5 GDN partial-RoPE shape).

## Common mistakes

| Mistake | Fix |
|---------|-----|
| `__launch_bounds__(HD, 2)` at HD=128 | **ptxas MISCOMPILE** — correct math, garbage output. Our scan kernel must use `(HD, 1)`. |
| `__launch_bounds__` on a regular kernel "to help" | Costs −4.5% to −20% on paths that don't need it. Only the documented exceptions. |
| Adding `--use_fast_math` | Breaks fp32 oracle parity. See above. |
| Clamped L2 norm `rsqrt(max(sum_sq, 1e-12))` | HF uses **additive** `rsqrt(sum_sq + 1e-6)`. Identical on a healthy head, **10× apart** on a near-zero one. Pinned by a test — don't "optimize" it back. |
| Assuming H100's 228 KB SMEM | ~99 KB opt-in here. Query `sharedMemPerBlockOptin`. |
| `__noinline__` on a device inner-loop helper | Spills to local memory (DRAM). Use `__forceinline__`. |
| Missing `__syncthreads()` after `cp.async.wait_group` | Race on the SMEM read. |
| Pointer advance without `sizeof(T)` | Cost the reference engine a long-context cliff at prompt > 1024. |
| Registers > 48/thread | `--ptxas-options=-v`, then refactor. |
| Materializing the attention S/P tile in SMEM | Becomes barrier/L1TEX-bound with tensor cores idle. Keep row max/sum and S/P fragments register-resident. |
| L2 access-policy window past `cudaDevAttrMaxAccessPolicyWindowSize` | Silent CUDA error / IMA on the 5090 (128 MiB max). Clamp it. |
| Copying a block-reduction from a reference engine | The reference engine's L2-norm block reduction has a **data race** — every thread reads `s_reduce[0]` then writes `s_reduce[d]` with no barrier between (its `gdn.cu:129-131` + 3 sibling sites). Ours must not. |
| Claiming a source tweak is "perf-neutral" without a SASS diff | `cuobjdump -sass` — byte-identical SASS is proof; a bench is not. |

## PTX worth having

Only the parts that apply to a bf16/fp32 GDN engine. The FP4/FP8 block-scaled MMA templates
in the reference engine's `ptx-patterns.md` are for a path braid explicitly excludes (`docs/THESIS.md`
§6.1: FP4 inside attention math refuted 4×, NVFP4 on GDN projections −9…−20%).

```cuda
// 16 B async global→shared, with the barrier that is easy to forget
asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(glob));
asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group 0;\n");
__syncthreads();                       // REQUIRED before reading smem

// warp-XOR reduce (sum/max) in 5 instructions
#pragma unroll
for (int o = 16; o >= 1; o >>= 1) val += __shfl_xor_sync(0xFFFFFFFF, val, o);

// streaming load, bypasses L1 — for one-shot reads (KV, state) only, never weights
const float4 v = __ldcs(reinterpret_cast<const float4*>(p));
```

## Where to look next

- `references/known-issues.md` — dead ends, load-bearing fixes, negative results. Read before
  proposing a lever.
- `docs/ARCHITECTURE.md` §6 (numerics contract, state precision) and §7 (kernels,
  sm_120a constraints); `docs/THESIS.md` §6.1 (dead ground).
- Kernel source: `braid/kernels/csrc/`. Oracle: `braid/reference/gdn_ref.py`.
