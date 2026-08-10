# braid Agent Skills — Index

Project-scoped skills for agentic work on braid (`.claude/skills/*/SKILL.md`). Each
description states when to fire **and** when not to — keep that property when editing.

| Skill | Covers | Pairs with |
|---|---|---|
| [remote-gpu-workflow](remote-gpu-workflow/SKILL.md) | Getting work onto the box: `make test-remote`, rsync semantics, provisioning, JIT builds, instance lifecycle and cost | benchmark-remote-5090 |
| [sm120-gdn-kernels](sm120-gdn-kernels/SKILL.md) | Writing/reviewing CUDA for sm_120a — compile flags, SMEM budget, capture safety, silent-wrong-output traps; dead-ends ledger in `references/` | benchmark-remote-5090 |
| [benchmark-remote-5090](benchmark-remote-5090/SKILL.md) | Timing, profiling and A/B methodology; noise floor, host health, roofline, ncu/nsys, publishing numbers | sm120-gdn-kernels |

Boundaries (to avoid trigger collisions):

- *Run it on the GPU* → remote-gpu-workflow · *write the kernel* → sm120-gdn-kernels ·
  *is this number real* → benchmark-remote-5090.
- A build failure is sm120-gdn-kernels (flags, arch, instruction) unless it is a missing
  dependency or a stale remote, which is remote-gpu-workflow.
- A slow kernel starts at benchmark-remote-5090 — measure before optimizing. Only once the
  bottleneck is identified does sm120-gdn-kernels apply.

## Provenance

The sm_120a hardware ledger and the measurement contract were absorbed from the reference
engine's own `.claude/skills/` (`sm120-cuda-expert`, `benchmark-cuda`) — same silicon, same
class of box. Rows carried over unverified are marked **[ref]** in
`sm120-gdn-kernels/references/known-issues.md`; everything unmarked is braid's own measurement.

Two places where braid deliberately **diverges** from that source, both load-bearing:

- **No `--use_fast_math`.** It breaks fp32 oracle parity at the tolerances braid's tests
  assert. The reference engine uses it in release builds.
- **FP16 recurrent state is treated as open, not refuted.** See
  `docs/ARCHITECTURE.md` §6 — their FP16 `h_state` failure was a buffer overrun, and they run
  FP16 `h_state` on Mamba2 today.

Not ported, and why: their PR/release mechanics, GGUF and NVFP4 `StorageTier` dispatch,
CUTLASS cache wiring, symbol-graph queries, and the FP4/FP8 MMA PTX templates — all belong to
a C++ engine with paths braid excludes as non-goals. Their output-degeneration check and
model-onboarding skills are worth revisiting once braid has a loader and a sampler producing
tokens; there is nothing to degenerate yet.
