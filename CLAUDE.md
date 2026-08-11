# braid — working agreement

A single-GPU inference engine for hybrid language models (attention interleaved with Gated
DeltaNet), built to serve them at concurrency on one RTX 5090.

Read [`docs/CHECKPOINT.md`](docs/CHECKPOINT.md) first — it is written to be read cold and says
where the engine actually is. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is what braid is,
[`docs/THESIS.md`](docs/THESIS.md) is why, [`docs/ROADMAP.md`](docs/ROADMAP.md) is the build
order and the gates.

---

## Nothing runs locally

There is no GPU and no CUDA toolkit on the dev machine. Every test, benchmark and kernel build
runs on a rented vast.ai RTX 5090. A local `pytest` that dies at import means "you forgot
`-remote`", not "the code is broken".

| | |
|---|---|
| Run the suite | `make test-remote` |
| Run part of it | `make test-remote ARGS="tests/test_gdn_decode_kernel.py -k slot"` |
| Start / stop / price the box | `make gpu-start` · `make gpu-stop` · `make gpu-status` |
| First boot after a rebuild | `make provision` |
| Lint / format | `make lint` · `make fmt` (these *are* local) |

`scripts/remote.sh` does `rsync -az --delete` before every command, so **the local tree is the
only source of truth** and anything written on the remote inside `/root/braid` is destroyed on
the next invocation. Write outputs to `/root/`, outside the synced directory.

**The box bills ~$0.79/hr while running.** Stop it as soon as a batch of GPU work is done.
Always `stop`, never `destroy` — the 300 GB disk holds staged models and a built llama.cpp
tree that cost a re-download and a ~10 min rebuild to replace.

## The skills carry the detail

Three project skills in [`.claude/skills/`](.claude/skills/README.md), each stating when it
fires and when it does not. Consult them rather than re-deriving:

- **remote-gpu-workflow** — getting work onto the box: rsync semantics, provisioning, JIT
  builds, instance lifecycle, cost.
- **sm120-gdn-kernels** — writing CUDA for sm_120a: compile flags, SMEM budget, capture
  safety, silent-wrong-output traps, and a dead-ends ledger.
- **benchmark-remote-5090** — timing and profiling method: noise floor, host health, roofline,
  ncu/nsys, and what has to be true before a number is published.

Boundary: *run it on the GPU* → remote-gpu-workflow · *write the kernel* → sm120-gdn-kernels ·
*is this number real* → benchmark-remote-5090.

## Measurement discipline

This is the part of the project most easily damaged by carelessness.

- Every number that reaches a doc or the README is labelled **measured** or **projected**.
  Never let a projection acquire the voice of a result.
- A speedup is not real until it clears the noise floor (`make bench-noise`) and survives a
  median-of-N across separate processes. One run is an anecdote.
- The honest comparison is against llama.cpp on the same box at its own default settings.
  Publish losing rows unchanged — the c=1 row is expected to lose, and hiding it would make
  the whole curve untrustworthy.
- When a published claim turns out wrong, correct it in place and say so in the checkpoint.
  There is a precedent for this and it should stay cheap to do.

## Code conventions

- Python 3.10+, 4-space indent, ~100 columns. `make lint` (ruff) is the gate; `make fmt`
  formats. Both run locally and cost nothing.
- CUDA lives in `braid/kernels/csrc/`, JIT-compiled through `torch.utils.cpp_extension` — there
  is no CMake and no C++ build tree.
- **No `--use_fast_math`.** It breaks fp32 oracle parity at the tolerances the tests assert.
  (The reference engine uses it; braid deliberately does not.)
- Bench and diagnostic code deletes large tensors between arms to free VRAM, and times through
  closures. Ruff reads both as errors, so `braid/bench/` and `scripts/` carry per-file ignores
  in `pyproject.toml`. Keep that pattern rather than working around it.
- A kernel change lands with a parity test against the reference implementation in
  `braid/reference/`, not just a speed number.

## Layout

```
braid/model/      the engine — layers, cache, graph capture, quantisation
braid/kernels/    CUDA sources + the JIT loader
braid/serve/      scheduling and serving (Phase 4, in progress)
braid/bench/      benchmarks that produce publishable numbers
braid/reference/  slow, obviously-correct implementations the tests trust
scripts/          one-off diagnostics and remote/provisioning shell
tests/            the gate; all but two files need a real GPU
docs/             ARCHITECTURE · THESIS · ROADMAP · CHECKPOINT
```

`docs/runbooks/` and `docs/superpowers/` exist on disk but are gitignored on purpose — internal
working documents, not published.

## Commits

Conventional-commit subjects (`feat:`, `fix:`, `perf:`, `docs:`), body explaining *why* and
carrying the measured delta where there is one. Commits are attributed to the repo owner
alone — **do not add a Co-Authored-By or any other AI-attribution trailer.**
