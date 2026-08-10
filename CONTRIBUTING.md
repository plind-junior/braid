# Contributing to braid

braid is a measurement-driven project: every optimisation in the tree earned its place
with a number, and every claim in the README is reproducible. Contributions are welcome —
kernels, serving features, benchmarks, docs — and the same standard applies to all of
them. This document tells you what you need, what the automation will do to your PR, and
what gets a change merged.

## What you need

**A real RTX 5090 (`sm_120`).** There is no GPU in CI and nearly every test file needs
one. A rented box works fine — vast.ai rents 5090s for under a dollar an hour, and the
whole workflow assumes a remote box:

```bash
export BRAID_SSH_KEY=~/.ssh/yourkey BRAID_SSH_HOST=root@host BRAID_SSH_PORT=22

make provision      # installs torch, pytest, ninja, ccache on the box
make test-remote    # runs the suite there (rsyncs your tree first)
make lint           # ruff — local, free, and the CI gate
```

`scripts/remote.sh` rsyncs your local tree to the box with `--delete` before every
command, so your local tree is the only source of truth — anything written on the box
inside `/root/braid` is destroyed on the next invocation. Write outputs to `/root/`,
outside the synced directory.

No local GPU is needed for docs, scripts, or the GPU-free tests
(`tests/test_gdn_ref.py`, `tests/test_loader.py`, `tests/test_pr_eval_bot.py`).

## What happens to your PR

Three layers of automation, modeled on measurement discipline rather than process for
its own sake:

1. **An automatic first-pass review** (Claude) on every non-draft PR. It reviews what a
   careful reader can check from the diff: measurement discipline, parity-test coverage,
   capture safety, ordinary bugs. It never approves or merges — the verdict is the
   maintainer's.
2. **The attestation gate.** A PR touching `braid/` or `tests/` must tick the
   **Tested on RTX 5090** box in the PR template. CI has no GPU, so the ticked box is
   the only signal the suite ran at all. Tick it only after a real run — the box is an
   attestation, not a formality. Draft PRs are exempt until marked ready.
3. **The eval bot — your speedup claim gets measured.** For each eligible PR head, a bot
   starts a real 5090, runs the suite on your *merged* tree, then benches it against
   `main` **in the same box session** (median of 3 reps, arm order alternated, the
   serving-shaped decode configuration at B=16 and B=64) and posts the measured table as
   a comment with one of these labels:

   | label | meaning |
   |---|---|
   | `eval:pass` | measured speedup beyond the ±2% same-session noise bar |
   | `eval:noise` | measured delta within ±2% — not distinguishable from nothing |
   | `eval:reject` | suite failed, or a measured regression beyond 2% at any batch |
   | `eval:error` | evaluation infrastructure failed; retried once automatically |

   Each head commit is evaluated once; pushing a new commit re-queues it. A regression
   anywhere outranks an improvement anywhere else. The bot never merges.

Merging is manual, by the maintainer, after all of the above.

## The rules that get a change merged

**A kernel change lands with a parity test.** Every CUDA kernel in the tree is asserted
against the slow, obviously-correct implementation in `braid/reference/` — most of them
to bit-identity, not tolerance. A speed number without a parity test is not reviewable.

**No `--use_fast_math`.** It breaks fp32 oracle parity at the tolerances the tests
assert. This is deliberate and non-negotiable.

**Decode-path code stays capture-safe.** The decode step runs inside captured CUDA
graphs: no allocation, no host synchronisation, no CPU-dependent control flow inside
anything the graph captures.

**Numbers follow the measurement rules** (see the README section): labelled measured or
projected, medians across processes with spreads printed, same-session A/B only, losing
rows published unchanged. One run is an anecdote. If your PR claims a speedup, fill the
before/after table in the PR template with real numbers from `make bench-scaling` — the
eval bot will check them against its own.

**Code conventions.** Python 3.10+, 4-space indent, ~100 columns; `make lint` (ruff) is
the gate and `make fmt` applies safe autofixes. CUDA lives in `braid/kernels/csrc/`,
JIT-compiled through `torch.utils.cpp_extension` — there is no CMake. Bench code
deliberately deletes large tensors between arms and times through closures; `braid/bench/`
and `scripts/` carry per-file ruff ignores for this, keep that pattern.

**Commits.** Conventional subjects (`feat:`, `fix:`, `perf:`, `docs:`, `ci:`), body
explaining *why*, carrying the measured delta when there is one.

**Scope.** One lever per PR. A PR that changes a kernel, refactors the scheduler, and
edits the bench harness cannot be evaluated — the A/B stops meaning anything.

## Where things are

```
braid/model/      the engine — layers, cache, graph capture, quantisation
braid/kernels/    CUDA sources + the JIT loader
braid/serve/      scheduling and serving
braid/bench/      benchmarks that produce publishable numbers
braid/reference/  slow, obviously-correct implementations the tests trust
scripts/          diagnostics, remote/provisioning shell, the PR eval bot
tests/            the gate
```

Good first contributions: the deferred items (prefix caching, paged KV, FP8 KV cache),
longer-context sweeps, or anything on the profiler's launch-count list — ask in an issue
first if you want the current state of any of them.
