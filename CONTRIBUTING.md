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
3. **The eval bot — your speedup claim gets measured, and the PR does not grade
   itself.** For each eligible PR head, a bot starts a real 5090, runs the suite on
   your *merged* tree, then benches it against `main` **in the same box session**
   (median of 3 reps, arm order alternated, the serving-shaped decode configuration at
   B=16 and B=64) and posts the measured table as a comment with one of these labels:

   | label | measured | what happens |
   |---|---|---|
   | `eval:landmark` | speedup beyond **+25%** | gate passes, but **never auto-merges** — see below |
   | `eval:major` | **+10% to +25%** | escalated to 7 reps; merges if both rounds agree |
   | `eval:pass` | above the bar, up to +10% | merges |
   | `eval:noise` | within ±bar | merges — measured harmless |
   | `eval:slower` | **−bar to −5%** | human decides: this may be a fair trade for correctness |
   | `eval:reject` | worse than −5%, or the suite failed | blocked |
   | `eval:tainted` | touched the harness, not cleared by cross-check | blocked, human reads the diff |
   | `eval:error` | infrastructure failed | retried automatically |

   **The bar is not a constant.** It is `max(2%, 3 × this session's observed spread)`.
   Three reps of the same tree normally land within 0.04% at B=16, so 2% is the floor
   in practice — but a session that scatters (thermal throttling, a noisy neighbour)
   raises its own bar and correctly downgrades a marginal claim. The bar only ever
   tightens; lowering the published 2% floor would need a fresh `make bench-noise` study.

   **Why the best result gets the most scrutiny.** Above +10% the bot runs four more
   reps and requires the two rounds to agree before merging. Above +25% it does not
   auto-merge at all. On an engine that has already been profiled hard, an
   extraordinary number is more likely to be work that quietly stopped happening than
   a breakthrough, and the suite catches that only where a test covers it. The comment
   also reports the *shape* of the gain — uniform, batch-skewed, latency-skewed —
   because "+16% at B=64, flat at B=16" is a different claim from a uniform +8%.

   The harness is **pinned**: `braid/bench/`, `braid/reference/` and `tests/` are taken
   from the base commit, not from your PR — your engine code is measured by main's
   bench and gated by main's tests against main's oracles. Modifying any of those paths
   (or the bot itself) doesn't skip the eval, but it caps your best verdict at
   `eval:tainted` until a human reads the harness diff. New files you *add* under those
   paths don't taint, but they don't run during the eval either — your new parity test
   joins the gate once merged. The arms are integrity-checked between reps (checksums
   on the baseline tree, its JIT cache, and the model directory); drift aborts the eval.

   Each head commit is evaluated once; pushing a new commit re-queues it. A regression
   anywhere outranks an improvement anywhere else.

4. **Auto-merge — earned, not default.** A PR merges itself when it has cleared every
   gate a reviewer would rely on: `eval:pass`, `eval:noise`, or a confirmed
   `eval:major`, with an attested receipt
   (intel-verified, TEE verdict byte-identical to the local scorer), pinned to the exact
   evaluated head sha so a commit pushed mid-eval can never ride in unevaluated.
   Requiring a *speedup* would mean a bug fix could never merge itself, so measured-
   harmless counts too. Docs-only PRs skip the GPU entirely and merge on green CI.
   `eval:landmark`, `eval:slower`, `eval:reject`, `eval:tainted`, `eval:error`, an
   unstable confirmation round, and anything without a receipt wait for the maintainer.
   Docs-only means *prose only* — `docs/**.md`, README, CONTRIBUTING, LICENSE. A PR
   touching `scripts/`, `.github/`, or build config is never docs-only, whatever else
   it leaves alone.

   **Changing the harness.** A PR that edits `braid/bench/` gets a **cross-check**: the
   bot runs your bench beside the pinned one, same engine, same session, and compares
   every configuration they share. Agree within ±2% and the harness change is cleared —
   your PR rejoins the ordinary pass/noise ladder and can auto-merge, with the comparison
   table in the comment. Disagree and it stays `eval:tainted` for a human. (This is
   evidence, not proof: configurations your bench *adds* have no pinned counterpart and
   are not validated.) Changes to `tests/` or `braid/reference/` are never cleared this
   way — weakening a parity assertion doesn't change throughput, so no measurement can
   expose it. Those always get human eyes.

Every verdict is recorded three ways: the PR comment and label, a `braid/eval` commit
status on your head sha (branch protection on `main` requires it), and an append-only
git note under `refs/notes/braid-eval` carrying the full measured record — fetch it
with `git fetch origin refs/notes/braid-eval:refs/notes/braid-eval && git notes
--ref=braid-eval show <sha>` if you want to audit any verdict ever issued.

Residual honesty note:
PR code runs as root on the eval box, so software-only isolation has limits — the
checks turn a silent one-line cheat into overt sabotage that has to survive human
review of your diff.

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
