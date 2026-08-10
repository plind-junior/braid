---
name: remote-gpu-workflow
description: Use when running anything on braid's GPU — tests, benchmarks, kernel builds, model staging, llama.cpp — or when the box is off, unreachable, or billing. Triggers on make test-remote, make provision, scripts/remote.sh, rsync, ssh, vastai, "start the GPU", "run the tests", "connection refused", "no CUDA device", "how much is this costing", first-run JIT compile times. Do NOT use for interpreting a measurement (benchmark-remote-5090) or for kernel source questions (sm120-gdn-kernels).
---

# Remote GPU workflow — braid

**Nothing runs locally.** There is no CUDA toolkit and no GPU on the dev machine; every test,
benchmark and kernel build executes on a rented vast.ai RTX 5090. Local `pytest` will fail at
import, and that failure means "you forgot `-remote`", not "the code is broken".

## The one thing that will bite you

`scripts/remote.sh` runs **`rsync -az --delete`** from the local repo to `/root/braid` before
every command. The local tree is the only source of truth. **Anything created or edited on the
remote inside the repo directory is destroyed on the next invocation** — a hotfix typed over
SSH, a results file written to the repo root, a `git stash` made remotely. Write outputs to
`/root/` (outside the synced directory) and pull them back explicitly.

Excluded from the sync: `.git`, `__pycache__`, `*.pyc`, `.pytest_cache`, `.ruff_cache`,
`.venv`. So the remote has no git history — run git locally, always.

## Commands

| Task | Command |
|---|---|
| Start the box | `make gpu-start` |
| **Stop the box** | `make gpu-stop` |
| Cost / state / real rental hours | `make gpu-status` |
| Install deps (first boot after a rebuild) | `make provision` |
| Full test suite | `make test-remote` |
| One test file | `make test-remote ARGS="tests/test_gdn_decode_kernel.py -s"` |
| Anything else | `./scripts/remote.sh <command...>` |

`ARGS` **replaces** the default target rather than appending, so the form above runs exactly
that file.

## Cost and lifecycle

The instance bills **~$0.79/hr while running** and storage-only while stopped. Stop it as soon
as a batch of GPU work is done — an idle running box is pure loss.

**Always `stop`, never `destroy`.** Stop preserves the 300 GB disk holding ~13 GB of staged
models and the built llama.cpp tree; destroying costs a re-download and a ~10 min rebuild.
`make gpu-stop` and `make gpu-start` are the only lifecycle commands that should ever run.

`uptime` inside the container reports the **host kernel's** uptime, not the rental duration —
it will read weeks. `make gpu-status` reads `duration` from the API, which is the real number.

## Connection

```bash
export BRAID_SSH_KEY=~/.ssh/rtx5090 BRAID_SSH_HOST=root@ssh5.vast.ai BRAID_SSH_PORT=15458
```

Defaults are baked into `scripts/remote.sh`. **vast.ai can reassign the SSH proxy host and port
across a stop/start cycle** — if a command hangs or gets `Connection refused` right after
`make gpu-start`, re-check the instance's current host/port before debugging anything else.
`ConnectTimeout` is 25 s, so a wrong port looks like a hang, not an error.

## Provisioning

`make provision` installs `python3-pip`, `ninja-build`, `ccache`, `rsync`, then `pytest` and
`numpy`. Notes worth knowing:

- **torch is preinstalled** on the image (2.11.0+cu128) — the script deliberately does not
  reinstall it. A `pip install torch` that "fixes" something will pull a build for the wrong
  CUDA and break the kernel path.
- **numpy is required, not optional.** Without it torch 2.11 emits "Failed to initialize NumPy"
  and some tensor conversions fail outright.
- Provisioning prints `torch.cuda.get_arch_list()`, which reports only `sm_120`. That is
  correct and not a problem: it constrains torch's prebuilt kernels, not what nvcc emits for
  ours. braid's JIT sets `TORCH_CUDA_ARCH_LIST=12.0a` itself.
- The Nsight suite (`ncu`, `nsys`) is **not** installed by provisioning. Check before planning
  a profiling session.

## Kernel builds

CUDA extensions JIT-build on first use via `torch.utils.cpp_extension.load` — the first
`make test-remote` after a source change to `braid/kernels/csrc/` pays a compile, later ones
hit the cache. `MAX_JOBS=8` is set because nvcc is memory-hungry; `ccache` is installed to make
rebuilds cheap. The build output lands in `~/.cache/torch_extensions` on the remote unless
`TORCH_EXTENSIONS_DIR` redirects it into the repo (where the next rsync would delete it).

If a build error mentions an arch or an unsupported instruction, read
`sm120-gdn-kernels` — the flags in `braid/kernels/loader.py` are deliberate and the two
non-obvious ones (`12.0a`, no `--use_fast_math`) are load-bearing.

## Before a measurement run

Benchmarks have preconditions the test suite does not — no other GPU process, warm clocks,
health sampling. Those belong to `benchmark-remote-5090`; check it before trusting a number
that came off this box.

## Common mistakes

| Mistake | Consequence |
|---|---|
| Editing a file over SSH on the remote | Silently destroyed by the next `rsync --delete` |
| Writing benchmark output into the repo dir on the remote | Same — write to `/root/`, copy back explicitly |
| Leaving the box running after a batch | ~$0.79/hr for nothing |
| `vastai destroy` instead of `stop` | Loses 13 GB of staged models and the llama.cpp build |
| Reading `uptime` as rental duration | Reports host kernel uptime; use `make gpu-status` |
| Running `pytest` locally | No GPU, no toolkit — always `make test-remote` |
| Reinstalling torch to fix an import | Wrong CUDA build; torch is preinstalled |
| Debugging a "hang" after `gpu-start` | Usually a reassigned SSH port, not a code problem |
