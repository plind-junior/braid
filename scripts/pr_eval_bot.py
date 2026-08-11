#!/usr/bin/env python3
"""braid PR auto-evaluator: does the PR *really* update the performance?

The CI gates check the claim's paperwork; this bot checks the claim. For each
eligible PR head it starts the rented RTX 5090, runs the test suite on the
PR's merged tree, benches that tree against origin/main **in the same box
session**, posts the measured table as a PR comment, applies an eval:* label,
and stops the box. Modeled on sparkinfer's pr_eval_bot, with braid's
measurement discipline built in rather than bolted on:

  * Same-session A/B, arms alternated per rep, median of --reps. The ~2% rule
    (cross-session deltas under ~2% are not attributable to code on this box)
    is satisfied by construction: both arms run in one session, and the
    verdict bar is that same 2%.
  * The measured arm is `graphed-kvbucket` from braid.bench.decode_speed —
    the serving-shaped configuration — at the locked-target batches 16 and 64.
  * A no-op PR must come back `eval:noise`, not `eval:pass`. That is the
    bot's own null test.

Trust boundary — the part that must never be relaxed:

  * PR code executes ONLY on the rented box (that is what evaluating it
    means; the box holds staged public models and nothing secret). It is
    checked out locally via `git archive` — never built, imported, or run on
    this machine — and rsynced to /root/braid_eval/, outside the dev tree.
  * The gh and vast credentials live on this machine and are never copied to
    the box. The bot itself always runs from the local main tree.
  * The PR does not grade itself. The harness — `braid/bench/`,
    `braid/reference/`, `tests/` — is pinned: those directories are taken
    from the BASE sha and overlaid onto the PR tree before it is pushed to
    the box, so the PR's engine code is measured by main's bench and gated
    by main's tests against main's oracles. A PR that modifies any pinned
    path (or this bot, or the workflows) is evaluated anyway but its best
    verdict is capped at `eval:tainted` — the numbers are trustworthy, the
    harness diff still needs human eyes before merge. New files ADDED under
    pinned paths are not tainting (they cannot weaken the gate) but they do
    not run during the eval; the comment lists them as unexercised.
  * The arms are isolated as far as software can manage on a shared root
    box: per-eval wipe of both JIT extension caches, checksum re-push of the
    main tree before every main rep (PR code runs first in the session and
    could otherwise edit the baseline), hash-lock of main's JIT cache after
    its first build, GPU-idle assertion before each main rep, and a
    model-directory manifest taken before any PR code runs and re-verified
    before the verdict. Any drift aborts the eval as TAMPER SUSPECTED.
  * Residual risk, stated honestly: PR code runs as root on the box, so a
    determined attacker can still compromise the interpreter, site-packages,
    or rc files, or leave a daemon that races the checks (TOCTOU). Closing
    that requires a fresh container per arm — on the roadmap. The checks
    above turn a silent one-line cheat into overt sabotage code that has to
    survive human review of the diff.

Durable evidence — verdicts outlive editable comments:

  * Every verdict lands as a `braid/eval` commit status on the PR head sha
    (statuses are what branch protection can require).
  * The full measured record (the canonical evidence bundle, the verdict
    document, and the attested receipt) is appended as a git note on the
    head under refs/notes/braid-eval and pushed — an append-only audit
    trail anyone can fetch — and mirrored to a local ledger at
    ~/braid-pr-eval-ledger.jsonl.
  * The verdict itself is attested: the policy (scripts/eval_scorer.py) is
    shipped byte-identical into a polaris.computer Intel TDX machine with
    the evidence bundle (egress sealed), re-derives the verdict there, and
    the returned DCAP quote binds scorer + bundle + verdict in its
    report_data. scripts/verify_receipt.py recomputes every binding
    offline. The local and TEE verdicts must agree byte-for-byte or the
    eval aborts. Scope honesty: the receipt proves the SCORING; the 5090
    measurement itself still happens outside any TEE (consumer GPUs have
    no confidential-compute mode) — same ceiling as every attested-eval
    pipeline on consumer hardware.

Box ownership — the part that keeps this from costing money or corrupting a
measurement:

  * If the box is already running, the bot SKIPS the whole cycle: a running
    box is somebody's session, and rsyncing or benching under it would break
    the never-touch-a-box-mid-benchmark rule. Evals wait for the next poll.
  * If the bot starts the box, it stops it in a finally block, with retries,
    and screams into the log if the stop fails. Never destroys.

Graded verdicts (the ladder lives in eval_scorer.verdict):

  * The bar is not a constant. It is max(2%, 3x this session's observed
    spread) — 2% stays the floor, and a jittery session raises its own bar.
  * > +25% eval:landmark · +10..25% eval:major · bar..+10% eval:pass ·
    within +/-bar eval:noise · -bar..-5% eval:slower · worse or suite
    failed eval:reject. Above +10% the bot escalates --confirm-reps more
    reps and requires the two rounds to agree.

Outcome policy (enforced by settle, kill switch BRAID_AUTOMERGE=0). A
verdict ends the PR: it is merged, or it is closed with the reason. The
only things that wait are the ones that are the BOT's fault, never the
contributor's.

  * eval:pass, eval:noise and (confirmed) eval:major with an attested
    receipt -> the bot merges, pinned to the exact evaluated head sha
    (--match-head-commit): a commit pushed mid-eval can never ride in
    unevaluated. Requiring a speedup would mean a bug fix could never
    merge itself, so measured-harmless counts too.
  * eval:reject, eval:slower and eval:tainted -> the bot closes the PR
    and says why. Closing is reversible and the comment says so: push a
    commit and reopen, and the new head re-evaluates from scratch.
  * eval:landmark waits for a human: past 25% on this engine the likelier
    explanation is work that stopped happening, and a person should say
    so. It is the only verdict that waits.
  * eval:error waits and retries — the eval broke, not the PR. So does a
    mergeable verdict with no attested receipt (a polaris.computer
    outage) and an unstable confirmation round. Closing a PR for our own
    infrastructure would blame the contributor for our box.
  * A `hold` label is never closed, matching eligible().
  * Docs-only PRs (prose allowlist only — never scripts/, .github/, or
    build config) merge on green CI without touching the GPU. They never
    reach a verdict, so the binary outcome does not govern them.

POLICY_VERSION is stamped into the verdict marker, so bumping it makes
every open PR look un-evaluated and it is re-measured under current code.
That is the whole staleness mechanism: no code path ever acts on a label
produced by a scorer or a policy that has since changed. Bump it
deliberately — a bump re-runs the box for every open eligible PR.

Usage:
  python3 scripts/pr_eval_bot.py              # poll: eval all eligible heads
  python3 scripts/pr_eval_bot.py --pr 7       # eval one PR (still eligibility-checked)
  python3 scripts/pr_eval_bot.py --pr 7 --force   # bypass eligibility+idempotency (smoke tests)
  python3 scripts/pr_eval_bot.py --dry-run    # print the plan, touch nothing
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

# The verdict policy lives in eval_scorer.py — one file, imported here and
# shipped byte-identical into the TEE for the attested receipt. The bot must
# never grow its own copy of these functions.
_SCORER_PATH = pathlib.Path(__file__).resolve().parent / "eval_scorer.py"
_spec = importlib.util.spec_from_file_location("eval_scorer", _SCORER_PATH)
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)
classify_taint = scorer.classify_taint
verdict = scorer.verdict
median = scorer.median
NOISE_PCT = scorer.NOISE_PCT
PINNED_DIRS = scorer.PINNED_DIRS
INFRA_FILES = scorer.INFRA_FILES
INFRA_PREFIXES = scorer.INFRA_PREFIXES

INSTANCE = os.environ.get("VAST_INSTANCE", "47055458")
SSH_KEY = os.path.expanduser(os.environ.get("BRAID_SSH_KEY", "~/.ssh/rtx5090"))
SSH_HOST = os.environ.get("BRAID_SSH_HOST", "root@ssh5.vast.ai")
SSH_PORT = os.environ.get("BRAID_SSH_PORT", "15458")
BENCH_MODEL = os.environ.get("BRAID_EVAL_MODEL_DIR", "/root/models/Qwen3.5-9B")
REMOTE_BASE = "/root/braid_eval"          # outside /root/braid: never collides with dev rsync
EXT_BASE = "/root/torch_ext_eval"         # JIT caches, one per arm, never the dev cache
ARM = "graphed-kvbucket"                  # the serving-shaped arm decode_speed reports
EVAL_BATCHES = [16, 64]                   # the locked targets the verdict is read at
RUNTIME_PREFIXES = ("braid/", "tests/")
STATUS_CONTEXT = "braid/eval"
POLARIS_ATTEST_URL = os.environ.get("BRAID_POLARIS_ATTEST_URL",
                                    "https://polaris.computer/v1/attest")
POLARIS_KEY = os.environ.get("BRAID_POLARIS_API_KEY", "")
SCORER_WORKLOAD = "python3 /in/eval_scorer.py /in/bundle.json"
AUTOMERGE = os.environ.get("BRAID_AUTOMERGE", "1") != "0"
# Bump when the outcome policy or the scorer changes meaning. Every open PR
# is then re-measured under the new rules instead of being settled on a
# verdict the current code would not reach. Costs a box run per open PR.
POLICY_VERSION = 2
GH_USER = os.environ.get("BRAID_GH_USER", "plind-junior")   # never post as another identity
MIN_FREE_GIB = float(os.environ.get("BRAID_MIN_FREE_GIB", "30"))  # suite peaks at ~26 of 32
NOTES_REF = "braid-eval"                  # refs/notes/braid-eval
LEDGER = os.path.expanduser("~/braid-pr-eval-ledger.jsonl")
EVAL_LABELS = {
    "eval:landmark": ("5A2D82", "measured speedup beyond 25% — extraordinary, human reads it"),
    "eval:major": ("0A6EA1", "measured speedup beyond 10%, confirmed with escalated reps"),
    "eval:pass": ("0E8A16", "measured speedup beyond the session's noise bar"),
    "eval:noise": ("FBCA04", "measured delta within the noise bar — harmless"),
    "eval:slower": ("E36209", "small measured regression — may be a fair trade, human decides"),
    "eval:tainted": ("5319E7", "touches harness files, not cleared by cross-check"),
    "eval:reject": ("B60205", "suite failed or measured regression beyond 5%"),
    "eval:error": ("D93F0B", "evaluation could not complete"),
}
STATUS_STATE = {                          # eval label -> commit-status state
    "eval:landmark": "success",           # gate passed; the merge is the human's
    "eval:major": "success",
    "eval:pass": "success",
    "eval:noise": "success",
    "eval:slower": "failure",
    "eval:tainted": "failure",
    "eval:reject": "failure",
    "eval:error": "error",
}
SYNC_EXCLUDES = [".git", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", ".venv"]


# ---------------------------------------------------------------------------
# Pure policy — everything here is unit-tested locally (tests/test_pr_eval_bot.py)
# and none of it shells out.

def ticked_5090(body: str | None) -> bool:
    """Same convention as the pr-gate workflow: a ticked box on a 5090 line."""
    import re
    if not body:
        return False
    return any("5090" in ln and re.search(r"-\s*\[\s*[xX]\s*\]", ln)
               for ln in body.split("\n"))


def touches_runtime(paths: list[str]) -> bool:
    return any(p.startswith(RUNTIME_PREFIXES) for p in paths)


def overlay_harness(pr_dir: str, base_dir: str, dirs: tuple[str, ...] = PINNED_DIRS) -> None:
    """Replace the PR tree's harness dirs with the base tree's, wholesale.

    Wholesale (delete then copy), not a merge: a merge would keep PR-added
    files, and a PR-added conftest.py could monkeypatch the trusted tests
    from inside their own pytest process.
    """
    for d in dirs:
        dst, src = os.path.join(pr_dir, d), os.path.join(base_dir, d)
        shutil.rmtree(dst, ignore_errors=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst)


def marker(sha: str) -> str:
    """The verdict stamp. Carries POLICY_VERSION so a verdict reached under
    an older policy stops counting as evaluated and is measured again."""
    return f"<!-- braid-eval sha={sha} policy={POLICY_VERSION} -->"


def error_marker(sha: str) -> str:
    return f"<!-- braid-eval-error sha={sha} -->"


def already_evaluated(comment_bodies: list[str], sha: str) -> bool:
    """A verdict parks the head for good; errors get one automatic retry.

    Error comments carry a different marker so a transient infra failure is
    retried on the next poll — but only once, or a persistent failure would
    start the box (and bill) every cycle forever. After two errors the head is
    parked until a new commit arrives.
    """
    bodies = [b or "" for b in comment_bodies]
    if any(marker(sha) in b for b in bodies):
        return True
    return sum(error_marker(sha) in b for b in bodies) >= 2


def pick_tok_s(bench_stdout: str, batch: int, arm: str = ARM) -> float:
    """tok/s for (arm, batch) from decode_speed --json output.

    The bench prints health lines around the JSON, so take the last line that
    parses as an object with "arms". A missing arm is an error, not a zero —
    a PR that renames or drops the serving arm must fail loudly.
    """
    doc = None
    for ln in bench_stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                cand = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(cand, dict) and "arms" in cand:
                doc = cand
    if doc is None:
        raise ValueError("no decode_speed JSON found in bench output")
    for a in doc["arms"]:
        if a.get("name") == arm and a.get("batch") == batch:
            return float(a["tok_per_s"])
    raise ValueError(f"arm {arm!r} batch {batch} missing from bench output")


def build_bundle(pr: int, head: str, eval_sha: str, base_sha: str, mode: str,
                 batches: list[int], reps: int, tests_ok: bool,
                 samples: dict, name_status: list, suite_tail: str,
                 model_manifest: str, main_ext_hash: str | None,
                 crosscheck: dict | None = None,
                 confirm: dict | None = None) -> dict:
    """The canonical evidence bundle: everything the verdict is a function
    of, in one JSON document. The scorer (locally AND inside the TEE)
    derives the verdict from this and nothing else; its hash is bound into
    the attested receipt's report_data."""
    return {
        "schema": scorer.SCHEMA_IN,
        "pr": pr, "head": head, "eval_sha": eval_sha, "base_sha": base_sha,
        "mode": mode, "arm": ARM, "batches": [int(b) for b in batches],
        "reps": reps, "noise_pct": NOISE_PCT, "box": INSTANCE,
        "tests_ok": tests_ok,
        "suite_tail_sha256": hashlib.sha256(suite_tail.encode()).hexdigest(),
        "samples": {a: {str(b): list(v) for b, v in per.items()}
                    for a, per in samples.items()},
        "name_status": [list(p) for p in name_status],
        "integrity": {"model_manifest": model_manifest,
                      "main_ext_hash": main_ext_hash or ""},
        "crosscheck": crosscheck,
        "confirm": confirm,
    }


MERGEABLE_LABELS = ("eval:pass", "eval:noise", "eval:major")
# Why each closing verdict closes, in the contributor's terms. These reach a
# public comment, so they explain rather than announce.
CLOSING_LABELS = {
    "eval:reject": "the pinned test suite failed on the merged tree, or the "
                   "measured regression is worse than 5%",
    "eval:slower": "the same-session A/B measured a regression past the "
                   "session's noise bar",
    "eval:tainted": "it modifies files the measurement itself depends on, and "
                    "the cross-check could not clear the numbers",
}


def settle(label: str, receipt: dict | None, stable: bool | None = None,
           labels: tuple[str, ...] = ()) -> tuple[str, str]:
    """The declared outcome policy: a verdict ends the PR.

    Returns one of "merge", "close" or "wait", with the reason. Pure, so it
    is unit-tested without a GPU, a box or a token.

    A PR merges itself when it has cleared every gate a reviewer would rely
    on: main's test suite passed on the merged tree, the same-session A/B
    found no regression, nothing in the diff can have influenced the
    measurement, and the verdict was recomputed inside a TEE. That is
    eval:pass (measured faster) and eval:noise (measured harmless) alike —
    demanding a speedup would mean a bug fix could never merge itself.
    eval:major merges on the same terms once its escalated confirmation
    round agrees with the first.

    A PR is closed, with the reason, when the measurement says the change
    is not wanted as it stands: eval:reject, eval:slower, eval:tainted.
    Closing is reversible and the comment says so.

    Only two things wait, and both are the bot's fault rather than the
    contributor's. eval:error means the evaluation could not complete. A
    mergeable verdict with no verified receipt means polaris.computer was
    unreachable, and an unstable confirmation round means our own two
    rounds disagreed — closing on any of those would blame a contributor
    for our box. eval:landmark also waits, by explicit policy: past 25% the
    likeliest explanation on this engine is work that stopped happening
    rather than a breakthrough, and a person should say so.

    The caller aborts the eval if the TEE verdict differed from the local
    one, so a non-None receipt here means they agreed.
    """
    if "hold" in labels:
        return "wait", "hold label — the maintainer is keeping this open"
    if label in CLOSING_LABELS:
        return "close", CLOSING_LABELS[label]
    if label == "eval:landmark":
        return "wait", ("past +25% — extraordinary enough that a human reads it "
                        "before it lands")
    if label not in MERGEABLE_LABELS:
        # eval:error and anything a future verdict() learns to emit.
        return "wait", f"{label} — the evaluation, not the PR, is what failed"
    if stable is False:
        return "wait", ("the confirmation round disagreed with the first — our "
                        "measurement is unstable, so the PR is not judged on it")
    if receipt is None:
        return "wait", ("no attested receipt — the attestation service was "
                        "unreachable, which is our problem and not this PR's")
    if not receipt.get("verification", {}).get("intel_verified"):
        return "wait", "receipt exists but is not intel_verified — not judged on it"
    return "merge", f"{label} with an intel-verified attested receipt"


def close_comment(label: str, reason: str) -> str:
    """The message a closed PR carries. States the measurement that closed
    it and the exact way back — a close is a verdict, not a door slam."""
    return (
        f"## Closed on the measured verdict — `{label}`\n\n"
        f"The evaluation above closed this PR because {reason}.\n\n"
        "**This is reversible.** The verdict is about the code as it stands at "
        "this head, not about the idea. Push a commit and reopen: the new head "
        "carries no verdict stamp, so it is measured again from scratch on the "
        "same box, with the same pinned harness.\n\n"
        "If you think the measurement itself is wrong, say so in a comment and "
        "label the PR `hold` — the bot never closes a held PR, and the "
        "maintainer will read it.\n\n"
        "<sub>Automated outcome (scripts/pr_eval_bot.py). Every verdict ends in "
        "a merge or a close; the receipt for this one is permanent in "
        f"`refs/notes/{NOTES_REF}`.</sub>"
    )


INERT_FILES = ("README.md", "CONTRIBUTING.md", "LICENSE", "CHANGELOG.md")


def inert_paths(paths: list[str]) -> bool:
    """Prose only — nothing that executes, gates, or configures anything.

    An ALLOWLIST, deliberately. The tempting definition is 'touches no
    runtime path', but that is true of scripts/pr_eval_bot.py, of
    .github/workflows/, of the Makefile and pyproject.toml — so a PR
    rewriting the grader itself would have sailed through the docs-only
    merge with nobody reading it. Anything not named here goes to a human,
    including .github/ (its PR template carries the 5090 attestation
    checkbox) and every dotfile.
    """
    if not paths:
        return False
    return all(p in INERT_FILES or (p.startswith("docs/") and p.endswith(".md"))
               for p in paths)


def docs_only_automerge(info: dict, checks: list[dict]) -> tuple[bool, str]:
    """Docs-only PRs never reach the GPU, so their gate is ordinary CI.

    `eligible()` has already decided this PR touches no runtime path, but
    that is far too weak to merge on: see `inert_paths`. Merging also
    requires everything else to be green — a docs PR can break the lint job
    or carry a blocking review — and an ordinary open, non-held, non-draft
    PR.
    """
    labels = [lb["name"] for lb in info.get("labels", [])]
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, ready PR"
    if "hold" in labels:
        return False, "hold label"
    paths = [f["path"] for f in info.get("files", [])]
    if not inert_paths(paths):
        outside = [p for p in paths
                   if not (p in INERT_FILES
                           or (p.startswith("docs/") and p.endswith(".md")))]
        return False, (f"touches non-prose paths a human must read: "
                       f"{', '.join(outside[:4])}")
    if not checks:
        return False, "no CI checks reported yet"
    for c in checks:
        state = (c.get("conclusion") or c.get("state") or "").upper()
        name = c.get("name") or c.get("context") or "?"
        if name == STATUS_CONTEXT:
            continue                      # our own not-required stamp
        if state in ("", "PENDING", "IN_PROGRESS", "QUEUED"):
            return False, f"check {name} is still running"
        if state not in ("SUCCESS", "NEUTRAL", "SKIPPED"):
            return False, f"check {name} is {state.lower()}"
    return True, "docs-only and every CI check is green"


def automerge(pr: int, head: str) -> tuple[bool, str]:
    """Merge the exact evaluated head — and only it. --match-head-commit
    makes GitHub refuse if anything was pushed after the eval snapshot."""
    r = run(["gh", "pr", "merge", str(pr), "--merge",
             "--match-head-commit", head], timeout=120, check=False)
    if r.returncode == 0:
        return True, "merged"
    return False, (r.stderr or r.stdout).decode()[-400:]


def close_with_reason(pr: int, label: str, reason: str) -> tuple[bool, str]:
    """Close the PR and leave the reason in the same call, so a PR can never
    end up closed with no explanation attached to it."""
    r = run(["gh", "pr", "close", str(pr), "--comment",
             close_comment(label, reason)], timeout=120, check=False)
    if r.returncode == 0:
        return True, "closed"
    return False, (r.stderr or r.stdout).decode()[-400:]


def request_receipt(bundle: dict) -> dict | None:
    """Re-score the bundle inside a polaris.computer Intel TDX machine and
    return the attested receipt (raw DCAP quote + bindings), or None.

    Best-effort by design: the receipt is evidence, not the gate — a
    polaris.computer outage must not block evaluations. But a receipt that
    RETURNS a different verdict than the local scorer is handled by the
    caller as a hard error, because the scorer is deterministic and a
    divergence means a bug or tampering."""
    if not POLARIS_KEY:
        print(">> BRAID_POLARIS_API_KEY unset — verdict will not carry a TDX receipt")
        return None
    payload = json.dumps({
        "nonce": bundle["head"],                    # 40 hex chars: the PR head sha
        "workload": SCORER_WORKLOAD,
        "egress": "none",
        "files": {
            "/in/eval_scorer.py":
                base64.b64encode(_SCORER_PATH.read_bytes()).decode(),
            "/in/bundle.json":
                base64.b64encode(scorer.canonical(bundle).encode()).decode(),
        },
    }).encode()
    req = urllib.request.Request(
        POLARIS_ATTEST_URL, data=payload,
        headers={"Authorization": f"Bearer {POLARIS_KEY}",
                 "Content-Type": "application/json",
                 # Cloudflare fronts the API and 403s the default urllib UA
                 "User-Agent": "braid-pr-eval-bot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            doc = json.load(r)
    except (OSError, json.JSONDecodeError) as e:
        print(f"!! attested receipt unavailable: {e}", file=sys.stderr)
        return None
    if doc.get("exit_code") != 0:
        print(f"!! scorer failed inside the TEE (exit {doc.get('exit_code')}): "
              f"{str(doc.get('stdout', ''))[:400]}", file=sys.stderr)
        return None
    return doc


# ---------------------------------------------------------------------------
# Shell plumbing

def run(cmd: list[str], timeout: int = 120, check: bool = True,
        input_: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=False, input=input_,
                          timeout=timeout, check=check)


def pin_gh_identity() -> None:
    """Post as the repo owner, never as whatever `gh` last switched to.

    This machine's gh keyring holds several accounts (other projects push
    as other identities), and `gh auth switch` is global — an unrelated
    script flipping the active account once made this bot comment under the
    wrong user. Resolve the intended account's token explicitly and export
    it for every gh subprocess; GH_TOKEN outranks the keyring's active
    account, so the identity cannot drift mid-run.
    """
    if os.environ.get("GH_TOKEN"):
        return
    r = run(["gh", "auth", "token", "--user", GH_USER], check=False)
    token = r.stdout.decode().strip()
    if r.returncode != 0 or not token:
        raise SystemExit(
            f"!! refusing to run: no gh token for {GH_USER!r}. The bot must "
            f"post as the repo owner, and the active gh account may be "
            f"someone else. Fix with: gh auth login --user {GH_USER}")
    os.environ["GH_TOKEN"] = token
    who = run(["gh", "api", "user", "--jq", ".login"], check=False).stdout.decode().strip()
    if who != GH_USER:
        raise SystemExit(f"!! gh identity is {who!r}, expected {GH_USER!r} — aborting")
    print(f">> posting as {GH_USER}")


def gh(args: list[str], timeout: int = 60) -> str:
    return run(["gh", *args], timeout=timeout).stdout.decode()


def gh_json(args: list[str]) -> object:
    return json.loads(gh(args))


class BoxUnavailable(RuntimeError):
    """vast.ai has no GPU for us right now. Not a fault, and not the PR's —
    the next poll simply tries again."""


# vast.ai answers `start instance` with a plain-English refusal and exit 0
# when the host has nothing free, and *queues* the state change. The box then
# never boots, every SSH route times out, and the old code spent seven
# minutes concluding "ssh never came up" — a true statement about the wrong
# thing, once per poll, with a start/stop cycle billed for each.
UNAVAILABLE_SIGNATURES = (
    "resources are currently unavailable",
    "state change queued",
    "no available capacity",
    "insufficient capacity",
)


def start_was_refused(stdout: str) -> str | None:
    """The refusal reason in vast.ai's reply, or None if the start took."""
    low = (stdout or "").lower()
    for sig in UNAVAILABLE_SIGNATURES:
        if sig in low:
            return " ".join((stdout or "").split())[:200]
    return None


def endpoint_candidates(ssh_url: str, info: dict,
                        default: tuple[str, str]) -> list[tuple[str, str]]:
    """Every route vast.ai publishes to the box, best-known-first. Pure.

    vast.ai exposes two: a *proxy* (ssh_host/ssh_port, e.g. ssh5.vast.ai,
    always routable) and a *direct* one (public_ipaddr/direct_port_start,
    faster, but only reachable if the host actually forwards that range).
    Both are remapped across stop/start cycles.

    `vastai ssh-url` reports the direct route whether or not it works. The
    first version of this code trusted it and *replaced* the endpoint with
    it, which traded a working proxy for an unreachable direct port and
    spent seven minutes failing to connect to a box that was up the whole
    time — the exact 'ssh never came up' it was written to prevent, in the
    other direction. So this never replaces: it collects, and the caller
    latches whichever route answers first.
    """
    cands: list[tuple[str, str]] = []
    m = re.match(r"ssh://(.+):(\d+)\s*$", (ssh_url or "").strip())
    if m:
        cands.append((m.group(1), m.group(2)))
    host, port = info.get("ssh_host"), info.get("ssh_port")
    if host and port:
        cands.append((host if "@" in str(host) else f"root@{host}", str(port)))
    ip, direct = info.get("public_ipaddr"), info.get("direct_port_start")
    if ip and direct:
        cands.append((f"root@{ip}", str(direct)))
    cands.append(default)                 # compiled-in default, never first
    seen, ordered = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


class Box:
    """The rented 5090, held politely: skip if busy, stop if we started it."""

    def __init__(self) -> None:
        self.we_started = False
        self.ssh_host, self.ssh_port = SSH_HOST, SSH_PORT

    def status(self) -> str:
        out = run(["vastai", "show", "instance", INSTANCE, "--raw"]).stdout
        return json.loads(out).get("actual_status", "unknown")

    def resolve_endpoints(self) -> list[tuple[str, str]]:
        """Every published route to the box, best-known-first."""
        if os.environ.get("BRAID_SSH_HOST"):
            return [(SSH_HOST, SSH_PORT)]
        url = run(["vastai", "ssh-url", INSTANCE], check=False).stdout.decode()
        try:
            info = json.loads(
                run(["vastai", "show", "instance", INSTANCE, "--raw"]).stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError):
            info = {}
        return endpoint_candidates(url, info, (SSH_HOST, SSH_PORT))

    def ssh(self, cmd: str, timeout: int = 120,
            connect_timeout: int = 25) -> subprocess.CompletedProcess:
        return run(["ssh", "-i", SSH_KEY, "-p", self.ssh_port,
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", f"ConnectTimeout={connect_timeout}", self.ssh_host, cmd],
                   timeout=timeout, check=False)

    def start(self) -> None:
        r = run(["vastai", "start", "instance", INSTANCE])
        # Before the wait, not after: we_started must be true so the queued
        # state change is cancelled by stop(), or a box that frees up an hour
        # later boots with nobody watching and bills until someone notices.
        self.we_started = True
        refused = start_was_refused(r.stdout.decode(errors="replace"))
        if refused:
            raise BoxUnavailable(refused)
        deadline = time.time() + 420              # ~7 min of patience
        cands: list[tuple[str, str]] = []
        resolved_at = 0.0
        tried: set[str] = set()
        while time.time() < deadline:
            time.sleep(10)
            if not cands or time.time() - resolved_at > 60:
                cands = self.resolve_endpoints()  # mapping can appear late
                resolved_at = time.time()
            for host, port in cands:
                tried.add(f"{host}:{port}")
                self.ssh_host, self.ssh_port = host, port
                if self.ssh("true", timeout=20, connect_timeout=10).returncode == 0:
                    print(f">> ssh up at {host}:{port}")
                    return
                if time.time() >= deadline:
                    break
        raise RuntimeError("box started but ssh never came up; tried "
                           + ", ".join(sorted(tried)))

    def stop(self) -> None:
        if not self.we_started:
            return
        for attempt in range(4):
            try:
                run(["vastai", "stop", "instance", INSTANCE])
                self.we_started = False
                print(f">> box {INSTANCE} stopped")
                return
            except subprocess.SubprocessError as e:
                print(f"!! stop attempt {attempt + 1} failed: {e}", file=sys.stderr)
                time.sleep(15)
        print(f"!! BOX {INSTANCE} MAY STILL BE RUNNING AND BILLING — stop it "
              f"by hand: vastai stop instance {INSTANCE}", file=sys.stderr)

    def gpu_busy(self) -> bool:
        r = self.ssh("nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
        return r.returncode != 0 or int(r.stdout.decode().strip() or "1") != 0

    def gpu_idle_within(self, seconds: int = 60) -> bool:
        """CUDA contexts linger in nvidia-smi for a few seconds after a
        bench process exits; a tamper check that fires inside that teardown
        window is a false alarm (it cost PR #5 an eval:error). Grace-poll
        before calling it tamper — a real daemon stays busy past it."""
        deadline = time.time() + seconds
        while True:
            if not self.gpu_busy():
                return True
            if time.time() > deadline:
                return False
            time.sleep(5)

    def _rsync_ssh(self) -> str:
        return (f"ssh -i {SSH_KEY} -p {self.ssh_port} "
                f"-o StrictHostKeyChecking=accept-new")

    def push_tree(self, local_dir: str, remote_dir: str) -> None:
        ex = [f"--exclude={e}" for e in SYNC_EXCLUDES]
        run(["rsync", "-az", "--delete", *ex, "-e", self._rsync_ssh(),
             f"{local_dir}/", f"{self.ssh_host}:{remote_dir}/"], timeout=300)

    def verify_tree(self, local_dir: str, remote_dir: str) -> list[str]:
        """Checksum-compare the remote tree against the local truth and
        repair it. Returns the itemized drift — non-empty means something on
        the box rewrote the tree since the last push, which after PR code
        has run is tamper evidence, not noise."""
        ex = [f"--exclude={e}" for e in SYNC_EXCLUDES]
        r = run(["rsync", "-azc", "--delete", "--itemize-changes", *ex,
                 "-e", self._rsync_ssh(),
                 f"{local_dir}/", f"{self.ssh_host}:{remote_dir}/"], timeout=600)
        return [ln for ln in r.stdout.decode().splitlines() if ln.strip()]

    def wipe(self, *paths: str) -> None:
        joined = " ".join(paths)
        self.ssh(f"rm -rf {joined} && mkdir -p {joined}")

    def tree_hash(self, path: str) -> str:
        """One sha256 over every file under path (symlinks followed, order
        fixed). Used to lock the main arm's JIT cache and the model dir."""
        r = self.ssh(
            f"cd {path} 2>/dev/null && find -L . -type f -print0 | LC_ALL=C sort -z"
            f" | xargs -0 sha256sum | sha256sum | cut -d' ' -f1 || echo MISSING",
            timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"tree_hash failed for {path}: {r.stderr.decode()[-500:]}")
        return r.stdout.decode().strip().splitlines()[-1]


# ---------------------------------------------------------------------------
# Getting the code without running it

def checkout_tree(sha: str, dest: str) -> None:
    """Materialise a commit into dest via git archive — nothing is executed."""
    ar = run(["git", "archive", sha], timeout=120)
    os.makedirs(dest, exist_ok=True)
    run(["tar", "-x", "-C", dest], input_=ar.stdout, timeout=120)


def fetch_arms(pr: int) -> tuple[str, str, str]:
    """(eval_sha, base_sha, mode). Prefer the merge ref: it measures what
    merging actually does. Fall back to head-vs-merge-base when GitHub has no
    merge ref (conflicts)."""
    run(["git", "fetch", "origin", "main"], timeout=120)
    base = run(["git", "rev-parse", "FETCH_HEAD"]).stdout.decode().strip()
    r = run(["git", "fetch", "origin", f"refs/pull/{pr}/merge"],
            timeout=120, check=False)
    if r.returncode == 0:
        merged = run(["git", "rev-parse", "FETCH_HEAD"]).stdout.decode().strip()
        return merged, base, "merge-vs-main"
    run(["git", "fetch", "origin", f"refs/pull/{pr}/head"], timeout=120)
    head = run(["git", "rev-parse", "FETCH_HEAD"]).stdout.decode().strip()
    mb = run(["git", "merge-base", base, head]).stdout.decode().strip()
    return head, mb, "head-vs-merge-base"


def diff_name_status(base_sha: str, eval_sha: str) -> list[tuple[str, str]]:
    """[(status, path)] for the PR's effective change, renames split so a
    rename of a pinned file shows up as a taint-carrying D."""
    out = run(["git", "diff", "--name-status", "--no-renames",
               base_sha, eval_sha], timeout=120).stdout.decode()
    pairs = []
    for ln in out.splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2:
            pairs.append((parts[0].strip(), parts[-1].strip()))
    return pairs


# ---------------------------------------------------------------------------
# GitHub layer

def pr_info(n: int) -> dict:
    return gh_json(["pr", "view", str(n), "--json",
                    "number,state,isDraft,labels,headRefOid,body,files,comments,title"])


def eligible(info: dict) -> tuple[bool, str]:
    labels = [lb["name"] for lb in info.get("labels", [])]
    if info.get("state") != "OPEN":
        return False, "not open"
    if info.get("isDraft"):
        return False, "draft"
    if "hold" in labels:
        return False, "hold label"
    paths = [f["path"] for f in info.get("files", [])]
    if not touches_runtime(paths):
        return False, "no braid/ or tests/ paths"
    if not (ticked_5090(info.get("body")) or "eval" in labels):
        return False, "no 5090 attestation (tick the box, or label `eval` to force)"
    bodies = [c.get("body", "") for c in info.get("comments", [])]
    if already_evaluated(bodies, info["headRefOid"]):
        return False, f"head {info['headRefOid'][:9]} already evaluated"
    return True, "eligible"


def ensure_labels() -> None:
    for name, (color, desc) in EVAL_LABELS.items():
        run(["gh", "label", "create", name, "--color", color,
             "--description", desc, "--force"], check=False)


def post_status(sha: str, state: str, description: str) -> None:
    """A `braid/eval` commit status on the head sha — the thing branch
    protection can require, and the thing a later comment edit cannot
    retroactively change for that sha."""
    run(["gh", "api", f"repos/{{owner}}/{{repo}}/statuses/{sha}",
         "-f", f"state={state}", "-f", f"context={STATUS_CONTEXT}",
         "-f", f"description={description[:140]}"], check=False)


def current_status(sha: str) -> str | None:
    try:
        doc = gh_json(["api", f"repos/{{owner}}/{{repo}}/commits/{sha}/status"])
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return None
    for st in doc.get("statuses", []):
        if st.get("context") == STATUS_CONTEXT:
            return st.get("state")
    return None


def ensure_status(sha: str, state: str, description: str) -> None:
    """Post only on change, so the */30 poll does not spam the statuses API."""
    if current_status(sha) != state:
        post_status(sha, state, description)


def stamp_queue_status(info: dict, ok: bool, why: str) -> None:
    """Give every open head a `braid/eval` status so required-status branch
    protection is livable: runtime PRs show pending until measured, and
    docs-only PRs are not held hostage by a check that will never run."""
    sha = info["headRefOid"]
    if ok:
        ensure_status(sha, "pending", "queued for measured eval on the RTX 5090")
    elif why == "no braid/ or tests/ paths":
        ensure_status(sha, "success", "eval not required — no runtime paths")
    elif why.startswith("no 5090 attestation"):
        ensure_status(sha, "pending", "awaiting the RTX 5090 attestation tick")


def record_evidence(head: str, record: dict) -> None:
    """Append the measured record where a comment edit cannot reach it: a
    git note on the head sha pushed to refs/notes/braid-eval (append-only —
    every edit is a commit on the notes ref), plus a local ledger line."""
    line = json.dumps(record, sort_keys=True)
    try:
        with open(LEDGER, "a") as f:
            f.write(line + "\n")
    except OSError as e:
        print(f"!! ledger append failed: {e}", file=sys.stderr)
    notes_spec = f"+refs/notes/{NOTES_REF}:refs/notes/{NOTES_REF}"
    run(["git", "fetch", "origin", notes_spec], timeout=60, check=False)
    for _ in range(2):
        r = run(["git", "notes", "--ref", NOTES_REF, "append", "-m", line, head],
                check=False)
        if r.returncode != 0:
            break
        if run(["git", "push", "origin", f"refs/notes/{NOTES_REF}"],
               timeout=60, check=False).returncode == 0:
            return
        run(["git", "fetch", "origin", notes_spec], timeout=60, check=False)
    print("!! could not push the eval note — evidence is in the ledger only",
          file=sys.stderr)


def apply_verdict(pr: int, label: str, body: str) -> None:
    ensure_labels()
    for other in EVAL_LABELS:
        if other != label:
            run(["gh", "pr", "edit", str(pr), "--remove-label", other], check=False)
    # Loud on failure: label writes need collaborator rights, and a silent
    # no-op once left a PR wearing a stale verdict while the comment said
    # otherwise. check=False keeps a label hiccup from losing the verdict
    # comment, but it must never be invisible.
    r = run(["gh", "pr", "edit", str(pr), "--add-label", label], check=False)
    if r.returncode != 0:
        print(f"!! could not apply label {label} to #{pr}: "
              f"{r.stderr.decode()[-200:]}", file=sys.stderr)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        gh(["pr", "comment", str(pr), "--body-file", path])
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# The evaluation itself

def bench_cmd(batches: list[int], json_out: bool = True) -> str:
    """The exact measurement a verdict is a function of.

    A contributor who benches a *nearby* configuration — a different arm, a
    different prompt length — and then reads a verdict computed from this one
    has no way to explain the disagreement. So this is the single definition,
    and `make bench-eval` asks the bot to print it (`--print-bench-cmd`)
    rather than keeping a copy that can drift. The only difference between
    the two is `--json`: the bot parses the output, a human reads the table.
    """
    bs = " ".join(str(b) for b in batches)
    return (f"python3 -B -m braid.bench.decode_speed --batches {bs} "
            f"--prompt-len 128 --quant all --state-dtype fp16"
            + (" --json" if json_out else ""))


def run_bench_all(box: Box, arm_dir: str, ext_dir: str,
                  batches: list[int]) -> dict[str, float]:
    """Every arm/batch row the bench reports, for the cross-check reference."""
    env = (f"BRAID_MODEL_DIR={BENCH_MODEL} TORCH_EXTENSIONS_DIR={ext_dir} "
           f"PYTHONPATH={arm_dir} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    r = box.ssh(f"cd {arm_dir} && {env} {bench_cmd(batches)}", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"reference bench failed in {arm_dir}:\n"
                           f"{r.stderr.decode()[-2000:]}")
    return all_arms(r.stdout.decode())


def run_bench(box: Box, arm_dir: str, ext_dir: str, batches: list[int]) -> dict[int, float]:
    env = f"BRAID_MODEL_DIR={BENCH_MODEL} TORCH_EXTENSIONS_DIR={ext_dir} PYTHONPATH={arm_dir}"
    r = box.ssh(f"cd {arm_dir} && {env} {bench_cmd(batches)}", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"bench failed in {arm_dir}:\n{r.stderr.decode()[-2000:]}")
    out = r.stdout.decode()
    return {b: pick_tok_s(out, b) for b in batches}


def free_vram_gib(box: Box) -> float:
    r = box.ssh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
    try:
        return int(r.stdout.decode().strip().splitlines()[-1]) / 1024
    except (ValueError, IndexError):
        return -1.0


def all_arms(bench_stdout: str) -> dict[str, float]:
    """{"arm@batch": tok_per_s} for every row the bench printed.

    The cross-check compares whole reports, not just the serving arm: a
    harness that inflates only the arm nobody looks at is still a harness
    that lies.
    """
    doc = None
    for ln in bench_stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                cand = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(cand, dict) and "arms" in cand:
                doc = cand
    if doc is None:
        raise ValueError("no decode_speed JSON found in bench output")
    return {f"{a.get('name')}@{a.get('batch')}": float(a["tok_per_s"])
            for a in doc["arms"] if a.get("tok_per_s") is not None}


def crosscheck_bench(box: Box, raw_dir: str, ext_dir: str, batches: list[int],
                     pinned: dict[str, float]) -> dict:
    """Run the PR's OWN bench and compare it against the pinned bench.

    Both measure the same engine in the same session; the only difference
    is whose harness did the measuring. Shared rows must agree within the
    noise bar or the harness distorts.
    """
    env = (f"BRAID_MODEL_DIR={BENCH_MODEL} TORCH_EXTENSIONS_DIR={ext_dir} "
           f"PYTHONPATH={raw_dir} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    r = box.ssh(f"cd {raw_dir} && {env} {bench_cmd(batches)}", timeout=1800)
    if r.returncode != 0:
        return {"ran": False, "error": r.stderr.decode()[-500:], "shared": {}, "added": []}
    theirs = all_arms(r.stdout.decode())
    shared, added = {}, []
    for key, val in sorted(theirs.items()):
        if key in pinned:
            base = pinned[key]
            shared[key] = {"pinned": base, "pr_harness": val,
                           "delta_pct": round((val - base) / base * 100, 3)}
        else:
            added.append(key)
    return {"ran": True, "reps": 1, "shared": shared, "added": added}


def run_suite(box: Box, arm_dir: str, ext_dir: str, tests: str) -> tuple[bool, str]:
    # expandable_segments keeps the caching allocator from fragmenting into
    # unusable holes across 300+ tests that build and drop whole engines; the
    # suite peaks at ~26 GiB of 32, so fragmentation alone can decide whether
    # the fp32 gates fit.
    env = (f"TORCH_EXTENSIONS_DIR={ext_dir} PYTHONPATH={arm_dir} "
           f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    r = box.ssh(f"cd {arm_dir} && {env} timeout 3600 python3 -B -m pytest {tests} -q",
                timeout=3720)
    tail = (r.stdout.decode() + r.stderr.decode())[-1500:]
    if r.returncode == 255:
        # ssh's own exit code: the transport failed, so pytest's result is
        # unknown. Unknown is not "the PR is broken".
        raise RuntimeError(
            "INFRASTRUCTURE — lost the ssh transport during the suite; the "
            f"box is gone or unreachable. Not the PR's fault.\n{tail[-600:]}")
    return r.returncode == 0, tail


def check_main_arm(box: Box, main_dir: str, main_ext_hash: str | None) -> None:
    """Everything that must hold before a main rep is trusted. PR code has
    already run in this session; any drift is tampering, not noise."""
    drift = box.verify_tree(main_dir, f"{REMOTE_BASE}/main")
    if drift:
        raise RuntimeError(
            "TAMPER SUSPECTED — the main tree drifted after PR code ran "
            "(repaired, eval aborted):\n" + "\n".join(drift[:20]))
    if main_ext_hash is not None:
        now = box.tree_hash(f"{EXT_BASE}/main")
        if now != main_ext_hash:
            raise RuntimeError(
                "TAMPER SUSPECTED — main's JIT extension cache changed between "
                f"reps ({main_ext_hash[:12]} -> {now[:12]})")
    if not box.gpu_idle_within(60):
        raise RuntimeError(
            "TAMPER SUSPECTED — a GPU process stayed alive for 60s before a "
            "main rep; PR code may have left a daemon behind")


def evaluate(box: Box, pr: int, info: dict, args, model_manifest: str) -> None:
    head = info["headRefOid"]
    eval_sha, base_sha, mode = fetch_arms(pr)
    name_status = diff_name_status(base_sha, eval_sha)
    tainted, unexercised = classify_taint(name_status)
    print(f">> PR #{pr} head={head[:9]}: {mode} — eval {eval_sha[:9]} vs {base_sha[:9]}"
          + (f"; tainted: {tainted}" if tainted else ""))

    with tempfile.TemporaryDirectory(prefix="braid-eval-") as tmp:
        pr_dir, main_dir = os.path.join(tmp, "pr"), os.path.join(tmp, "main")
        checkout_tree(eval_sha, pr_dir)
        checkout_tree(base_sha, main_dir)
        # A bench-only taint gets a chance to clear itself: keep an
        # un-overlaid copy so the PR's own harness can be run beside the
        # pinned one and compared.
        wants_crosscheck = scorer.crosscheckable(tainted)
        raw_dir = os.path.join(tmp, "prraw")
        if wants_crosscheck:
            checkout_tree(eval_sha, raw_dir)
        overlay_harness(pr_dir, main_dir)     # bench/reference/tests come from base
        box.ssh(f"mkdir -p {REMOTE_BASE} {EXT_BASE}")
        box.wipe(f"{EXT_BASE}/pr", f"{EXT_BASE}/main")   # no cache survives across PRs
        box.push_tree(pr_dir, f"{REMOTE_BASE}/pr")
        box.push_tree(main_dir, f"{REMOTE_BASE}/main")
        if wants_crosscheck:
            box.wipe(f"{EXT_BASE}/prraw")
            box.push_tree(raw_dir, f"{REMOTE_BASE}/prraw")

        # The suite needs nearly the whole card. Starting it while anything
        # else holds VRAM is how a healthy tree gets a false failure, so
        # wait for the GPU to drain and refuse to start starved.
        box.gpu_idle_within(120)
        free = free_vram_gib(box)
        print(f">> free VRAM before suite: {free:.1f} GiB")
        if 0 <= free < MIN_FREE_GIB:
            raise RuntimeError(
                f"INFRASTRUCTURE — only {free:.1f} GiB VRAM free before the "
                f"suite (need {MIN_FREE_GIB}); something else holds the card. "
                f"Not the PR's fault; retrying on the next poll.")

        print(f">> suite: pytest {args.tests} (pinned to base) on the PR tree")
        tests_ok, tail = run_suite(box, f"{REMOTE_BASE}/pr", f"{EXT_BASE}/pr", args.tests)
        print(f">> suite: {'ok' if tests_ok else 'FAILED'}")
        if not tests_ok and scorer.suite_infra_failure(tail):
            # Not the PR's fault: the box ran out of VRAM/disk. eval:error
            # retries; eval:reject would be a verdict against the code.
            raise RuntimeError(
                "INFRASTRUCTURE — the suite failed on resource exhaustion "
                "(VRAM/disk), not on the PR. Retrying on the next poll.\n"
                + tail[-800:])

        samples: dict[str, dict[int, list[float]]] = {
            "pr": {b: [] for b in args.batches}, "main": {b: [] for b in args.batches}}
        main_ext_hash: str | None = None
        if tests_ok:
            for rep in range(args.reps):
                order = ["pr", "main"] if rep % 2 == 0 else ["main", "pr"]
                for arm_name in order:
                    if arm_name == "main":
                        check_main_arm(box, main_dir, main_ext_hash)
                    got = run_bench(box, f"{REMOTE_BASE}/{arm_name}",
                                    f"{EXT_BASE}/{arm_name}", args.batches)
                    if arm_name == "main" and main_ext_hash is None:
                        main_ext_hash = box.tree_hash(f"{EXT_BASE}/main")
                    for b, v in got.items():
                        samples[arm_name][b].append(v)
                    print(f">> rep {rep + 1} {arm_name}: "
                          + " ".join(f"B={b} {v:.1f}" for b, v in sorted(got.items())))

        # Escalation: a big claim buys itself more evidence before anyone
        # believes it. Only fires above the major tier, so the ordinary PR
        # pays nothing.
        confirm = None
        if tests_ok and args.confirm_reps > 0:
            first = {a: {b: median(v) for b, v in per.items() if v}
                     for a, per in samples.items()}
            first_deltas = {b: (first["pr"][b] - first["main"][b]) / first["main"][b] * 100
                            for b in args.batches}
            if max(first_deltas.values()) > scorer.TIER_MAJOR:
                print(f">> claim above +{scorer.TIER_MAJOR:.0f}% — escalating "
                      f"{args.confirm_reps} confirmation reps")
                second: dict[str, dict[int, list[float]]] = {
                    "pr": {b: [] for b in args.batches},
                    "main": {b: [] for b in args.batches}}
                for rep in range(args.confirm_reps):
                    for arm_name in (["main", "pr"] if rep % 2 == 0 else ["pr", "main"]):
                        if arm_name == "main":
                            check_main_arm(box, main_dir, main_ext_hash)
                        got = run_bench(box, f"{REMOTE_BASE}/{arm_name}",
                                        f"{EXT_BASE}/{arm_name}", args.batches)
                        for b, v in got.items():
                            second[arm_name][b].append(v)
                            samples[arm_name][b].append(v)
                        print(f">> confirm {rep + 1} {arm_name}: "
                              + " ".join(f"B={b} {v:.1f}" for b, v in sorted(got.items())))
                sec = {a: {b: median(v) for b, v in per.items() if v}
                       for a, per in second.items()}
                sec_deltas = {b: (sec["pr"][b] - sec["main"][b]) / sec["main"][b] * 100
                              for b in args.batches}
                bar = scorer.effective_bar(samples)
                gaps = {b: abs(first_deltas[b] - sec_deltas[b]) for b in args.batches}
                agrees = max(gaps.values()) <= bar
                confirm = {
                    "reps": args.confirm_reps,
                    "first_deltas_pct": {str(b): round(v, 3)
                                         for b, v in first_deltas.items()},
                    "second_deltas_pct": {str(b): round(v, 3)
                                          for b, v in sec_deltas.items()},
                    "agrees": agrees,
                    "detail": (f"rounds differ by at most {max(gaps.values()):.1f}% "
                               f"against a {bar:.1f}% bar"),
                }
                print(f">> confirmation {'AGREES' if agrees else 'DISAGREES'}: "
                      f"{confirm['detail']}")

        crosscheck = None
        if tests_ok and wants_crosscheck:
            # Same engine, same session — only the harness differs. Uses the
            # pinned bench's own last PR-arm report as the reference, so the
            # two numbers describe the same work.
            pinned_ref = run_bench_all(box, f"{REMOTE_BASE}/pr",
                                       f"{EXT_BASE}/pr", args.batches)
            crosscheck = crosscheck_bench(box, f"{REMOTE_BASE}/prraw",
                                          f"{EXT_BASE}/prraw", args.batches, pinned_ref)
            ok, detail = scorer.crosscheck_agrees(crosscheck)
            print(f">> harness cross-check: {'AGREES' if ok else 'DISAGREES'} — {detail}")

        model_now = box.tree_hash(BENCH_MODEL)
        if model_now != model_manifest:
            raise RuntimeError(
                "TAMPER SUSPECTED — the model directory changed during the eval "
                f"({model_manifest[:12]} -> {model_now[:12]})")

    # One canonical bundle; one scorer; two executions. The local call decides
    # the verdict, the TEE call produces the receipt — and they must agree.
    bundle = build_bundle(pr, head, eval_sha, base_sha, mode, args.batches,
                          args.reps, tests_ok, samples, name_status, tail,
                          model_manifest, main_ext_hash, crosscheck, confirm)
    vdoc = scorer.score_bundle(bundle)
    expected_stdout = scorer.canonical(vdoc) + "\n"
    label, reason = vdoc["label"], vdoc["reason"]
    med, deltas = vdoc["medians"], vdoc["deltas_pct"]

    receipt = request_receipt(bundle)
    if receipt is not None and receipt.get("stdout") != expected_stdout:
        raise RuntimeError(
            "TEE-scored verdict differs from the local scorer on the same "
            "bundle — deterministic policy diverged; refusing to post. "
            f"local={expected_stdout!r} tee={receipt.get('stdout')!r}")

    rows = "\n".join(
        f"| {b} | {med['main'][str(b)]:.1f} | {med['pr'][str(b)]:.1f} "
        f"| {deltas[str(b)]:+.1f}% |"
        for b in args.batches) if deltas else ""
    suite_note = "suite passed" if tests_ok else f"suite FAILED — tail:\n```\n{tail}\n```"
    cc_ok = vdoc.get("crosscheck_ok")
    cc_why = (vdoc.get("crosscheck_detail")
              or "test and oracle changes cannot be validated by comparing throughput")
    cc_verdict_line = (
        f">\n> ✅ **Cross-check passed.** {cc_why.capitalize()}. The harness change "
        f"does not distort measurement, so it does not block this PR.\n" if cc_ok else
        f">\n> ⛔ **Not cleared by measurement** — {cc_why}. A human reads this "
        f"diff before it merges.\n")
    taint_note = (
        ("\n> **Harness files modified by this PR** — the eval measured with the "
         "pinned base versions either way:\n"
         + "".join(f"> - `{p}`\n" for p in tainted) + cc_verdict_line)
        if tainted else "")
    cc_table = ""
    if bundle.get("crosscheck", {}) and (bundle["crosscheck"] or {}).get("shared"):
        rows = "\n".join(
            f"| `{k}` | {v['pinned']:.1f} | {v['pr_harness']:.1f} | {v['delta_pct']:+.1f}% |"
            for k, v in sorted(bundle["crosscheck"]["shared"].items()))
        cc_table = ("\n**Harness cross-check** — the PR's own bench beside the pinned "
                    "bench, same engine, same session:\n\n"
                    "| config | pinned tok/s | PR's harness | delta |\n|---|--:|--:|--:|\n"
                    + rows + "\n")
    unex_note = (
        "\n> New files under pinned paths — cannot weaken the gate, but they did "
        "**not** run in this eval; they join the gate once merged:\n"
        + "".join(f"> - `{p}`\n" for p in unexercised) if unexercised else "")
    body = (
        f"{marker(head)}\n"
        f"## Measured on the RTX 5090 — `{label}`\n\n"
        f"Same-session A/B on box {INSTANCE}: `{mode}` "
        f"(`{eval_sha[:9]}` vs `{base_sha[:9]}`), median of {args.reps} reps with arm "
        f"order alternated, bench arm `{ARM}`, Qwen3.5-9B `--quant all "
        f"--state-dtype fp16 --prompt-len 128`. Every number below is "
        f"**measured**; the verdict bar is the same-session ±{NOISE_PCT:.0f}% rule.\n\n"
        f"Harness pinned to base `{base_sha[:9]}`: `braid/bench/`, `braid/reference/` "
        f"and `tests/` were overlaid from main before the PR tree reached the box — "
        f"the PR's engine is measured by main's bench and gated by main's tests. "
        f"Integrity: JIT caches wiped for this eval, main tree checksum-verified and "
        f"its cache hash-locked before every main rep, model manifest verified.\n\n"
        + (f"**Attested:** this verdict was independently recomputed inside an Intel "
           f"TDX machine ({receipt['tee_attestation'].get('kind', 'tdx')}, "
           f"polaris.computer). The DCAP quote binds the scorer "
           f"(`{receipt['tee_attestation']['bound_digest'][:19]}…`), the evidence "
           f"bundle and the verdict; intel_verified="
           f"{receipt.get('verification', {}).get('intel_verified')}. Receipt + "
           f"bundle in this PR's git note — check with `scripts/verify_receipt.py`.\n"
           if receipt is not None else
           "**Attested:** no TDX receipt for this run (attestation service "
           "unavailable); the verdict is the local scorer's alone.\n")
        + taint_note + unex_note + cc_table + "\n"
        + ("| batch | main tok/s | PR tok/s | delta |\n|--:|--:|--:|--:|\n" + rows + "\n\n"
           if rows else "")
        + f"**Verdict:** {reason}. {suite_note}\n\n"
        f"<sub>Automated eval (scripts/pr_eval_bot.py). Outcome policy: a verdict "
        f"ends this PR. `eval:pass`, `eval:noise` and confirmed `eval:major` with "
        f"an attested receipt auto-merge (the exact evaluated head only); "
        f"`eval:reject`, `eval:slower` and `eval:tainted` close it with the "
        f"reason, reversibly. `eval:landmark` waits for a human, and so does "
        f"anything our own infrastructure got wrong. A new push re-queues "
        f"evaluation. Verdict also lands as the `{STATUS_CONTEXT}` commit status "
        f"on `{head[:9]}` and as a git note under `refs/notes/{NOTES_REF}`. "
        f"Box stopped after the run.</sub>"
    )
    apply_verdict(pr, label, body)
    post_status(head, STATUS_STATE[label], f"{label}: {reason}")
    record_evidence(head, {
        "ts": int(time.time()), "label": label, "reason": reason,
        "bundle": bundle, "verdict_doc": vdoc, "receipt": receipt,
    })
    print(f">> PR #{pr}: {label} — {reason}")

    action, why = settle(label, receipt, vdoc.get("stable"),
                         tuple(lb["name"] for lb in info.get("labels", [])))
    if not AUTOMERGE and action != "wait":
        action, why = "wait", f"outcome disabled (BRAID_AUTOMERGE=0); would {action}"

    if action == "merge":
        merged, detail = automerge(pr, head)
        if merged:
            run(["gh", "pr", "comment", str(pr), "--body",
                 f"Auto-merged at `{head[:9]}`: {why}. The receipt for this "
                 f"verdict is permanent in `refs/notes/{NOTES_REF}`."], check=False)
            print(f">> PR #{pr}: auto-merged ({why})")
        else:
            run(["gh", "pr", "comment", str(pr), "--body",
                 f"Auto-merge was earned ({why}) but the merge call failed — "
                 f"maintainer action needed:\n```\n{detail}\n```"], check=False)
            print(f"!! PR #{pr}: auto-merge failed: {detail}", file=sys.stderr)
    elif action == "close":
        closed, detail = close_with_reason(pr, label, why)
        if closed:
            print(f">> PR #{pr}: closed — {why}")
        else:
            run(["gh", "pr", "comment", str(pr), "--body",
                 f"This verdict closes the PR ({why}) but the close call failed — "
                 f"maintainer action needed:\n```\n{detail}\n```"], check=False)
            print(f"!! PR #{pr}: close failed: {detail}", file=sys.stderr)
    else:
        print(f">> PR #{pr}: left open — {why}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pr", type=int, help="evaluate one PR instead of polling")
    p.add_argument("--force", action="store_true",
                   help="with --pr: skip eligibility and idempotency checks")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--confirm-reps", type=int, default=4,
                   help="extra reps run when the first round claims a big gain")
    p.add_argument("--batches", type=int, nargs="+", default=EVAL_BATCHES)
    p.add_argument("--tests", default="tests/")
    p.add_argument("--max-evals", type=int, default=3,
                   help="cost cap: at most this many PRs per cycle")
    p.add_argument("--print-bench-cmd", action="store_true",
                   help="print the bench invocation a verdict is derived from "
                        "and exit (what `make bench-eval` runs)")
    p.add_argument("--print-eval-arm", action="store_true",
                   help="print the decode_speed arm the verdict is read from "
                        "and exit")
    args = p.parse_args()

    # Both are pure lookups of the eval configuration, so they answer before
    # anything with a side effect — `make bench-eval` must not need a gh token.
    if args.print_bench_cmd:
        print(bench_cmd(args.batches, json_out=False))
        return 0
    if args.print_eval_arm:
        print(ARM)
        return 0

    pin_gh_identity()      # before ANY gh call: never post as another account

    if args.pr is not None:
        todo = [args.pr]
    else:
        todo = [pr["number"] for pr in gh_json(["pr", "list", "--json", "number"])]
    plan: list[tuple[int, dict]] = []
    for n in todo:
        info = pr_info(n)
        ok, why = eligible(info)
        if args.force and args.pr is not None:
            ok, why = True, "forced"
        print(f"-- PR #{n}: {why}")
        if not args.dry_run:
            stamp_queue_status(info, ok, why)
            # Docs-only PRs never reach the GPU, so their gate is ordinary
            # CI — no reason to make a human click merge on a typo fix.
            if not ok and why == "no braid/ or tests/ paths" and AUTOMERGE:
                try:
                    checks = gh_json(["pr", "checks", str(n), "--json",
                                      "name,state"]) or []
                except (subprocess.SubprocessError, json.JSONDecodeError):
                    checks = []
                merge_ok, merge_why = docs_only_automerge(info, checks)
                if merge_ok:
                    done, detail = automerge(n, info["headRefOid"])
                    print(f"-- PR #{n}: docs-only auto-merge "
                          f"{'OK' if done else 'FAILED'} — {detail}")
                    if done:
                        gh(["pr", "comment", str(n), "--body",
                            "Auto-merged: docs-only change (no `braid/` or "
                            "`tests/` paths, so no measured evaluation is "
                            "required) with every CI check green.\n\n<sub>"
                            "scripts/pr_eval_bot.py</sub>"])
                else:
                    print(f"-- PR #{n}: docs-only, no auto-merge — {merge_why}")
        if ok:
            plan.append((n, info))
    plan = plan[: args.max_evals]
    if not plan:
        print(">> nothing to evaluate")
        return 0
    if args.dry_run:
        print(f">> dry-run: would evaluate {[n for n, _ in plan]} — stopping here")
        return 0

    box = Box()
    status = box.status()
    if status == "running":
        print(">> box is already running — that is somebody's session, and "
              "touching it mid-work is forbidden. Deferring to the next poll.")
        return 0
    print(f">> box {INSTANCE} is {status}; starting")
    try:
        try:
            box.start()
        except BoxUnavailable as e:
            # Capacity, not a fault. Say so plainly and leave every PR
            # exactly as it was — no eval:error, no verdict, no close. The
            # next poll tries again.
            print(f">> box {INSTANCE} unavailable: {str(e).rstrip('.')}. "
                  f"Nothing evaluated; the next poll retries.")
            return 0
        if box.gpu_busy():
            raise RuntimeError("GPU busy right after start — refusing to eval")
        # The trusted baseline for the whole session, taken before ANY PR
        # code has had a chance to run on the box.
        model_manifest = box.tree_hash(BENCH_MODEL)
        print(f">> model manifest {model_manifest[:12]} ({BENCH_MODEL})")
        for n, info in plan:
            try:
                evaluate(box, n, info, args, model_manifest)
            except Exception as e:                       # noqa: BLE001 — verdict must land
                print(f"!! PR #{n} eval error: {e}", file=sys.stderr)
                # Reporting the failure must not itself be able to abort the
                # cycle: a gh hiccup here once killed the run before the
                # remaining PRs were evaluated, and before the box stopped.
                try:
                    apply_verdict(n, "eval:error",
                                  f"{error_marker(info['headRefOid'])}\n"
                                  f"## Evaluation error — `eval:error`\n\n"
                                  f"```\n{str(e)[-1800:]}\n```\n\n"
                                  f"<sub>Automated eval; retried once on the next "
                                  f"poll. After two errors this head is parked "
                                  f"until a new commit is pushed.</sub>")
                    post_status(info["headRefOid"], "error",
                                f"eval:error: {str(e)[:120]}")
                except Exception as report_err:          # noqa: BLE001
                    print(f"!! could not report the error for #{n}: {report_err}",
                          file=sys.stderr)
    finally:
        box.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
