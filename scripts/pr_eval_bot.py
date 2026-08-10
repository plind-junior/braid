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

Never merges. The verdict is a label and a comment; merging is the
maintainer's.

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
RUNTIME_PREFIXES = ("braid/", "tests/")
STATUS_CONTEXT = "braid/eval"
POLARIS_ATTEST_URL = os.environ.get("BRAID_POLARIS_ATTEST_URL",
                                    "https://polaris.computer/v1/attest")
POLARIS_KEY = os.environ.get("BRAID_POLARIS_API_KEY", "")
SCORER_WORKLOAD = "python3 /in/eval_scorer.py /in/bundle.json"
NOTES_REF = "braid-eval"                  # refs/notes/braid-eval
LEDGER = os.path.expanduser("~/braid-pr-eval-ledger.jsonl")
EVAL_LABELS = {
    "eval:pass": ("0E8A16", "measured speedup beyond the 2% bar"),
    "eval:noise": ("FBCA04", "measured delta within the 2% noise bar"),
    "eval:tainted": ("5319E7", "measured speedup, but the PR touches harness files"),
    "eval:reject": ("B60205", "suite failed or measured regression beyond 2%"),
    "eval:error": ("D93F0B", "evaluation could not complete"),
}
STATUS_STATE = {                          # eval label -> commit-status state
    "eval:pass": "success",
    "eval:noise": "success",
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
    return f"<!-- braid-eval sha={sha} -->"


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
                 model_manifest: str, main_ext_hash: str | None) -> dict:
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
    }


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


def gh(args: list[str], timeout: int = 60) -> str:
    return run(["gh", *args], timeout=timeout).stdout.decode()


def gh_json(args: list[str]) -> object:
    return json.loads(gh(args))


class Box:
    """The rented 5090, held politely: skip if busy, stop if we started it."""

    def __init__(self) -> None:
        self.we_started = False

    def status(self) -> str:
        out = run(["vastai", "show", "instance", INSTANCE, "--raw"]).stdout
        return json.loads(out).get("actual_status", "unknown")

    def ssh(self, cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return run(["ssh", "-i", SSH_KEY, "-p", SSH_PORT,
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=25", SSH_HOST, cmd],
                   timeout=timeout, check=False)

    def start(self) -> None:
        run(["vastai", "start", "instance", INSTANCE])
        self.we_started = True
        for _ in range(40):                       # ~7 min of patience
            time.sleep(10)
            if self.ssh("true", timeout=40).returncode == 0:
                return
        raise RuntimeError("box started but ssh never came up")

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

    def _rsync_ssh(self) -> str:
        return f"ssh -i {SSH_KEY} -p {SSH_PORT} -o StrictHostKeyChecking=accept-new"

    def push_tree(self, local_dir: str, remote_dir: str) -> None:
        ex = [f"--exclude={e}" for e in SYNC_EXCLUDES]
        run(["rsync", "-az", "--delete", *ex, "-e", self._rsync_ssh(),
             f"{local_dir}/", f"{SSH_HOST}:{remote_dir}/"], timeout=300)

    def verify_tree(self, local_dir: str, remote_dir: str) -> list[str]:
        """Checksum-compare the remote tree against the local truth and
        repair it. Returns the itemized drift — non-empty means something on
        the box rewrote the tree since the last push, which after PR code
        has run is tamper evidence, not noise."""
        ex = [f"--exclude={e}" for e in SYNC_EXCLUDES]
        r = run(["rsync", "-azc", "--delete", "--itemize-changes", *ex,
                 "-e", self._rsync_ssh(),
                 f"{local_dir}/", f"{SSH_HOST}:{remote_dir}/"], timeout=600)
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
    run(["gh", "pr", "edit", str(pr), "--add-label", label], check=False)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        gh(["pr", "comment", str(pr), "--body-file", path])
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# The evaluation itself

def bench_cmd(batches: list[int]) -> str:
    bs = " ".join(str(b) for b in batches)
    return (f"python3 -B -m braid.bench.decode_speed --batches {bs} "
            f"--prompt-len 128 --quant all --state-dtype fp16 --json")


def run_bench(box: Box, arm_dir: str, ext_dir: str, batches: list[int]) -> dict[int, float]:
    env = f"BRAID_MODEL_DIR={BENCH_MODEL} TORCH_EXTENSIONS_DIR={ext_dir} PYTHONPATH={arm_dir}"
    r = box.ssh(f"cd {arm_dir} && {env} {bench_cmd(batches)}", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"bench failed in {arm_dir}:\n{r.stderr.decode()[-2000:]}")
    out = r.stdout.decode()
    return {b: pick_tok_s(out, b) for b in batches}


def run_suite(box: Box, arm_dir: str, ext_dir: str, tests: str) -> tuple[bool, str]:
    env = f"TORCH_EXTENSIONS_DIR={ext_dir} PYTHONPATH={arm_dir}"
    r = box.ssh(f"cd {arm_dir} && {env} timeout 3600 python3 -B -m pytest {tests} -q",
                timeout=3720)
    tail = (r.stdout.decode() + r.stderr.decode())[-1500:]
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
    if box.gpu_busy():
        raise RuntimeError(
            "TAMPER SUSPECTED — a GPU process is alive before a main rep; "
            "PR code may have left a daemon behind")


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
        overlay_harness(pr_dir, main_dir)     # bench/reference/tests come from base
        box.ssh(f"mkdir -p {REMOTE_BASE} {EXT_BASE}")
        box.wipe(f"{EXT_BASE}/pr", f"{EXT_BASE}/main")   # no cache survives across PRs
        box.push_tree(pr_dir, f"{REMOTE_BASE}/pr")
        box.push_tree(main_dir, f"{REMOTE_BASE}/main")

        print(f">> suite: pytest {args.tests} (pinned to base) on the PR tree")
        tests_ok, tail = run_suite(box, f"{REMOTE_BASE}/pr", f"{EXT_BASE}/pr", args.tests)
        print(f">> suite: {'ok' if tests_ok else 'FAILED'}")

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

        model_now = box.tree_hash(BENCH_MODEL)
        if model_now != model_manifest:
            raise RuntimeError(
                "TAMPER SUSPECTED — the model directory changed during the eval "
                f"({model_manifest[:12]} -> {model_now[:12]})")

    # One canonical bundle; one scorer; two executions. The local call decides
    # the verdict, the TEE call produces the receipt — and they must agree.
    bundle = build_bundle(pr, head, eval_sha, base_sha, mode, args.batches,
                          args.reps, tests_ok, samples, name_status, tail,
                          model_manifest, main_ext_hash)
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
    taint_note = (
        "\n> **Harness files modified by this PR** (the eval used the pinned base "
        "versions; review these by hand before merging):\n"
        + "".join(f"> - `{p}`\n" for p in tainted) if tainted else "")
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
        + taint_note + unex_note + "\n"
        + ("| batch | main tok/s | PR tok/s | delta |\n|--:|--:|--:|--:|\n" + rows + "\n\n"
           if rows else "")
        + f"**Verdict:** {reason}. {suite_note}\n\n"
        f"<sub>Automated eval (scripts/pr_eval_bot.py). It never merges; a new push "
        f"re-queues evaluation. Verdict also lands as the `{STATUS_CONTEXT}` commit "
        f"status on `{head[:9]}` and as a git note under `refs/notes/{NOTES_REF}`. "
        f"Box stopped after the run.</sub>"
    )
    apply_verdict(pr, label, body)
    post_status(head, STATUS_STATE[label], f"{label}: {reason}")
    record_evidence(head, {
        "ts": int(time.time()), "label": label, "reason": reason,
        "bundle": bundle, "verdict_doc": vdoc, "receipt": receipt,
    })
    print(f">> PR #{pr}: {label} — {reason}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pr", type=int, help="evaluate one PR instead of polling")
    p.add_argument("--force", action="store_true",
                   help="with --pr: skip eligibility and idempotency checks")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--batches", type=int, nargs="+", default=[16, 64])
    p.add_argument("--tests", default="tests/")
    p.add_argument("--max-evals", type=int, default=3,
                   help="cost cap: at most this many PRs per cycle")
    args = p.parse_args()

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
        box.start()
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
                apply_verdict(n, "eval:error",
                              f"{error_marker(info['headRefOid'])}\n"
                              f"## Evaluation error — `eval:error`\n\n"
                              f"```\n{str(e)[-1800:]}\n```\n\n"
                              f"<sub>Automated eval; retried once on the next poll. "
                              f"After two errors this head is parked until a new "
                              f"commit is pushed.</sub>")
                post_status(info["headRefOid"], "error",
                            f"eval:error: {str(e)[:120]}")
    finally:
        box.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
