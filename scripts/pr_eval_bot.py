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
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

INSTANCE = os.environ.get("VAST_INSTANCE", "47055458")
SSH_KEY = os.path.expanduser(os.environ.get("BRAID_SSH_KEY", "~/.ssh/rtx5090"))
SSH_HOST = os.environ.get("BRAID_SSH_HOST", "root@ssh5.vast.ai")
SSH_PORT = os.environ.get("BRAID_SSH_PORT", "15458")
BENCH_MODEL = os.environ.get("BRAID_EVAL_MODEL_DIR", "/root/models/Qwen3.5-9B")
REMOTE_BASE = "/root/braid_eval"          # outside /root/braid: never collides with dev rsync
EXT_BASE = "/root/torch_ext_eval"         # JIT caches, one per arm, never the dev cache
ARM = "graphed-kvbucket"                  # the serving-shaped arm decode_speed reports
NOISE_PCT = 2.0                           # the same-session bar; also the verdict bar
RUNTIME_PREFIXES = ("braid/", "tests/")
EVAL_LABELS = {
    "eval:pass": ("0E8A16", "measured speedup beyond the 2% bar"),
    "eval:noise": ("FBCA04", "measured delta within the 2% noise bar"),
    "eval:reject": ("B60205", "suite failed or measured regression beyond 2%"),
    "eval:error": ("D93F0B", "evaluation could not complete"),
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


def verdict(tests_ok: bool, deltas_pct: dict[int, float],
            noise: float = NOISE_PCT) -> tuple[str, str]:
    """(label, reason). Regression anywhere outranks improvement anywhere."""
    if not tests_ok:
        return "eval:reject", "test suite failed on the merged tree"
    worst = min(deltas_pct.values())
    best = max(deltas_pct.values())
    if worst < -noise:
        b = min(deltas_pct, key=deltas_pct.get)
        return "eval:reject", f"measured regression at B={b}: {deltas_pct[b]:+.1f}%"
    if best > noise:
        b = max(deltas_pct, key=deltas_pct.get)
        return "eval:pass", f"measured speedup at B={b}: {deltas_pct[b]:+.1f}%"
    return "eval:noise", (f"all deltas within the ±{noise:.0f}% same-session bar "
                          f"(best {best:+.1f}%)")


def median(xs: list[float]) -> float:
    return statistics.median(xs)


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

    def push_tree(self, local_dir: str, remote_dir: str) -> None:
        ex = [f"--exclude={e}" for e in SYNC_EXCLUDES]
        run(["rsync", "-az", "--delete", *ex,
             "-e", f"ssh -i {SSH_KEY} -p {SSH_PORT} -o StrictHostKeyChecking=accept-new",
             f"{local_dir}/", f"{SSH_HOST}:{remote_dir}/"], timeout=300)


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


def evaluate(box: Box, pr: int, info: dict, args) -> None:
    head = info["headRefOid"]
    eval_sha, base_sha, mode = fetch_arms(pr)
    print(f">> PR #{pr} head={head[:9]}: {mode} — eval {eval_sha[:9]} vs {base_sha[:9]}")

    with tempfile.TemporaryDirectory(prefix="braid-eval-") as tmp:
        pr_dir, main_dir = os.path.join(tmp, "pr"), os.path.join(tmp, "main")
        checkout_tree(eval_sha, pr_dir)
        checkout_tree(base_sha, main_dir)
        box.ssh(f"mkdir -p {REMOTE_BASE} {EXT_BASE}")
        box.push_tree(pr_dir, f"{REMOTE_BASE}/pr")
        box.push_tree(main_dir, f"{REMOTE_BASE}/main")

    print(f">> suite: pytest {args.tests} on the PR tree")
    tests_ok, tail = run_suite(box, f"{REMOTE_BASE}/pr", f"{EXT_BASE}/pr", args.tests)
    print(f">> suite: {'ok' if tests_ok else 'FAILED'}")

    samples: dict[str, dict[int, list[float]]] = {
        "pr": {b: [] for b in args.batches}, "main": {b: [] for b in args.batches}}
    if tests_ok:
        for rep in range(args.reps):
            order = ["pr", "main"] if rep % 2 == 0 else ["main", "pr"]
            for arm_name in order:
                got = run_bench(box, f"{REMOTE_BASE}/{arm_name}",
                                f"{EXT_BASE}/{arm_name}", args.batches)
                for b, v in got.items():
                    samples[arm_name][b].append(v)
                print(f">> rep {rep + 1} {arm_name}: "
                      + " ".join(f"B={b} {v:.1f}" for b, v in sorted(got.items())))

    med = {a: {b: median(v) for b, v in per.items() if v} for a, per in samples.items()}
    deltas = {b: (med["pr"][b] - med["main"][b]) / med["main"][b] * 100
              for b in args.batches} if tests_ok else {}
    label, reason = verdict(tests_ok, deltas)

    rows = "\n".join(
        f"| {b} | {med['main'][b]:.1f} | {med['pr'][b]:.1f} | {deltas[b]:+.1f}% |"
        for b in args.batches) if deltas else ""
    suite_note = "suite passed" if tests_ok else f"suite FAILED — tail:\n```\n{tail}\n```"
    body = (
        f"{marker(head)}\n"
        f"## Measured on the RTX 5090 — `{label}`\n\n"
        f"Same-session A/B on box {INSTANCE}: `{mode}` "
        f"(`{eval_sha[:9]}` vs `{base_sha[:9]}`), median of {args.reps} reps with arm "
        f"order alternated, bench arm `{ARM}`, Qwen3.5-9B `--quant all "
        f"--state-dtype fp16 --prompt-len 128`. Every number below is "
        f"**measured**; the verdict bar is the same-session ±{NOISE_PCT:.0f}% rule.\n\n"
        + ("| batch | main tok/s | PR tok/s | delta |\n|--:|--:|--:|--:|\n" + rows + "\n\n"
           if rows else "")
        + f"**Verdict:** {reason}. {suite_note}\n\n"
        f"<sub>Automated eval (scripts/pr_eval_bot.py). It never merges; a new push "
        f"re-queues evaluation. Box stopped after the run.</sub>"
    )
    apply_verdict(pr, label, body)
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
        for n, info in plan:
            try:
                evaluate(box, n, info, args)
            except Exception as e:                       # noqa: BLE001 — verdict must land
                print(f"!! PR #{n} eval error: {e}", file=sys.stderr)
                apply_verdict(n, "eval:error",
                              f"{error_marker(info['headRefOid'])}\n"
                              f"## Evaluation error — `eval:error`\n\n"
                              f"```\n{str(e)[-1800:]}\n```\n\n"
                              f"<sub>Automated eval; retried once on the next poll. "
                              f"After two errors this head is parked until a new "
                              f"commit is pushed.</sub>")
    finally:
        box.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
