#!/usr/bin/env python3
"""braid eval scorer — the verdict policy as a pure, TEE-runnable program.

This file is the single source of truth for how a measured evaluation
becomes a verdict. `scripts/pr_eval_bot.py` imports its policy from here,
and the same file is shipped byte-for-byte into a polaris.computer Intel
TDX machine per evaluation (`/v1/attest`, egress=none), where it re-derives
the verdict from the canonical evidence bundle. The receipt's `files_sha256`
binds this file's hash and the bundle's hash into the TD quote's
report_data, and `result_sha256` binds the verdict printed on stdout — so a
third party can check that THIS policy, at the committed revision, produced
THAT verdict from THOSE measurements, without trusting the maintainer's
machine. `scripts/verify_receipt.py` performs the recomputation.

Rules for editing: stdlib only, deterministic (no clocks, no randomness, no
network — egress is sealed anyway), and every printed byte canonical
(sorted keys, fixed separators). Any behavior change here is a policy
change and lands with tests in tests/test_pr_eval_bot.py.

Usage:  python3 eval_scorer.py bundle.json    # or '-' for stdin
Prints the canonical verdict JSON on stdout; exits non-zero on a malformed
bundle.
"""
from __future__ import annotations

import json
import statistics
import sys

SCHEMA_IN = "braid-eval-bundle/1"
SCHEMA_OUT = "braid-eval-verdict/1"
NOISE_PCT = 2.0                           # the same-session bar; also the verdict bar
PINNED_DIRS = ("braid/bench", "braid/reference", "tests")  # harness: taken from base
INFRA_FILES = ("scripts/pr_eval_bot.py", "scripts/pr_eval_cron.sh",
               "scripts/eval_scorer.py")
INFRA_PREFIXES = (".github/",)


INFRA_SIGNATURES = (
    "outofmemoryerror",           # torch.OutOfMemoryError
    "cuda out of memory",
    "cuda error: out of memory",
    "no space left on device",
    "fits in",                    # conftest's VRAM budget refusing to build a stack
    "unable to load the kernel",  # JIT/driver trouble, not the PR's logic
    "no route to host",           # the rented box vanished mid-suite
    "connection closed by remote host",
    "connection timed out",
    "connection refused",
    "broken pipe",
)


def suite_infra_failure(tail: str) -> bool:
    """Did the suite fail because the BOX ran out of something?

    A gate that cannot tell 'your code is broken' from 'the box ran out of
    VRAM' will eventually blame a contributor for an infrastructure
    failure — it did, on PRs #5 and #6, whose trees were fine (main's own
    suite passes 311/311 on the same box, peaking at 26 GiB of 32). Those
    runs must become eval:error, which retries, not eval:reject, which is a
    verdict against the code.

    Deliberately one-directional: a real assertion failure never matches
    these, and an OOM that also hides a real failure gets retried and
    caught on the next run.
    """
    low = tail.lower()
    return any(sig in low for sig in INFRA_SIGNATURES)


def median(xs: list[float]) -> float:
    return statistics.median(xs)


def _pinned(path: str) -> bool:
    return any(path == d or path.startswith(d + "/") for d in PINNED_DIRS)


def classify_taint(name_status) -> tuple[list[str], list[str]]:
    """Split a base..PR name-status diff into (tainted, unexercised).

    tainted: harness files the PR modifies or deletes, plus any touch at all
    of the eval infrastructure (this file included). The eval still runs —
    with the pinned harness, so the numbers stay honest — but the best
    reachable verdict is eval:tainted; the harness diff needs human eyes.

    unexercised: files ADDED under pinned dirs. An addition cannot weaken
    the pinned gate, so it does not taint — but the overlay means it never
    runs during the eval either (a new parity test joins the gate once
    merged).
    """
    tainted, unexercised = [], []
    for status, path in (tuple(p) for p in name_status):
        if path in INFRA_FILES or path.startswith(INFRA_PREFIXES):
            tainted.append(path)
        elif _pinned(path):
            (unexercised if status == "A" else tainted).append(path)
    return tainted, unexercised


def verdict(tests_ok: bool, deltas_pct: dict[int, float],
            tainted=(), noise: float = NOISE_PCT) -> tuple[str, str]:
    """(label, reason). Regression anywhere outranks improvement anywhere;
    a taint caps a would-be pass at eval:tainted (the measurement used the
    pinned harness and is trustworthy — the PR's harness diff is not)."""
    if not tests_ok:
        return "eval:reject", "pinned test suite failed on the merged tree"
    worst = min(deltas_pct.values())
    best = max(deltas_pct.values())
    if worst < -noise:
        b = min(deltas_pct, key=deltas_pct.get)
        return "eval:reject", f"measured regression at B={b}: {deltas_pct[b]:+.1f}%"
    if best > noise:
        b = max(deltas_pct, key=deltas_pct.get)
        if tainted:
            return "eval:tainted", (
                f"measured speedup at B={b}: {deltas_pct[b]:+.1f}% — but the PR "
                f"modifies harness files; review them by hand before merging")
        return "eval:pass", f"measured speedup at B={b}: {deltas_pct[b]:+.1f}%"
    return "eval:noise", (f"all deltas within the ±{noise:.0f}% same-session bar "
                          f"(best {best:+.1f}%)")


def canonical(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def score_bundle(bundle: dict) -> dict:
    """Canonical bundle in, verdict document out. Deterministic; the bot and
    the TEE both call exactly this, so their outputs must be byte-identical."""
    if bundle.get("schema") != SCHEMA_IN:
        raise ValueError(f"unknown bundle schema: {bundle.get('schema')!r}")
    batches = [int(b) for b in bundle["batches"]]
    tests_ok = bool(bundle["tests_ok"])
    tainted, unexercised = classify_taint(bundle.get("name_status", []))

    medians: dict[str, dict[str, float]] = {}
    deltas: dict[int, float] = {}
    if tests_ok:
        if not batches:
            raise ValueError("tests passed but no batches to score")
        m = {arm: {b: median([float(v) for v in bundle["samples"][arm][str(b)]])
                   for b in batches} for arm in ("pr", "main")}
        deltas = {b: (m["pr"][b] - m["main"][b]) / m["main"][b] * 100 for b in batches}
        medians = {arm: {str(b): round(v, 3) for b, v in per.items()}
                   for arm, per in m.items()}
    label, reason = verdict(tests_ok, deltas, tainted)

    return {
        "schema": SCHEMA_OUT,
        "pr": bundle["pr"], "head": bundle["head"],
        "eval_sha": bundle["eval_sha"], "base_sha": bundle["base_sha"],
        "label": label, "reason": reason, "noise_pct": NOISE_PCT,
        "tests_ok": tests_ok,
        "medians": medians,
        "deltas_pct": {str(b): round(v, 3) for b, v in sorted(deltas.items())},
        "tainted": tainted, "unexercised": unexercised,
    }


def main() -> int:
    src = sys.stdin if len(sys.argv) < 2 or sys.argv[1] == "-" else open(sys.argv[1])
    with src:
        bundle = json.load(src)
    print(canonical(score_bundle(bundle)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
