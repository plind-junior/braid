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


CROSSCHECKABLE = ("braid/bench",)   # numbers can be compared; oracles and tests cannot


def crosscheckable(tainted) -> bool:
    """Is every tainted path one whose honesty a measurement can settle?

    A modified bench can be cross-checked: run it beside the pinned bench
    and see whether it reports the same throughput for the same work. A
    modified test or reference oracle cannot — weakening a parity
    assertion does not change tok/s, so no number exposes it. Those stay
    with a human no matter how well the benches agree.
    """
    if not tainted:
        return False
    return all(any(p == d or p.startswith(d + "/") for d in CROSSCHECKABLE)
               for p in tainted)


def crosscheck_agrees(crosscheck, noise: float = NOISE_PCT) -> tuple[bool, str]:
    """Does the PR's own bench report what the pinned bench reports?

    The PR's harness and the pinned harness measure the same engine in the
    same session, so on any configuration they share they must agree within
    the same bar the verdict uses. A bench rigged to inflate (the red-team
    case multiplied tok/s by 1.5) disagrees by 50% and is caught here.

    Limits, stated so nobody mistakes this for proof: configurations the PR
    ADDS have no pinned counterpart and are not validated, and a harness
    that distorts only under some condition it can detect would pass. This
    is evidence, not a proof of honesty.
    """
    if not crosscheck:
        return False, "no cross-check was run"
    shared = crosscheck.get("shared") or {}
    if not shared:
        return False, "the PR's bench shares no configuration with the pinned bench"
    worst_key, worst = None, 0.0
    for key, row in shared.items():
        d = abs(float(row["delta_pct"]))
        if d > worst:
            worst_key, worst = key, d
    if worst > noise:
        return False, (f"the PR's own bench disagrees with the pinned bench by "
                       f"{worst:.1f}% at {worst_key} (bar ±{noise:.0f}%)")
    added = crosscheck.get("added") or []
    note = (f"; {len(added)} added configuration(s) have no pinned counterpart "
            f"and were not validated" if added else "")
    return True, (f"the PR's own bench agrees with the pinned bench within "
                  f"±{noise:.0f}% on {len(shared)} shared configuration(s) "
                  f"(worst {worst:.1f}%){note}")


TIER_MAJOR = 10.0        # above this a gain is big enough to want confirming
TIER_LANDMARK = 25.0     # above this it is extraordinary and a human reads it
SLOWER_FLOOR = -5.0      # between -bar and this, a regression may be a fair trade


def observed_spread_pct(samples) -> float:
    """Worst relative spread across every arm/batch series in this session.

    The empirical noise of THIS run, not a remembered constant. Three reps
    of the same tree typically land within 0.04% at B=16 and 0.2% at B=64;
    a session that scatters much wider is telling you its own numbers are
    soft, and the bar should rise to match.
    """
    worst = 0.0
    for per_batch in (samples or {}).values():
        for values in (per_batch or {}).values():
            vals = [float(v) for v in values if v is not None]
            if len(vals) < 2:
                continue
            mid = median(vals)
            if mid:
                worst = max(worst, (max(vals) - min(vals)) / mid * 100)
    return worst


def effective_bar(samples, noise: float = NOISE_PCT) -> float:
    """The bar this session earns. Only ever tighter than the published
    rule: 2% stays the floor, and a jittery session raises it. Loosening
    below 2% is a separate argument that needs a noise-floor study, not a
    formula tucked into the scorer."""
    return max(noise, 3.0 * observed_spread_pct(samples))


def shape(deltas_pct: dict, bar: float) -> str:
    """How the gain is distributed across batch sizes.

    A uniform +8% and '+16% at B=64, flat at B=16' are different
    engineering claims: the second is batching efficiency, which does
    nothing for a single-stream user. Reported, never folded into the
    label — the ladder has to stay readable at a glance.
    """
    if not deltas_pct:
        return "n/a"
    items = sorted((int(b), v) for b, v in deltas_pct.items())
    vals = [v for _, v in items]
    best, worst = max(vals), min(vals)
    if best > bar and worst < -bar:
        return "mixed"
    if abs(best) <= bar:
        return "flat"
    if best - worst <= abs(best) / 3.0:
        return "uniform"
    peak = max(items, key=lambda kv: kv[1])[0]
    return "batch-skewed" if peak == max(b for b, _ in items) else "latency-skewed"


def verdict(tests_ok: bool, deltas_pct: dict[int, float],
            tainted=(), noise: float = NOISE_PCT) -> tuple[str, str]:
    """(label, reason) on the graded ladder.

    Order of precedence, strongest first: a failed suite or a material
    regression outranks every gain; a modest regression is its own tier
    because a correctness fix that costs 3% may be exactly the right trade
    and should not look like a catastrophe; an uncleared harness taint
    blocks any non-reject outcome; and above that the gain tiers rise.

    The top tier deliberately does NOT auto-merge. On an engine that has
    already been profiled hard, +25% is likelier to be work that quietly
    stopped happening than a breakthrough — and the suite only catches
    that where a test covers it. Evidence demanded scales with how
    surprising the number is.
    """
    if not tests_ok:
        return "eval:reject", "pinned test suite failed on the merged tree"
    worst = min(deltas_pct.values())
    best = max(deltas_pct.values())
    if worst < SLOWER_FLOOR:
        b = min(deltas_pct, key=deltas_pct.get)
        return "eval:reject", f"measured regression at B={b}: {deltas_pct[b]:+.1f}%"
    if worst < -noise:
        b = min(deltas_pct, key=deltas_pct.get)
        if tainted:
            return "eval:tainted", (
                f"measured {deltas_pct[b]:+.1f}% at B={b} and the PR modifies "
                f"harness files ({', '.join(tainted)})")
        return "eval:slower", (
            f"measured regression at B={b}: {deltas_pct[b]:+.1f}% — inside the "
            f"{SLOWER_FLOOR:.0f}% band, so this may be a fair trade for "
            f"correctness; a human decides")
    if tainted:
        # Taint blocks at every non-reject outcome, not just at a speedup.
        # A PR that rewrites the bench and happens to measure flat is still
        # a PR that rewrites the bench: the pinned harness keeps the NUMBER
        # honest, it does not make the harness diff safe to merge. Capping
        # only the pass path left the red-team PR (tok_per_s * 1.5, measured
        # noise) wearing a green, mergeable status.
        where = (f"best {best:+.1f}%" if best <= noise
                 else f"speedup at B={max(deltas_pct, key=deltas_pct.get)}: {best:+.1f}%")
        return "eval:tainted", (
            f"measured {where}, but the PR modifies harness files "
            f"({', '.join(tainted)}); review them by hand before merging")
    if best > noise:
        b = max(deltas_pct, key=deltas_pct.get)
        if best > TIER_LANDMARK:
            return "eval:landmark", (
                f"measured speedup at B={b}: {deltas_pct[b]:+.1f}% — beyond "
                f"{TIER_LANDMARK:.0f}%, which on this engine wants a person to "
                f"confirm nothing stopped happening; no auto-merge")
        if best > TIER_MAJOR:
            return "eval:major", (
                f"measured speedup at B={b}: {deltas_pct[b]:+.1f}% — beyond "
                f"{TIER_MAJOR:.0f}%, confirmed with an escalated rep count")
        return "eval:pass", f"measured speedup at B={b}: {deltas_pct[b]:+.1f}%"
    return "eval:noise", (f"all deltas within the ±{noise:.1f}% same-session bar "
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

    # A bench-only taint can be cleared by measurement: if the PR's own
    # harness reports what the pinned harness reports, the harness change
    # does not distort, and the PR rejoins the ordinary pass/noise ladder.
    # Test and oracle taints are never clearable this way.
    cc_ok, cc_detail = (crosscheck_agrees(bundle.get("crosscheck"))
                        if crosscheckable(tainted) else (False, ""))
    effective_taint = [] if cc_ok else tainted

    bar = effective_bar(bundle.get("samples"))
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
    label, reason = verdict(tests_ok, deltas, effective_taint, bar)
    if cc_ok and tests_ok:
        reason += f" — harness change cleared: {cc_detail}"

    # A tier above eval:pass is only believed if a second round of reps
    # agreed with the first; the bot escalates and records that here.
    conf = bundle.get("confirm") or {}
    stable = bool(conf.get("agrees")) if conf else None
    if conf and not stable:
        reason += (f" — but the confirmation round disagreed with the first "
                   f"({conf.get('detail', 'unstable')}); no auto-merge")

    return {
        "schema": SCHEMA_OUT,
        "pr": bundle["pr"], "head": bundle["head"],
        "eval_sha": bundle["eval_sha"], "base_sha": bundle["base_sha"],
        "label": label, "reason": reason,
        "noise_pct": NOISE_PCT, "bar_pct": round(bar, 3),
        "spread_pct": round(observed_spread_pct(bundle.get("samples")), 4),
        "shape": shape({str(b): v for b, v in deltas.items()}, bar),
        "tests_ok": tests_ok,
        "medians": medians,
        "deltas_pct": {str(b): round(v, 3) for b, v in sorted(deltas.items())},
        "tainted": tainted, "unexercised": unexercised,
        "crosscheck_ok": cc_ok, "crosscheck_detail": cc_detail,
        "stable": stable,
    }


def main() -> int:
    src = sys.stdin if len(sys.argv) < 2 or sys.argv[1] == "-" else open(sys.argv[1])
    with src:
        bundle = json.load(src)
    print(canonical(score_bundle(bundle)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
