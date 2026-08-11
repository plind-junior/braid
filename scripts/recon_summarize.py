"""Turn serve_recon.sh's row file into a readable recon table.

Sibling of serve_h2h_summarize.py, with the honesty columns kept and the
median-across-reps machinery removed, because there is only ONE rep. That
absence is the point: this printer stamps every table it emits RECON / NOT
PUBLISHABLE so a screenshot of it cannot be mistaken for the README's curve.

Reads `row RECON <arm> vram_mib=<n> <json>` lines. braid is the reference
column; the delta on each competitor row is braid relative to that competitor,
so a positive number means braid is ahead — the same orientation the published
llama.cpp table uses.
"""
from __future__ import annotations

import json
import sys

REF = "braid"

# Startup decisions serve_recon.sh reads back out of each server's own log.
# These are pulled out of the general meta dump and printed under their own
# banner because they are the EXPLANATION for the table, not trivia about it: a
# row saying braid leads vLLM means something different depending on whether
# vLLM resolved its FlashInfer GDN kernel or fell back to Triton, and on this
# card it falls back (SM90 / capability-family-100 gates, consumer Blackwell is
# 12.0). Buried among twenty meta lines that distinction gets skimmed past.
RECEIPT_TAGS = {"gdn_backend", "mamba_cache_mode"}

# Below this, an arm did not generate the token budget it was asked for, so it
# spent part of the run at reduced concurrency and its tok/s is not comparable
# to an arm that ran full. 0.99 rather than 1.0 because a server's own reported
# count can differ from the client's by a token on the final event.
GEN_FULL = 0.99


def _receipt_key(text: str) -> str:
    """The decision, with any `pid ... INFO ... [file:line]` prefix removed.

    A log prefix carries a pid and a timestamp that differ on every restart, so
    the raw line is unique per server start and useless as a dedup key. The
    sentence after the source location is the part that is actually a claim.
    """
    return text.rsplit("] ", 1)[-1].strip()


def _gen_ratio(d: dict) -> float | None:
    """Fraction of the requested token budget an arm actually generated.

    None for rows written before `tokens_expected` existed — reported as "-"
    rather than as 100%, because "this run predates the check" and "this run
    passed the check" are different claims.
    """
    exp = d.get("tokens_expected")
    if not exp:
        return None
    return d["tokens"] / exp


def _gen_pct_str(d: dict) -> str:
    r = _gen_ratio(d)
    if r is None:
        return "-"
    return f"{r * 100:.0f}%" + ("" if r >= GEN_FULL else "!")


def main() -> None:
    argv = sys.argv[1:]
    # --arms braid,vllm restricts the TABLE, not the file. Rows for excluded
    # arms stay in the row file and are reported below the table as collected
    # -but-not-compared, so narrowing the comparison can never look like the
    # other arm was never run.
    keep: set[str] | None = None
    if "--arms" in argv:
        i = argv.index("--arms")
        keep = set(argv[i + 1].split(","))
        del argv[i:i + 2]
    path = argv[0] if argv else "/root/serve_recon.txt"
    pts: dict[tuple[str, int], dict] = {}
    order: list[str] = []
    meta: list[str] = []
    receipts: list[str] = []
    failed: list[str] = []
    excluded: dict[str, list[dict]] = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("meta") or line.startswith("ABORT"):
                tag = line.split(" ", 2)
                if len(tag) == 3 and tag[1] in RECEIPT_TAGS:
                    # Deduped HERE as well as in serve_recon.sh, because row
                    # files written before that fix carry one copy per server
                    # restart (nine, on the 2026-08-11 v3 run) and those files
                    # still have to render.
                    r = f"{tag[1]}: {_receipt_key(tag[2])}"
                    if r not in receipts:
                        receipts.append(r)
                else:
                    meta.append(line)
                continue
            if "ARM_FAILED" in line:
                failed.append(line)
                continue
            if not line.startswith("row RECON "):
                continue
            _, _, arm, vram, payload = line.split(" ", 4)
            d = json.loads(payload)
            d["vram_mib"] = int(vram.split("=")[1])
            d["arm"] = arm
            if keep is not None and arm not in keep:
                excluded.setdefault(arm, []).append(d)
                continue
            pts[(arm, d["concurrency"])] = d
            if arm not in order:
                order.append(arm)

    for m in meta:
        print(m)
    print("\n*** RECON — 1 rep, NOT PUBLISHABLE. One run is an anecdote; a "
          "published row needs median-of-N across separate processes. ***\n")

    if receipts:
        print("engine decisions, read back from the servers' own logs — these "
              "explain the table, they are not assumed by it:")
        for r in receipts:
            print(f"    {r}")
        print()
    else:
        print("no engine-decision receipts found in this row file — the table "
              "below reports WHAT happened with nothing on WHY.\n")

    concs = sorted({c for (_, c) in pts})
    hdr = (f"{'c':>4} {'arm':<11} {'tok/s':>9} {'vs braid':>9} "
           f"{'TTFT p50':>10} {'ITL p50':>9} {'gen%':>6} {'VRAM MiB':>9} "
           f"{'err':>4} {'cbusy':>6}")
    print(hdr)
    print("-" * len(hdr))
    for c in concs:
        ref = pts.get((REF, c))
        for arm in order:
            d = pts.get((arm, c))
            if d is None:
                print(f"{c:>4} {arm:<11} {'-':>9}")
                continue
            delta = "ref"
            if arm != REF and ref and d["tok_s"]:
                delta = f"{(ref['tok_s'] / d['tok_s'] - 1) * 100:+.1f}%"
            star = "*" if d.get("client_bound") else ""
            print(f"{c:>4} {arm:<11} {d['tok_s']:>9.1f}{star:<1} {delta:>8} "
                  f"{d['ttft_ms_p50']:>9.0f} {d['itl_ms_p50']:>8.2f} "
                  f"{_gen_pct_str(d):>6} "
                  f"{d['vram_mib']:>9} {d['errors']:>4} "
                  f"{d.get('client_busy', 0):>6.2f}")
        print()

    bound = [(a, c) for (a, c), d in pts.items() if d.get("client_bound")]
    errs = [(a, c, d["errors"]) for (a, c), d in pts.items() if d["errors"]]
    short = [(a, c, r) for (a, c), d in pts.items()
             if (r := _gen_ratio(d)) is not None and r < GEN_FULL]
    if short:
        print("! these arms stopped short of the token budget, so they spent "
              "part of the run at reduced concurrency — their tok/s is NOT "
              "comparable to an arm that ran full:")
        for a, c, r in sorted(short):
            d = pts[(a, c)]
            print(f"    {a} c={c}: {d['tokens']}/{d['tokens_expected']} "
                  f"tokens ({r * 100:.0f}%) — is ignore_eos reaching this "
                  f"engine?")
    if bound:
        print("* client-bound — the load generator saturated its core; these "
              "measure the client, not the server:")
        for a, c in sorted(bound):
            print(f"    {a} c={c} (client_busy "
                  f"{pts[(a, c)]['client_busy']:.2f})")
    if errs:
        print("points with request errors:")
        for a, c, n in sorted(errs):
            print(f"    {a} c={c}: {n} errors "
                  f"{pts[(a, c)].get('error_sample')}")
    for line in failed:
        print(line)
    if excluded:
        print("collected but excluded from the comparison "
              "(--arms), still measured:")
        for arm, ds in sorted(excluded.items()):
            pairs = ", ".join(f"c={d['concurrency']}: {d['tok_s']:.1f} tok/s"
                              for d in sorted(ds, key=lambda x: x["concurrency"]))
            print(f"    {arm}: {pairs}")
    if not bound and not errs and not failed and not short:
        print("no client-bound points, no request errors, no failed arms, "
              "every arm generated its full token budget.")
    print("\nNote: vLLM preallocates KV to --gpu-memory-utilization (0.9 by "
          "default), so its VRAM column is a RESERVATION, not a working set, "
          "and is not comparable to braid's.")


if __name__ == "__main__":
    main()
