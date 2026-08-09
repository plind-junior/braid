"""Aggregate the rows `head_to_head.sh` emits into the table that gets published.

Separate from the shell so the reduction is reviewable and so a published table
can be regenerated from the raw rows without re-running the box.

Input is one row per (arm, rep, batch):

    <arm> <rep> <batch> <tok/s> [<peak GiB>]

**The median, not the mean, and the spread printed beside it.** One run is an
anecdote; a mean over five hides the one that was 8% low because another tenant
arrived. If the spread is not small the table is not publishable, and the only
way for a reader to know that is for it to be on the page.
"""
from __future__ import annotations

import statistics as st
import sys
from collections import defaultdict


def main(path: str, base: str = "llamacpp") -> None:
    tps: dict[tuple[str, int], list[float]] = defaultdict(list)
    peak: dict[tuple[str, int], list[float]] = defaultdict(list)
    order: list[str] = []
    for line in open(path):
        f = line.split()
        if len(f) < 4 or not f[2].isdigit():
            continue
        arm, b = f[0], int(f[2])
        tps[(arm, b)].append(float(f[3]))
        if len(f) > 4:
            peak[(arm, b)].append(float(f[4]))
        if arm not in order:
            order.append(arm)

    if not tps:
        raise SystemExit(f"no rows parsed from {path}")
    batches = sorted({b for _, b in tps})
    others = [a for a in order if a != base]

    head = f"{'B':>4} {base:>11}"
    for a in others:
        head += f" {a:>11} {'vs':>8}"
    print(head)
    print("-" * len(head))
    for b in batches:
        ref = st.median(tps[(base, b)]) if (base, b) in tps else float("nan")
        line = f"{b:>4} {ref:>11.1f}"
        for a in others:
            v = tps.get((a, b))
            if not v:
                line += f" {'-':>11} {'-':>8}"
                continue
            m = st.median(v)
            line += f" {m:>11.1f} {(m / ref - 1) * 100:>+7.1f}%"
        print(line)

    print("\nspread (max-min)/median, worst over batches:")
    for a in order:
        pts = [(k, v) for k, v in tps.items() if k[0] == a]
        worst = max((max(v) - min(v)) / st.median(v) * 100 for _, v in pts)
        n = min(len(v) for _, v in pts)
        print(f"  {a:<11} {worst:>5.2f}%   ({n} processes per point)")

    if peak:
        print("\nbraid peak VRAM, GiB:")
        for a in others:
            row = "  ".join(f"B={b}:{max(peak[(a, b)]):.1f}"
                            for b in batches if (a, b) in peak)
            if row:
                print(f"  {a:<11} {row}")

    print(f"\nscaling B={batches[0]} -> B={batches[-1]}:")
    for a in order:
        lo, hi = (a, batches[0]), (a, batches[-1])
        if lo in tps and hi in tps:
            print(f"  {a:<11} {st.median(tps[hi]) / st.median(tps[lo]):>5.1f}x")


if __name__ == "__main__":
    main(*sys.argv[1:])
