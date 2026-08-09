"""Turn serve_h2h.sh's row file into the publishable server-vs-server table.

Reads `row <rep> <json>` lines, takes medians across reps per (server,
concurrency), prints the comparison plus every honesty column the client
records: error counts, and — load-bearing — the `client_bound` flag. A point
where the single-threaded client saturated its core is printed with its
numbers STRUCK to `(client-bound)` rather than silently dropped or silently
published; either of those would misstate what was measured.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/serve_h2h.txt"
    points = defaultdict(list)      # (server, c) -> [point dict]
    meta = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("meta"):
                meta.append(line)
                continue
            if not line.startswith("row "):
                continue
            _, rep, payload = line.split(" ", 2)
            d = json.loads(payload)
            d["rep"] = int(rep)
            points[(d["server"], d["concurrency"])].append(d)

    for m in meta:
        print(m)

    def med(ps, key):
        return statistics.median(p[key] for p in ps)

    def spread(ps, key):
        xs = sorted(p[key] for p in ps)
        m = statistics.median(xs)
        return (xs[-1] - xs[0]) / m * 100 if m else 0.0

    concs = sorted({c for (_, c) in points})
    print(f"\n{'c':>4} {'llama tok/s':>12} {'braid tok/s':>12} {'delta':>8} "
          f"{'llama TTFT p50':>15} {'braid TTFT p50':>15} "
          f"{'llama ITL p50':>14} {'braid ITL p50':>14}")
    worst = defaultdict(float)
    for c in concs:
        row = {}
        for server in ("llama", "braid"):
            ps = points.get((server, c), [])
            if not ps:
                row[server] = None
                continue
            errs = sum(p["errors"] for p in ps)
            bound = any(p.get("client_bound") for p in ps)
            row[server] = {
                "tok_s": med(ps, "tok_s"),
                "ttft": med(ps, "ttft_ms_p50"),
                "itl": med(ps, "itl_ms_p50"),
                "reps": len(ps), "errors": errs, "bound": bound,
                "busy": max(p.get("client_busy", 0.0) for p in ps),
            }
            worst[server] = max(worst[server], spread(ps, "tok_s"))

        def cell(s, key, fmt):
            if row[s] is None:
                return f"{'-':>1}"
            v = fmt.format(row[s][key])
            return v + ("*" if row[s]["bound"] else "")

        delta = "-"
        if row["llama"] and row["braid"] and row["llama"]["tok_s"]:
            delta = f"{(row['braid']['tok_s'] / row['llama']['tok_s'] - 1) * 100:+.1f}%"
        print(f"{c:>4} {cell('llama', 'tok_s', '{:.1f}'):>12} "
              f"{cell('braid', 'tok_s', '{:.1f}'):>12} {delta:>8} "
              f"{cell('llama', 'ttft', '{:.0f} ms'):>15} "
              f"{cell('braid', 'ttft', '{:.0f} ms'):>15} "
              f"{cell('llama', 'itl', '{:.2f} ms'):>14} "
              f"{cell('braid', 'itl', '{:.2f} ms'):>14}")

    print("\nspread (max-min)/median of tok/s, worst over points:")
    for server in ("llama", "braid"):
        print(f"  {server:<6} {worst[server]:.2f}%")

    flagged = [(s, c) for (s, c), ps in points.items()
               if any(p.get("client_bound") for p in ps)]
    errored = [(s, c, sum(p['errors'] for p in ps))
               for (s, c), ps in points.items() if any(p["errors"] for p in ps)]
    if flagged:
        print("\n* client-bound points (the client saturated its core; these "
              "measure the client, not the server):")
        for s, c in sorted(flagged):
            busy = max(p.get("client_busy", 0) for p in points[(s, c)])
            print(f"  {s} c={c} (client_busy {busy:.2f})")
    if errored:
        print("\npoints with request errors:")
        for s, c, n in sorted(errored):
            print(f"  {s} c={c}: {n} errors")
    if not flagged and not errored:
        print("\nno client-bound points, no request errors — every number is "
              "a server measurement.")


if __name__ == "__main__":
    main()
