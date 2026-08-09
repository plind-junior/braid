"""Turn one `decode_speed --json` run into head-to-head rows on stdout.

    <label> <rep> <batch> <tok/s> <peak GiB>

Only the `graphed-kvbucket` arm, because that is the configuration braid would
actually serve in and the only one comparable to llama.cpp's S_TG. Health, the
quantized groups and the decode-step weight total go to **stderr**, so they are
visible while the run streams without landing in the file the aggregator reads.
"""
from __future__ import annotations

import json
import sys


def main(rep: str, label: str) -> None:
    d = json.load(sys.stdin)
    for a in d["arms"]:
        if a["name"] == "graphed-kvbucket":
            print(label, rep, a["batch"], round(a["tok_per_s"], 2),
                  round(a.get("peak_gib", 0.0), 2))
    step = sum(d.get("step_bytes", {}).values()) / 2 ** 30
    print(f"health {label} rep{rep} {d['health']} | decode-step weights "
          f"{step:.2f} GiB | fp8 {d.get('quant') or ['none']}", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:])
