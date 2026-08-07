#!/usr/bin/env bash
# ROADMAP Phase 4 item 4 — braid vs llama.cpp, both arms in ONE session.
#
# Three things this exists to get right, each of which invalidates the result on
# its own:
#
#   1. **Same session.** The published braid and llama.cpp numbers were taken on
#      different days. Host state on this box drifts 8-15%, so a cross-day delta
#      of that size is not evidence of anything. Both arms run here, minutes
#      apart, with clocks and power sampled during each.
#
#   2. **Same shape.** `llama-batched-bench -npp 128 -ntg 128` decodes at KV
#      128..256. braid's decode bench defaulted to an 8-token prompt, i.e. KV
#      8..200, which is a materially cheaper step -- decode attention reads the
#      whole live KV every step. This passes `--prompt-len 128` so both arms
#      carry the same KV.
#
#   3. **ABBA order.** Arms alternate which goes first on each rep, so a monotone
#      drift in host state (thermal, or another tenant arriving) lands on both
#      arms rather than on whichever ran second.
#
# What is compared is **decode-only aggregate throughput**: llama.cpp's S_TG
# column against braid's `graphed-kvbucket` arm. braid's *served* number from
# serve_bench includes prefill and is NOT comparable to S_TG; mixing them is the
# single easiest way to publish a wrong ratio here.
set -euo pipefail

REPS=${REPS:-5}
NPL=${NPL:-1,2,4,8,16}
BATCHES=${BATCHES:-1 2 4 8 16}
NPP=${NPP:-128}
NTG=${NTG:-128}
LLAMA_DIR=${LLAMA_DIR:-/root/llama.cpp}
GGUF=${GGUF:-/root/models/Qwen3.5-4B-Q8_0.gguf}
BIN="$LLAMA_DIR/build/bin"
# Extra flags for the braid arm, e.g. BRAID_EXTRA=--quant-mlp. The roadmap
# requires both the bf16 and FP8 arms be published, labelled; publishing only
# the slower one understates braid exactly as surely as hiding a losing row
# overstates it.
BRAID_EXTRA=${BRAID_EXTRA:-}

# Nothing else may be on the card. A forgotten server reads ~-12% and explains a
# "regression" for free.
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "ABORT: $busy other process(es) on the GPU" >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
  exit 1
fi

echo "meta llamacpp_build $(cd "$LLAMA_DIR" && git rev-parse --short HEAD)"
echo "meta gguf $GGUF"
echo "meta shape npp=$NPP ntg=$NTG reps=$REPS braid_extra=${BRAID_EXTRA:-none}"

run_llamacpp() {   # $1 = rep
  "$BIN/llama-batched-bench" -m "$GGUF" -c 8192 -b 2048 -ub 512 \
    -npp "$NPP" -ntg "$NTG" -npl "$NPL" -ngl 99 -fa 1 2>/dev/null \
    | awk -v r="$1" -F'|' '/^\|[ ]*[0-9]/ {
        gsub(/ /,"",$4); gsub(/ /,"",$9);
        print "llamacpp", r, $4, $9 }'
  # fields: 4 = B (parallel), 9 = S_TG t/s
}

run_braid() {      # $1 = rep
  python3 -B -m braid.bench.decode_speed --json --prompt-len "$NPP" \
      --steps "$NTG" --batches $BATCHES $BRAID_EXTRA \
    | python3 -c '
import json,sys
d = json.load(sys.stdin)
rep = sys.argv[1]
for a in d["arms"]:
    if a["name"] == "graphed-kvbucket":
        print("braid", rep, a["batch"], round(a["tok_per_s"], 2))
print("health braid", rep, d["health"], file=sys.stderr)
' "$1"
}

for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then order="braid llamacpp"; else order="llamacpp braid"; fi
  for arm in $order; do
    "run_$arm" "$rep"
  done
done
