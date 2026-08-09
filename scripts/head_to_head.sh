#!/usr/bin/env bash
# ROADMAP Phase 4 item 4 — braid vs llama.cpp, every arm in ONE session.
#
# Four things this exists to get right, each of which invalidates the result on
# its own:
#
#   1. **Same session.** The published braid and llama.cpp numbers were once
#      taken on different days. Host state on this box drifts 8-15%, so a
#      cross-day delta of that size is not evidence of anything. Every arm runs
#      here, minutes apart, with clocks and power sampled during each.
#
#      That includes braid's *own* arms. bf16 and fp8 used to be separate
#      invocations, which quietly compared numbers taken an hour apart — the
#      same mistake one level down. `BRAID_ARMS` runs them all in this session.
#
#   2. **Same shape.** `llama-batched-bench -npp 128 -ntg 128` decodes at KV
#      128..256. braid's decode bench defaulted to an 8-token prompt, i.e. KV
#      8..200, which is a materially cheaper step -- decode attention reads the
#      whole live KV every step. This passes `--prompt-len 128` so both arms
#      carry the same KV.
#
#   3. **Rotated order.** The arm that goes first rotates by rep, so a monotone
#      drift in host state (thermal, or another tenant arriving) is spread over
#      every arm rather than landing on whichever ran second. With two arms this
#      is ABBA; with four it is the same idea one step further.
#
#   4. **Context sized for the widest point.** `llama-batched-bench` skips any
#      row needing more KV than `-c` allows, silently. See CTX below.
#
# What is compared is **decode-only aggregate throughput**: llama.cpp's S_TG
# column against braid's `graphed-kvbucket` arm. braid's *served* number from
# serve_bench includes prefill and is NOT comparable to S_TG; mixing them is the
# single easiest way to publish a wrong ratio here.
#
#     REPS=5 NPL=1,2,4,8,16,32,64 BATCHES="1 2 4 8 16 32 64" \
#       GGUF=/root/models/Qwen3.5-9B-Q8_0.gguf \
#       BRAID_MODEL_DIR=/root/models/Qwen3.5-9B bash scripts/head_to_head.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REPS=${REPS:-5}
NPL=${NPL:-1,2,4,8,16}
BATCHES=${BATCHES:-1 2 4 8 16}
NPP=${NPP:-128}
NTG=${NTG:-128}
LLAMA_DIR=${LLAMA_DIR:-/root/llama.cpp}
GGUF=${GGUF:-/root/models/Qwen3.5-4B-Q8_0.gguf}
BIN="$LLAMA_DIR/build/bin"

# braid arms as `label:flags`, separated by `|`. The roadmap requires every arm
# be published, labelled: publishing only the fastest understates what it cost
# to get there exactly as surely as hiding a losing row overstates the win.
BRAID_ARMS=${BRAID_ARMS:-"bf16:|fp8-mlp:--quant mlp|fp8-all:--quant all"}

# Raw rows land outside the synced tree -- `scripts/remote.sh` rsyncs with
# --delete before every command, so anything written under the repo is destroyed
# on the next invocation.
ROWS=${ROWS:-/root/h2h_rows.txt}

# `llama-batched-bench` needs `-c` to cover `max(NPL) * (NPP + NTG)` and
# **skips** any row it cannot fit rather than failing. Left at a constant, the
# widest batches would simply be absent and the table would still look complete.
# Derived, so growing the sweep cannot silently drop its top end.
CTX=${CTX:-$(python3 -c "
import sys
npl = [int(x) for x in sys.argv[1].split(',')]
print(max(npl) * (int(sys.argv[2]) + int(sys.argv[3])))" "$NPL" "$NPP" "$NTG")}

# Nothing else may be on the card. A forgotten server reads ~-12% and explains a
# "regression" for free.
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "ABORT: $busy other process(es) on the GPU" >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
  exit 1
fi

echo "meta llamacpp_build $(cd "$LLAMA_DIR" && git rev-parse --short HEAD)"
echo "meta braid_commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "meta gguf $GGUF"
echo "meta braid_model ${BRAID_MODEL_DIR:-/root/models/Qwen3.5-4B}"
echo "meta shape npp=$NPP ntg=$NTG ctx=$CTX reps=$REPS"
echo "meta braid_arms $BRAID_ARMS"

run_llamacpp() {   # $1 = rep
  "$BIN/llama-batched-bench" -m "$GGUF" -c "$CTX" -b 2048 -ub 512 \
    -npp "$NPP" -ntg "$NTG" -npl "$NPL" -ngl 99 -fa 1 2>/dev/null \
    | awk -v r="$1" -F'|' '/^\|[ ]*[0-9]/ {
        gsub(/ /,"",$4); gsub(/ /,"",$9);
        print "llamacpp", r, $4, $9 }'
  # fields: 4 = B (parallel), 9 = S_TG t/s
}

run_braid() {      # $1 = rep, $2 = label, $3.. = flags
  local rep="$1" label="$2"
  shift 2
  python3 -B -m braid.bench.decode_speed --json --prompt-len "$NPP" \
      --steps "$NTG" --batches $BATCHES "$@" \
    | python3 scripts/h2h_row.py "$rep" "$label"
}

IFS='|' read -r -a ARMS <<< "$BRAID_ARMS"
PLAN=("llamacpp")
for a in "${ARMS[@]}"; do PLAN+=("braid:$a"); done
n=${#PLAN[@]}

: > "$ROWS"
for rep in $(seq 1 "$REPS"); do
  for i in $(seq 0 $((n - 1))); do
    entry="${PLAN[$(( (i + rep - 1) % n ))]}"
    if [ "$entry" = "llamacpp" ]; then
      run_llamacpp "$rep" | tee -a "$ROWS"
    else
      spec="${entry#braid:}"
      # shellcheck disable=SC2086
      run_braid "$rep" "${spec%%:*}" ${spec#*:} | tee -a "$ROWS"
    fi
  done
done

echo
python3 scripts/h2h_summarize.py "$ROWS"
