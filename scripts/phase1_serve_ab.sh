#!/usr/bin/env bash
# PHASE 1 — price the two cheap serving-layer levers phase 0 pointed at.
#
# Phase 0 (2026-08-11, docs/runbooks/vllm-recon.md) put braid's engine AHEAD of
# vLLM-FP8 in-process (3,322 vs 3,164 tok/s at c=64) and located the entire
# deficit in the HTTP layer: the same Scheduler.step() takes 27.5 ms per tick
# under the server against an 11.9 ms in-process ITL, while the loop is idle
# 0.7% of the window and the queue fan-out costs 0.29 ms. The inflation is
# INSIDE step(), which is where the scheduler's Python runs between
# GIL-releasing CUDA calls, with 64 handler threads contending.
#
# Two levers, both A/B-able by environment alone so every arm is the same
# binary on the same weights:
#   BRAID_FAST_FRAME      hand-built SSE frame instead of json.dumps per token
#                         (byte-identical; 28/28 pairs checked offline)
#   BRAID_SWITCH_INTERVAL widen CPython's GIL handoff period from its 5 ms
#                         default, so the scheduler thread finishes a burst of
#                         Python before preemption
#
# The switch-interval arms are as much PROBE as fix: if throughput does not
# move with them, GIL contention is not the mechanism and the answer is
# structural (one thread, not 65) rather than a knob.
#
# Measurement shape is phase 0's C2 arm — ramp 0, 2 requests per stream — the
# cleanest HTTP configuration and the one that matches serve_bench's
# all-at-once submission. Baseline to beat: 2,283 tok/s.
#
# Each arm is its own process (the env is read at startup), which is what
# "median across separate processes" requires; the 3 reps inside an arm bound
# that arm's own spread. Arm order is rotated on a second pass so a drift
# cannot be read as a lever.
set -euo pipefail

MODEL_DIR=${BRAID_MODEL_DIR:-/root/models/Qwen3.5-9B}
OUT=${OUT:-/root/phase1.txt}
LOGS=${LOGS:-/root/phase1-logs}
PORT=${PORT:-8091}
C=${C:-64}
NPP=${NPP:-128}
NTG=${NTG:-64}
REPS=${REPS:-3}

mkdir -p "$LOGS"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "ABORT: $busy other process(es) on the GPU" >&2
  exit 1
fi

echo "meta PHASE1 c=$C npp=$NPP ntg=$NTG reps=$REPS shape=ramp0/2rps" | tee -a "$OUT"
echo "meta driver $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1)" | tee -a "$OUT"

# arm name -> "FAST_FRAME SWITCH_INTERVAL" ("-" = leave unset)
arm_env() {
  case "$1" in
    base)      echo "0 -" ;;
    frame)     echo "1 -" ;;
    frame_si20)  echo "1 0.02" ;;
    frame_si100) echo "1 0.10" ;;
  esac
}

run_arm() {  # $1 = arm name, $2 = pass label
  local ff si log pid
  read -r ff si <<<"$(arm_env "$1")"
  log="$LOGS/$1-$2.log"
  if [ "$si" = "-" ]; then
    BRAID_FAST_FRAME="$ff" BRAID_MODEL_DIR="$MODEL_DIR" PYTHONPATH=/root/braid \
      python3 -B -m braid.serve.server --model-dir "$MODEL_DIR" \
      --capacity "$C" --max-len $((NPP + NTG + 16)) \
      --quant all --state-dtype fp16 --port "$PORT" >"$log" 2>&1 &
  else
    BRAID_FAST_FRAME="$ff" BRAID_SWITCH_INTERVAL="$si" \
      BRAID_MODEL_DIR="$MODEL_DIR" PYTHONPATH=/root/braid \
      python3 -B -m braid.serve.server --model-dir "$MODEL_DIR" \
      --capacity "$C" --max-len $((NPP + NTG + 16)) \
      --quant all --state-dtype fp16 --port "$PORT" >"$log" 2>&1 &
  fi
  pid=$!
  local ok=1
  for _ in $(seq 1 240); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then ok=0; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "row PHASE1 $1 $2 ARM_FAILED (server exited)" | tee -a "$OUT"
      tail -c 600 "$log" | tee -a "$OUT"
      return 0
    fi
    sleep 1
  done
  if [ "$ok" -ne 0 ]; then
    echo "row PHASE1 $1 $2 ARM_FAILED (never healthy)" | tee -a "$OUT"
    kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    return 0
  fi

  client() {
    PYTHONPATH=/root/braid python3 -B scripts/serve_client.py \
      --server braid --model phase1 --port "$PORT" --concurrency "$C" \
      --prompt-len "$NPP" --max-new-tokens "$NTG" --requests-per-stream 2 \
      --ramp-ms 0 --health-url "http://127.0.0.1:$PORT/health" --seed 1234
  }
  # Rep 0 is discarded: first-touch allocation and graph warmup land in it.
  client >/dev/null 2>&1 || true
  for r in $(seq 1 "$REPS"); do
    echo -n "row PHASE1 $1 $2 rep=$r ff=$ff si=$si " | tee -a "$OUT"
    client | tee -a "$OUT" || echo "REP_FAILED" | tee -a "$OUT"
  done
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 5
}

for a in base frame frame_si20 frame_si100; do run_arm "$a" p1; done
# Reverse order: an arm that only wins when it runs first is measuring drift.
for a in frame_si100 frame_si20 frame base; do run_arm "$a" p2; done

python3 - "$OUT" <<'PY' | tee -a "$OUT"
import json, re, statistics, sys
rows = {}
for line in open(sys.argv[1]):
    m = re.match(r"row PHASE1 (\S+) (\S+) rep=\d+ ff=\S+ si=\S+ (\{.*\})", line.strip())
    if not m:
        continue
    arm, _, payload = m.groups()
    try:
        d = json.loads(payload)
    except json.JSONDecodeError:
        continue
    rows.setdefault(arm, []).append(d)
if not rows:
    print("meta PHASE1 SUMMARY no parseable rows")
else:
    base = None
    print(f"{'arm':<13} {'n':>2} {'tok/s med':>10} {'spread%':>8} "
          f"{'vs base':>8} {'srv_step_ms':>12} {'idle%':>6} {'ITL p50':>8}")
    order = ["base", "frame", "frame_si20", "frame_si100"]
    for arm in [a for a in order if a in rows] + [a for a in rows if a not in order]:
        ds = rows[arm]
        t = sorted(x["tok_s"] for x in ds)
        med = statistics.median(t)
        spread = (t[-1] - t[0]) / med * 100 if len(t) > 1 else 0.0
        if arm == "base":
            base = med
        delta = f"{(med / base - 1) * 100:+.1f}%" if base else "ref"
        step = statistics.median([x.get("srv_step_ms", 0) for x in ds])
        idle = statistics.median([x.get("srv_idle_pct", 0) for x in ds])
        itl = statistics.median([x["itl_ms_p50"] for x in ds])
        print(f"{arm:<13} {len(t):>2} {med:>10.1f} {spread:>7.2f}% {delta:>8} "
              f"{step:>12.2f} {idle:>5.1f}% {itl:>8.2f}")
    print("meta PHASE1 baseline_to_beat=2283 in_process_arm_B=3322 "
          "vllm_fp8_http=3164")
PY

echo "PHASE1_DONE" | tee -a "$OUT"
