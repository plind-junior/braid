#!/usr/bin/env bash
# PHASE 2 — threaded HTTP front end vs the asyncio one, same engine.
#
# The chain that led here, all measured (docs/runbooks/vllm-recon.md):
#   phase 0  braid in-process 3,322 tok/s at c=64 BEATS vLLM-FP8's 3,164;
#            over HTTP it is 2,283. The same Scheduler.step() costs 27.5 ms
#            per tick under the server against an 11.9 ms in-process ITL.
#   phase 0  eliminated arrival pattern (idle 0.7%) and the queue fan-out
#            (0.29 ms/tick).
#   phase 1  eliminated GIL handoff cadence (setswitchinterval 20 ms and
#            100 ms are nulls) and bought +3.9% from a hand-built SSE frame.
#            64 cores on the box, so it is not core starvation either.
# What is left is that one thread runs Python at a time and 64 handler threads
# serialize against the scheduler's own Python inside step(). Phase 2 removes
# the 64 threads: one event loop, and ONE scheduler->loop handoff per tick
# instead of one per stream.
#
# Both arms are the same binary, same weights, same `--quant all
# --state-dtype fp16`, and the same measurement shape as phase 1 (ramp 0,
# 2 requests per stream, c=64). The only difference is `--http`.
#
# Targets: beat the threaded arm's 2,312, and close on arm B's 3,322. vLLM-FP8
# over HTTP on this box is 3,164 — that is the bar that matters.
set -euo pipefail

MODEL_DIR=${BRAID_MODEL_DIR:-/root/models/Qwen3.5-9B}
OUT=${OUT:-/root/phase2.txt}
LOGS=${LOGS:-/root/phase2-logs}
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

echo "meta PHASE2 c=$C npp=$NPP ntg=$NTG reps=$REPS shape=ramp0/2rps" | tee -a "$OUT"
echo "meta driver $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1)" | tee -a "$OUT"

run_arm() {  # $1 = http mode (threaded|asyncio), $2 = pass label
  local log pid ok
  log="$LOGS/$1-$2.log"
  BRAID_MODEL_DIR="$MODEL_DIR" PYTHONPATH=/root/braid \
    python3 -B -m braid.serve.server --model-dir "$MODEL_DIR" \
    --capacity "$C" --max-len $((NPP + NTG + 16)) \
    --quant all --state-dtype fp16 --port "$PORT" --http "$1" \
    >"$log" 2>&1 &
  pid=$!
  ok=1
  for _ in $(seq 1 240); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then ok=0; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "row PHASE2 $1 $2 ARM_FAILED (server exited)" | tee -a "$OUT"
      tail -c 800 "$log" | tee -a "$OUT"
      return 0
    fi
    sleep 1
  done
  if [ "$ok" -ne 0 ]; then
    echo "row PHASE2 $1 $2 ARM_FAILED (never healthy)" | tee -a "$OUT"
    kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    return 0
  fi

  client() {
    PYTHONPATH=/root/braid python3 -B scripts/serve_client.py \
      --server braid --model phase2 --port "$PORT" --concurrency "$C" \
      --prompt-len "$NPP" --max-new-tokens "$NTG" --requests-per-stream 2 \
      --ramp-ms 0 --health-url "http://127.0.0.1:$PORT/health" --seed 1234
  }
  client >/dev/null 2>&1 || true          # discard: allocation + graph warmup
  for r in $(seq 1 "$REPS"); do
    echo -n "row PHASE2 $1 $2 rep=$r " | tee -a "$OUT"
    client | tee -a "$OUT" || echo "REP_FAILED" | tee -a "$OUT"
  done
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 5
}

for a in threaded asyncio; do run_arm "$a" p1; done
for a in asyncio threaded; do run_arm "$a" p2; done   # reverse: drift check

python3 - "$OUT" <<'PY' | tee -a "$OUT"
import json, re, statistics, sys
rows, per_pass = {}, {}
for line in open(sys.argv[1]):
    m = re.match(r"row PHASE2 (\S+) (\S+) rep=\d+ (\{.*\})", line.strip())
    if not m:
        continue
    arm, p, payload = m.groups()
    try:
        d = json.loads(payload)
    except json.JSONDecodeError:
        continue
    rows.setdefault(arm, []).append(d)
    per_pass.setdefault((arm, p), []).append(d["tok_s"])
if not rows:
    print("meta PHASE2 SUMMARY no parseable rows")
    raise SystemExit
base = None
print(f"{'arm':<10} {'n':>2} {'tok/s med':>10} {'spread%':>8} {'vs thr':>8} "
      f"{'srv_step_ms':>12} {'idle%':>6} {'ITL p50':>8} {'TTFT p50':>9}")
for arm in ["threaded", "asyncio"]:
    if arm not in rows:
        continue
    ds = rows[arm]
    t = sorted(x["tok_s"] for x in ds)
    med = statistics.median(t)
    if arm == "threaded":
        base = med
    delta = f"{(med / base - 1) * 100:+.1f}%" if base else "ref"
    print(f"{arm:<10} {len(t):>2} {med:>10.1f} "
          f"{(t[-1] - t[0]) / med * 100:>7.2f}% {delta:>8} "
          f"{statistics.median([x.get('srv_step_ms', 0) for x in ds]):>12.2f} "
          f"{statistics.median([x.get('srv_idle_pct', 0) for x in ds]):>5.1f}% "
          f"{statistics.median([x['itl_ms_p50'] for x in ds]):>8.2f} "
          f"{statistics.median([x['ttft_ms_p50'] for x in ds]):>9.1f}")
print()
print("per-pass medians (a win must survive BOTH orders):")
for (arm, p), v in sorted(per_pass.items()):
    print(f"    {arm:<10} {p} {statistics.median(v):>9.1f}")
print("meta PHASE2 refs in_process_armB=3322 vllm_fp8_http=3164 "
      "phase1_threaded_best=2312")
PY

echo "PHASE2_DONE" | tee -a "$OUT"
