#!/usr/bin/env bash
# PHASE 3 — the engine in its own process, against the two in-process arms.
#
# The chain that led here, all measured (docs/runbooks/vllm-recon.md), c=64:
#   kernels alone   6,570.8 tok/s   (decode_speed B=64, N=5, spread 0.14%)
#   in-process      3,321.6         (serve_bench; BEATS vLLM-FP8's 3,164)
#   vLLM-FP8 HTTP   3,164           <- the bar
#   HTTP threaded   2,283           phase 0 baseline
#   + fast frame    2,318.5         phase 1, +3.9%
#   + asyncio       2,534.0         phase 2, +9.3%, srv_step_ms 27.12 -> 25.06
# Phase 2 confirmed the mechanism (65 threads -> 2) and left step() at 25.06 ms
# against the ~19 ms arm B implies. The remaining sharing is one GIL between
# serving Python and engine Python. Phase 3 removes it: two processes, a unix
# socket, one frame per tick. vLLM does exactly this (`EngineCore pid=...`).
#
# Three arms, IDENTICAL command lines except `--http`. Two passes in reversed
# order, because a win that only appears in one order is host drift.
#
# What would falsify the phase-3 hypothesis: split lands inside the asyncio
# arm's spread, or srv_step_ms does not move. Publish that if it happens.
set -euo pipefail

MODEL_DIR=${BRAID_MODEL_DIR:-/root/models/Qwen3.5-9B}
OUT=${OUT:-/root/phase3.txt}
LOGS=${LOGS:-/root/phase3-logs}
PORT=${PORT:-8092}
C=${C:-64}
NPP=${NPP:-128}
NTG=${NTG:-64}
REPS=${REPS:-3}
# `split` first on purpose: it is the only untried arm, and a new arm that
# cannot come up should cost one model load to find out, not three. Pass 2
# reverses this, so both orders are still covered.
ARMS=${ARMS:-"split asyncio threaded"}

mkdir -p "$LOGS"

gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }

wait_gpu_free() {  # $1 = seconds
  local i
  for i in $(seq 1 "${1:-60}"); do
    [ -z "$(gpu_pids)" ] && return 0
    sleep 1
  done
  # A leftover engine process holding 20 GB is not something to measure
  # around: the next arm would either OOM or quietly run slower.
  echo "meta PHASE3 sweeping leftover GPU pids: $(gpu_pids | tr '\n' ' ')" \
    | tee -a "$OUT"
  pkill -9 -f 'braid.serve.engine_proc' 2>/dev/null || true
  pkill -9 -f 'braid.serve.server' 2>/dev/null || true
  sleep 5
  [ -z "$(gpu_pids)" ]
}

if [ -n "$(gpu_pids)" ]; then
  echo "ABORT: other process(es) on the GPU: $(gpu_pids | tr '\n' ' ')" >&2
  exit 1
fi

echo "meta PHASE3 c=$C npp=$NPP ntg=$NTG reps=$REPS shape=ramp0/2rps arms=$ARMS" \
  | tee -a "$OUT"
echo "meta driver $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1)" \
  | tee -a "$OUT"

run_arm() {  # $1 = http mode, $2 = pass label
  local log pid ok
  log="$LOGS/$1-$2.log"
  BRAID_MODEL_DIR="$MODEL_DIR" PYTHONPATH=/root/braid \
    python3 -B -m braid.serve.server --model-dir "$MODEL_DIR" \
    --capacity "$C" --max-len $((NPP + NTG + 16)) \
    --quant all --state-dtype fp16 --port "$PORT" --http "$1" \
    >"$log" 2>&1 &
  pid=$!
  ok=1
  for _ in $(seq 1 300); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then ok=0; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "row PHASE3 $1 $2 ARM_FAILED (server exited)" | tee -a "$OUT"
      tail -c 900 "$log" | tee -a "$OUT"
      wait_gpu_free 30 || true
      return 0
    fi
    sleep 1
  done
  if [ "$ok" -ne 0 ]; then
    echo "row PHASE3 $1 $2 ARM_FAILED (never healthy)" | tee -a "$OUT"
    kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    wait_gpu_free 30 || true
    return 0
  fi

  # RECEIPT. The whole claim of this phase is "the engine is a different
  # process", so read it off the box rather than off the design: /health
  # reports both pids, and nvidia-smi says which of them owns the card.
  if [ "$1" = split ]; then
    echo "receipt PHASE3 pids $(grep -m1 -o 'front end pid [0-9]*, engine pid [0-9]*' "$log" || echo '?')" \
      | tee -a "$OUT"
    echo "receipt PHASE3 gpu_owner $(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader | tr '\n' ';')" \
      | tee -a "$OUT"
    echo "receipt PHASE3 launched_pid $pid (front end; the GPU owner above must differ)" \
      | tee -a "$OUT"
  fi

  client() {
    PYTHONPATH=/root/braid python3 -B scripts/serve_client.py \
      --server braid --model phase3 --port "$PORT" --concurrency "$C" \
      --prompt-len "$NPP" --max-new-tokens "$NTG" --requests-per-stream 2 \
      --ramp-ms 0 --health-url "http://127.0.0.1:$PORT/health" --seed 1234
  }
  client >/dev/null 2>&1 || true          # discard: allocation + graph warmup
  for r in $(seq 1 "$REPS"); do
    echo -n "row PHASE3 $1 $2 rep=$r " | tee -a "$OUT"
    client | tee -a "$OUT" || echo "REP_FAILED" | tee -a "$OUT"
  done

  # SIGTERM, not SIGKILL: in split mode the front end's signal handler is what
  # stops the engine process, and an orphaned engine would poison the next arm.
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  wait_gpu_free 60 || echo "meta PHASE3 GPU still busy after $1 $2" | tee -a "$OUT"
  sleep 5
}

for a in $ARMS; do run_arm "$a" p1; done
# shellcheck disable=SC2086
for a in $(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' '); do run_arm "$a" p2; done

python3 - "$OUT" <<'PY' | tee -a "$OUT"
import json, re, statistics, sys
rows, per_pass, receipts = {}, {}, []
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("receipt PHASE3") and line not in receipts:
        receipts.append(line)
    m = re.match(r"row PHASE3 (\S+) (\S+) rep=\d+ (\{.*\})", line)
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
    print("meta PHASE3 SUMMARY no parseable rows")
    raise SystemExit
for r in receipts:
    print(r.replace("receipt PHASE3 ", "meta receipt "))
base = None
print(f"{'arm':<10} {'n':>2} {'tok/s med':>10} {'spread%':>8} {'vs thr':>8} "
      f"{'srv_step_ms':>12} {'idle%':>6} {'ITL p50':>8} {'TTFT p50':>9} "
      f"{'gen%':>5}")
for arm in ["threaded", "asyncio", "split"]:
    if arm not in rows:
        continue
    ds = rows[arm]
    t = sorted(x["tok_s"] for x in ds)
    med = statistics.median(t)
    if arm == "threaded":
        base = med
    delta = f"{(med / base - 1) * 100:+.1f}%" if base else "ref"
    exp = sum(x.get("tokens_expected", 0) for x in ds)
    gen = sum(x.get("tokens", 0) for x in ds) / exp * 100 if exp else 0.0
    print(f"{arm:<10} {len(t):>2} {med:>10.1f} "
          f"{(t[-1] - t[0]) / med * 100:>7.2f}% {delta:>8} "
          f"{statistics.median([x.get('srv_step_ms', 0) for x in ds]):>12.2f} "
          f"{statistics.median([x.get('srv_idle_pct', 0) for x in ds]):>5.1f}% "
          f"{statistics.median([x['itl_ms_p50'] for x in ds]):>8.2f} "
          f"{statistics.median([x['ttft_ms_p50'] for x in ds]):>9.1f} "
          f"{gen:>4.0f}%")
print()
print("per-pass medians (a win must survive BOTH orders):")
for (arm, p), v in sorted(per_pass.items()):
    print(f"    {arm:<10} {p} {statistics.median(v):>9.1f}")
err = sum(x.get("errors", 0) for ds in rows.values() for x in ds)
print(f"\nerrors across all reps: {err}")
print("meta PHASE3 refs in_process_armB=3322 vllm_fp8_http=3164 "
      "phase2_asyncio=2534 phase1_threaded=2318")
PY

echo "PHASE3_DONE" | tee -a "$OUT"
