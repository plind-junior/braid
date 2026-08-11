#!/usr/bin/env bash
# PHASE 0 — split braid's c=64 deficit into "kernels", "scheduler", "HTTP" and
# "load pattern".
#
# Round 1 (2026-08-11) established the first three:
#   A  decode_speed B=64  6,581 tok/s   9.72 ms/step   kernels alone
#   B  serve_bench  c=64  3,326 tok/s  11.89 ms ITL    + prefill + scheduler
#   C  HTTP         c=64  2,107 tok/s  19.67 ms ITL    + the serving layer
# against vLLM-FP8's 3,164 / 11.65 ms over HTTP on the same box the same day.
# So braid's engine BEATS vLLM in-process and the whole deficit is B->C.
#
# Round 1 could not say WHY, for a reason that was its own defect: arm B
# submits all 64 requests at once while arm C ramped starts 10 ms apart and ran
# two sequential requests per stream, so B->C mixes serving-layer cost with
# arrival pattern (TTFT p50 1,709 ms vs 171 ms). Round 2 separates them by
# holding the server fixed and varying only the client:
#
#   C1  ramp 10 ms, 2 req/stream   the recon baseline (expect ~2,107)
#   C2  ramp 0,     2 req/stream   isolates the ramp; matches serve_bench's
#                                  all-at-once submission
#   C3  ramp 0,     1 req/stream   also removes request turnaround entirely
#
# C1->C2 prices the ramp. C2->C3 prices turnaround. Whatever is left at C3
# against arm B is the serving layer itself, and it is the only part a code
# change in braid/serve/ can address. Every HTTP row also carries the server's
# own per-tick counters (srv_step_ms / srv_fanout_ms / srv_idle_pct), scoped by
# the client to exactly the measured wave.
#
# Arm A runs N=5 because round 1's single rep read 6,581 against the README's
# published 5,232 (+25.8%, far outside the 8-15% host-drift band). One rep
# against a published median-of-5 is not grounds to correct a published row;
# five is.
#
# Rows go to /root/, outside the rsync --delete zone. Do NOT edit the local
# tree while this runs: scripts/remote.sh rsyncs before every command.
set -euo pipefail

MODEL_DIR=${BRAID_MODEL_DIR:-/root/models/Qwen3.5-9B}
OUT=${OUT:-/root/phase0.txt}
LOGS=${LOGS:-/root/phase0-logs}
PORT=${PORT:-8091}
C=${C:-64}
NPP=${NPP:-128}
NTG=${NTG:-64}
QUANT=${QUANT:-all}
STATE=${STATE:-fp16}
REPS=${REPS:-5}

mkdir -p "$LOGS"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "ABORT: $busy other process(es) on the GPU" >&2
  exit 1
fi

echo "meta PHASE0 c=$C npp=$NPP ntg=$NTG quant=$QUANT state=$STATE reps=$REPS" | tee -a "$OUT"
echo "meta driver $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1)" | tee -a "$OUT"

# --- Arm A: kernels alone, N=5 -------------------------------------------
# One module per process is a hard requirement on the 9B (two module-scoped
# engines no longer share a 31 GB card), so every rep is its own invocation --
# which is also what "median across separate processes" means here.
# --prompt-len 128 --steps 128 puts KV at 128..256, the range the published v2
# sweep states for every arm; the module's own default is prompt-len 8 and
# using it would compare a different KV regime against 5,232.
echo "== arm A: decode_speed B=$C, $REPS reps" | tee -a "$OUT"
for r in $(seq 1 "$REPS"); do
  PYTHONPATH=/root/braid python3 -B -m braid.bench.decode_speed \
    --batches "$C" --prompt-len 128 --steps 128 --max-len 512 \
    --quant "$QUANT" --state-dtype "$STATE" --json \
    >"$LOGS/armA-$r.json" 2>"$LOGS/armA-$r.err" \
    || echo "ARM_A_REP_${r}_FAILED" | tee -a "$OUT"
done
python3 - "$LOGS" "$REPS" <<'PY' | tee -a "$OUT"
import json, sys, statistics
logs, reps = sys.argv[1], int(sys.argv[2])
by = {}
for r in range(1, reps + 1):
    try:
        d = json.load(open(f"{logs}/armA-{r}.json"))
    except Exception:
        continue
    for a in d["arms"]:
        by.setdefault(a["name"], []).append(a["tok_per_s"])
for name, vals in by.items():
    if not vals:
        continue
    v = sorted(vals)
    spread = (v[-1] - v[0]) / statistics.median(v) * 100 if len(v) > 1 else 0.0
    print(f"row PHASE0 armA {name} n={len(v)} median={statistics.median(v):.1f} "
          f"min={v[0]:.1f} max={v[-1]:.1f} spread_pct={spread:.2f}")
PY

# --- Arm B: in-process serving -------------------------------------------
echo "== arm B: serve_bench c=$C" | tee -a "$OUT"
PYTHONPATH=/root/braid python3 -B -m braid.bench.serve_bench \
  --concurrency "$C" --prompt-len "$NPP" --max-new-tokens "$NTG" \
  --requests-per-stream 2 --quant "$QUANT" --state-dtype "$STATE" --json \
  >"$LOGS/armB.json" 2>"$LOGS/armB.err" || echo "ARM_B_FAILED" | tee -a "$OUT"
tail -c 2000 "$LOGS/armB.json" | tee -a "$OUT"

# --- Arm C: over HTTP, three client shapes against ONE server -------------
# One server for all three: a restart between them would put a fresh
# allocator, a fresh graph pool and a different thermal point under each shape,
# and the whole question here is a client-side difference.
echo "== arm C: HTTP c=$C (C1 ramped/2rps, C2 no-ramp/2rps, C3 no-ramp/1rps)" | tee -a "$OUT"
BRAID_MODEL_DIR="$MODEL_DIR" PYTHONPATH=/root/braid \
  python3 -B -m braid.serve.server --model-dir "$MODEL_DIR" \
  --capacity "$C" --max-len $((NPP + NTG + 16)) \
  --quant "$QUANT" --state-dtype "$STATE" --port "$PORT" \
  >"$LOGS/armC-server.log" 2>&1 &
SRV=$!
ok=1
for _ in $(seq 1 240); do
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then ok=0; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "ARM_C_FAILED: server exited during startup" | tee -a "$OUT"
    tail -c 900 "$LOGS/armC-server.log" | tee -a "$OUT"
    ok=2; break
  fi
  sleep 1
done

if [ "$ok" -eq 0 ]; then
  run_c() {  # $1 = label, $2 = ramp_ms, $3 = requests per stream
    echo -n "row PHASE0 armC $1 ramp_ms=$2 rps=$3 " | tee -a "$OUT"
    PYTHONPATH=/root/braid python3 -B scripts/serve_client.py \
      --server braid --model phase0 --port "$PORT" --concurrency "$C" \
      --prompt-len "$NPP" --max-new-tokens "$NTG" --requests-per-stream "$3" \
      --ramp-ms "$2" --health-url "http://127.0.0.1:$PORT/health" \
      --seed 1234 | tee -a "$OUT" || echo "ARM_C_${1}_FAILED" | tee -a "$OUT"
  }
  run_c C1 10 2
  run_c C2 0  2
  run_c C3 0  1
fi
kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true

echo "PHASE0_DONE" | tee -a "$OUT"
