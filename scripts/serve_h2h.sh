#!/usr/bin/env bash
# Server vs server: braid/serve over HTTP against llama-server, same box, same
# client, same prompts. This is the measurement the decode head-to-head could
# not make: it includes each side's scheduler, prefill path, HTTP layer and
# streaming — i.e. what a user actually connects to.
#
# Method notes, same contract as head_to_head.sh:
#   * Both servers are restarted per (concurrency, rep) point — parallelism is
#     a startup parameter on both sides (-np / --capacity), and a server sized
#     for 128 slots serving 1 stream would be a different configuration than
#     the one a c=1 operator would run.
#   * Arm order alternates per rep so drift cannot systematically favour one.
#   * llama-server gets the flags the published decode numbers used
#     (-ngl 99 -fa 1) plus -np/-c sized to the point; everything else default.
#   * The client publishes `client_busy` per point; a point where the client
#     saturated its core is labelled client_bound in the rows and must not be
#     published as a server number.
#   * Rows land under /root/, outside the rsync --delete zone.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-/root/llama.cpp}
BIN="$LLAMA_DIR/build/bin"
GGUF=${GGUF:-/root/models/Qwen3.5-9B-Q8_0.gguf}
MODEL_DIR=${BRAID_MODEL_DIR:-/root/models/Qwen3.5-9B}
CONCURRENCIES=${CONCURRENCIES:-"1 8 16 32 64 128"}
NPP=${NPP:-128}
NTG=${NTG:-64}
REPS=${REPS:-3}
ROWS=${ROWS:-/root/serve_h2h.txt}
LPORT=${LPORT:-8090}
BPORT=${BPORT:-8091}

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "ABORT: $busy other process(es) on the GPU" >&2
  exit 1
fi

echo "meta llamacpp_build $(cd "$LLAMA_DIR" && git rev-parse --short HEAD)" | tee -a "$ROWS"
echo "meta shape npp=$NPP ntg=$NTG reps=$REPS conc=$CONCURRENCIES" | tee -a "$ROWS"

wait_health() {  # $1 = url, $2 = name
  for _ in $(seq 1 240); do
    if curl -sf -o /dev/null "$1"; then return 0; fi
    sleep 1
  done
  echo "ABORT: $2 never became healthy at $1" >&2
  return 1
}

client() {  # $1 = server kind, $2 = port, $3 = concurrency, $4 = rep
  PYTHONPATH=/root/braid python3 -B scripts/serve_client.py \
    --server "$1" --port "$2" --concurrency "$3" \
    --prompt-len "$NPP" --max-new-tokens "$NTG" --requests-per-stream 2 \
    --seed $((1234 + $4)) \
    | awk -v r="$4" '{print "row", r, $0}' | tee -a "$ROWS"
}

run_llama() {  # $1 = concurrency, $2 = rep
  local ctx=$(( $1 * (NPP + NTG + 8) ))
  # --threads-http is CAPACITY, not tuning, and it must scale with -np: the
  # default is hardware_concurrency-1 (63 here) and each streaming response
  # holds a pool thread for its whole generation, so at c=128 half the
  # requests queue behind the pool and TTFT measures the queue — measured at
  # 287 SECONDS p50 before this flag, with the client provably idle. braid's
  # accept backlog got the identical treatment; a comparison where only one
  # side is provisioned for its slot count is not a comparison.
  "$BIN/llama-server" -m "$GGUF" -c "$ctx" -np "$1" -ngl 99 -fa 1 \
    --threads-http $(( $1 * 2 > 64 ? $1 * 2 : 64 )) \
    --host 127.0.0.1 --port "$LPORT" >/dev/null 2>&1 &
  local pid=$!
  wait_health "http://127.0.0.1:$LPORT/health" llama-server
  client llama "$LPORT" "$1" "$2"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
}

run_braid() {  # $1 = concurrency, $2 = rep
  BRAID_MODEL_DIR="$MODEL_DIR" PYTHONPATH=/root/braid \
    python3 -B -m braid.serve.server --model-dir "$MODEL_DIR" \
    --capacity "$1" --max-len $((NPP + NTG + 16)) \
    --quant all --state-dtype fp16 --port "$BPORT" >/dev/null 2>&1 &
  local pid=$!
  wait_health "http://127.0.0.1:$BPORT/health" braid
  client braid "$BPORT" "$1" "$2"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
  sleep 2   # let the driver reclaim before the next arm allocates
}

for rep in $(seq 1 "$REPS"); do
  for c in $CONCURRENCIES; do
    if [ $((rep % 2)) -eq 1 ]; then
      run_llama "$c" "$rep"; run_braid "$c" "$rep"
    else
      run_braid "$c" "$rep"; run_llama "$c" "$rep"
    fi
  done
done
echo "ALL_DONE" | tee -a "$ROWS"
