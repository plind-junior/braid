#!/usr/bin/env bash
# RECONNAISSANCE ONLY — braid vs llama.cpp vs vLLM vs SGLang, one rep.
#
# This is the stripped variant of serve_h2h.sh. It answers "is braid in the
# fight" cheaply; it does NOT produce a publishable number. One rep is an
# anecdote by docs/THESIS.md §4, and nothing here clears the median-of-N bar
# the README's rows are held to. Every row it writes is stamped RECON.
#
# Same contract as serve_h2h.sh where it matters:
#   * Restart per (arm, concurrency). Parallelism is a startup parameter on all
#     four engines (-np / --capacity / --max-num-seqs / --max-running-requests);
#     a server sized for 64 slots serving one stream is a different
#     configuration than the one a c=1 operator would run.
#   * Arm order ROTATES per concurrency point so drift cannot systematically
#     favour whichever engine happens to go first.
#   * Competitors run at their own defaults, sized to the point. vLLM keeps
#     chunked prefill, CUDA graphs and automatic prefix caching ON — they are
#     its defaults, and the client already issues distinct random prompts per
#     request, which keeps prefix caching out of the comparison without tuning
#     a competitor down. Qwen3.5's MTP speculative path stays off (also its
#     default); a spec-decode arm is a different experiment.
#   * The client publishes client_busy; a client_bound point is not a server
#     number.
#   * Rows and server logs land under /root/, outside the rsync --delete zone.
#
# Two additions this script makes over that contract, both about being able to
# EXPLAIN the rows rather than only report them:
#
#   * `vllm-noapc` is a CONTROL, not a competitor arm. With randomised prompts
#     vLLM's prefix cache can never hit — but leaving it on still forces the
#     hybrid cache into "align" mode, which keeps TWO recurrent-state blocks
#     resident per request instead of one (vllm/v1/kv_cache_interface.py,
#     `MambaSpec.max_memory_usage_bytes`). Publishing only the default arm
#     leaves "you measured vLLM carrying a cache it could never use" standing;
#     publishing both closes it. VLLM_NOAPC=0 drops it, saving one startup per
#     concurrency point.
#   * The GDN kernel backend vLLM resolves is CAPTURED from vLLM's own log, not
#     asserted here. On this card it is expected to read Triton/FLA: vLLM gates
#     its fast GDN prefill kernels on SM90 (FlashInfer) and on capability
#     FAMILY 100 (FlashInfer + CuteDSL), and consumer Blackwell is 12.0, so
#     both tests are false and all 24 GDN layers fall back to Triton. That is
#     the explanation for whatever the tok/s rows say, and it is the first
#     thing that goes stale when vLLM ships an sm_120 kernel — so the harness
#     reads it back every run instead of trusting a comment.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-/root/llama.cpp}
BIN="$LLAMA_DIR/build/bin"
GGUF=${GGUF:-/root/models/Qwen3.5-9B-Q8_0.gguf}
MODEL_DIR=${BRAID_MODEL_DIR:-/root/models/Qwen3.5-9B}
VENV_ROOT=${VENV_ROOT:-/root/venvs}
CONCURRENCIES=${CONCURRENCIES:-"1 16 64"}
# Stays 128: the 2026-08-11 rows in docs/runbooks/vllm-recon.md were taken at
# this shape, and the next run's job is to re-test THOSE rows with the EOS
# asymmetry fixed. Moving the default would silently make the re-run
# incomparable to the thing it exists to check.
# `NPP=512` is the follow-up shape once the c=64 gap is understood — it is where
# prefill starts to matter and where braid already has a published llama.cpp
# sweep to sit beside. `CONCURRENCIES="1 16 64 128"` is the capacity probe: at
# c=128 a bf16 vLLM is near the 32 GB ceiling (16.7 GiB weights + two recurrent
# state blocks per request) and "does not fit" is a reportable outcome. Both
# cost box time; neither is the default.
NPP=${NPP:-128}
NTG=${NTG:-64}
ROWS=${ROWS:-/root/serve_recon.txt}
LOGS=${LOGS:-/root/recon-logs}
LPORT=${LPORT:-8090}
BPORT=${BPORT:-8091}
VPORT=${VPORT:-8092}
SPORT=${SPORT:-8093}
SERVED=recon

mkdir -p "$LOGS"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "ABORT: $busy other process(es) on the GPU" >&2
  exit 1
fi

# Which arms are actually installed. A missing engine is a recorded outcome,
# not a crash — "SGLang does not serve this model on this card" is one of the
# three questions this pass exists to answer.
# NOTE: every conditional here is a full `if`, never a bare `[ ... ] && cmd`.
# Under `set -e` a bare test-and-command whose TEST fails returns non-zero and
# kills the script — so a missing SGLang venv would abort the whole sweep
# instead of recording itself as an unavailable arm.
ARMS="braid"
if [ -x "$BIN/llama-server" ]; then
  ARMS="$ARMS llama"
else
  echo "meta arm_unavailable llama (no $BIN/llama-server)" | tee -a "$ROWS"
fi
# An engine counts as installed only if it IMPORTS. A venv directory proves
# nothing: an interrupted `pip install` leaves `bin/python` behind, and a
# half-built venv detected as an arm would spend 900s failing its health check
# and land in the rows as if the engine had been measured and lost.
have() {  # $1 = engine
  [ -x "$VENV_ROOT/$1/bin/python" ] \
    && "$VENV_ROOT/$1/bin/python" -c "import $1" >/dev/null 2>&1
}
if have vllm; then
  ARMS="$ARMS vllm"
  if [ "${VLLM_FP8:-1}" = "1" ]; then ARMS="$ARMS vllm-fp8"; fi
  if [ "${VLLM_NOAPC:-1}" = "1" ]; then ARMS="$ARMS vllm-noapc"; fi
else
  echo "meta arm_unavailable vllm (not importable in $VENV_ROOT/vllm)" | tee -a "$ROWS"
fi
if have sglang; then
  ARMS="$ARMS sglang"
else
  echo "meta arm_unavailable sglang (not importable in $VENV_ROOT/sglang)" | tee -a "$ROWS"
fi

# RECON_ARMS overrides detection entirely — for running one engine's arms
# without paying for the others' startups.
ARMS=${RECON_ARMS:-$ARMS}

echo "meta RECON 1 rep — NOT PUBLISHABLE" | tee -a "$ROWS"
echo "meta shape npp=$NPP ntg=$NTG conc=$CONCURRENCIES arms=$ARMS" | tee -a "$ROWS"
echo "meta driver $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1)" | tee -a "$ROWS"
if [ -d "$LLAMA_DIR/.git" ]; then
  echo "meta llamacpp_build $(cd "$LLAMA_DIR" && git rev-parse --short HEAD)" | tee -a "$ROWS"
fi
for e in vllm sglang; do
  if [ -x "$VENV_ROOT/$e/bin/python" ]; then
    v=$("$VENV_ROOT/$e/bin/python" -c "import $e; print($e.__version__)" 2>/dev/null || echo unknown)
    echo "meta ${e}_version $v" | tee -a "$ROWS"
  fi
done

wait_health() {  # $1 = url, $2 = name, $3 = seconds, $4 = log, $5 = pid
  for _ in $(seq 1 "$3"); do
    if curl -sf -o /dev/null "$1"; then return 0; fi
    # A server that has ALREADY DIED must not be waited on for the full
    # timeout. vLLM's engine core crashed at 22s and the harness would have
    # sat on the health poll for the remaining 878s — per arm, per point,
    # billing $0.79/hr to watch a corpse. Liveness beats patience.
    if ! kill -0 "$5" 2>/dev/null; then
      echo "ABORT_ARM: $2 process exited during startup" | tee -a "$ROWS" >&2
      echo "meta ${2}_last_log $(tail -c 900 "$4" | tr '\n' ' ')" | tee -a "$ROWS"
      return 1
    fi
    sleep 1
  done
  echo "ABORT_ARM: $2 never became healthy at $1 after $3s" | tee -a "$ROWS" >&2
  echo "meta ${2}_last_log $(tail -c 600 "$4" | tr '\n' ' ')" | tee -a "$ROWS"
  return 1
}

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

# Pull a one-line startup decision out of a server's own log and record it as
# meta. DEDUPED on (tag, text): the same decision is logged by every restart, so
# emitting it per (arm, point) would bury the row file — but a decision that
# CHANGES between arms is exactly what wants to be visible, and dedup on the
# text rather than the tag keeps that visible instead of swallowing it.
RECEIPTS_SEEN=""
receipt() {  # $1 = log file, $2 = tag, $3 = EXTENDED regex, anchored at the message
  local line
  # `-o` with the pattern's own `.*` tail keeps the DECISION and drops the log
  # prefix. That matters twice over: the prefix carries a pid and a timestamp
  # that differ on every restart, so deduping on the raw line deduped nothing —
  # the 2026-08-11 v3 run wrote the same sentence nine times. Match from the
  # first informative word instead and the key is stable across restarts while
  # a genuinely different decision still gets its own line.
  line=$(grep -m1 -ohE -- "$3.*" "$1" 2>/dev/null | tr -s ' \t' ' ' | cut -c1-240) || true
  if [ -z "$line" ]; then return 0; fi
  case "$RECEIPTS_SEEN" in
    *"<$2:$line>"*) return 0 ;;
  esac
  RECEIPTS_SEEN="$RECEIPTS_SEEN<$2:$line>"
  echo "meta $2 $line" | tee -a "$ROWS"
}

client() {  # $1 = arm label, $2 = wire format, $3 = port, $4 = concurrency
  local out v
  out=$(PYTHONPATH=/root/braid python3 -B scripts/serve_client.py \
    --server "$2" --model "$SERVED" --port "$3" --concurrency "$4" \
    --prompt-len "$NPP" --max-new-tokens "$NTG" --requests-per-stream 2 \
    --seed 1234)
  # Sampled AFTER the run, while the server still holds its allocation — a
  # sample taken before the client starts misses the KV/state growth that is
  # the whole point of asking what ceiling each engine hits. Note that vLLM
  # preallocates to --gpu-memory-utilization (0.9 default), so its figure is a
  # reservation, not a working set.
  v=$(vram)
  echo "$out" | awk -v a="$1" -v v="$v" \
    '{print "row RECON", a, "vram_mib=" v, $0}' | tee -a "$ROWS"
}

# Each arm: start, wait, drive, kill. `stop_srv` waits for the port to actually
# free — the next arm allocates the whole card and a half-dead predecessor
# turns into an OOM that reads like a capacity result.
stop_srv() {  # $1 = pid
  kill "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
  sleep 5
}

run_braid() {  # $1 = concurrency
  local log="$LOGS/braid-c$1.log"
  BRAID_MODEL_DIR="$MODEL_DIR" PYTHONPATH=/root/braid \
    python3 -B -m braid.serve.server --model-dir "$MODEL_DIR" \
    --capacity "$1" --max-len $((NPP + NTG + 16)) \
    --quant all --state-dtype fp16 --port "$BPORT" >"$log" 2>&1 &
  local pid=$!
  local ok=0
  { wait_health "http://127.0.0.1:$BPORT/health" braid 240 "$log" "$pid" \
    && client braid braid "$BPORT" "$1"; } || ok=1
  stop_srv "$pid"
  return $ok
}

run_llama() {  # $1 = concurrency
  local log="$LOGS/llama-c$1.log"
  local ctx=$(( $1 * (NPP + NTG + 8) ))
  "$BIN/llama-server" -m "$GGUF" -c "$ctx" -np "$1" -ngl 99 -fa 1 \
    --threads-http $(( $1 * 2 > 64 ? $1 * 2 : 64 )) \
    --host 127.0.0.1 --port "$LPORT" >"$log" 2>&1 &
  local pid=$!
  local ok=0
  { wait_health "http://127.0.0.1:$LPORT/health" llama 240 "$log" "$pid" \
    && client llama llama "$LPORT" "$1"; } || ok=1
  stop_srv "$pid"
  return $ok
}

run_vllm() {  # $1 = concurrency, $2 = "" | "fp8" | "noapc"
  local label=vllm extra=() log
  case "${2:-}" in
    fp8)   label=vllm-fp8;   extra=(--quantization fp8) ;;
    # The control, not a tuned-down competitor — see the header. vLLM's own
    # default is prefix caching ON (config/cache.py: enable_prefix_caching=True).
    noapc) label=vllm-noapc; extra=(--no-enable-prefix-caching) ;;
  esac
  log="$LOGS/$label-c$1.log"
  # 900s: vLLM's startup includes weight load, torch.compile and CUDA-graph
  # capture. 240s is enough for braid and llama and NOT enough for this.
  "$VENV_ROOT/vllm/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name "$SERVED" \
    --max-num-seqs "$1" --max-model-len $((NPP + NTG + 16)) \
    "${extra[@]}" --host 127.0.0.1 --port "$VPORT" >"$log" 2>&1 &
  local pid=$!
  local ok=0
  { wait_health "http://127.0.0.1:$VPORT/health" "$label" 900 "$log" "$pid" \
    && client "$label" vllm "$VPORT" "$1"; } || ok=1
  stop_srv "$pid"
  # Read AFTER the process is down, so the log is complete. Both lines are
  # written during model init, but a server killed mid-capture can leave its
  # tail unflushed, and a receipt that is merely late reads like a decision
  # that was never made. Deliberately outside the `|| ok=1` block: an arm that
  # FAILED still resolved a backend, and that is often the interesting part.
  receipt "$log" gdn_backend 'Using [^ ]+ GDN prefill kernel'
  receipt "$log" mamba_cache_mode 'Mamba cache mode is set to'
  return $ok
}

run_sglang() {  # $1 = concurrency
  local log="$LOGS/sglang-c$1.log"
  "$VENV_ROOT/sglang/bin/python" -m sglang.launch_server \
    --model-path "$MODEL_DIR" --max-running-requests "$1" \
    --context-length $((NPP + NTG + 16)) \
    --host 127.0.0.1 --port "$SPORT" >"$log" 2>&1 &
  local pid=$!
  # /get_model_info answers only once the scheduler is up; /health can answer
  # from a process that has not loaded the model.
  local ok=0
  { wait_health "http://127.0.0.1:$SPORT/get_model_info" sglang 900 "$log" "$pid" \
    && client sglang sglang "$SPORT" "$1"; } || ok=1
  stop_srv "$pid"
  return $ok
}

dispatch() {  # $1 = arm name, $2 = concurrency
  case "$1" in
    braid)      run_braid "$2" ;;
    llama)      run_llama "$2" ;;
    vllm)       run_vllm  "$2" ;;
    vllm-fp8)   run_vllm  "$2" fp8 ;;
    vllm-noapc) run_vllm  "$2" noapc ;;
    sglang)     run_sglang "$2" ;;
  esac
}

# shellcheck disable=SC2086
set -- $ARMS
n=$#
i=0
for c in $CONCURRENCIES; do
  # Rotate the arm order by the point index: with one rep there is no
  # alternation to lean on, so rotation is the only thing keeping "went first"
  # from being confounded with "is faster".
  for k in $(seq 0 $((n - 1))); do
    idx=$(( (k + i) % n + 1 ))
    dispatch "${!idx}" "$c" || echo "row RECON ${!idx} c=$c ARM_FAILED" | tee -a "$ROWS"
  done
  i=$((i + 1))
done
echo "ALL_DONE" | tee -a "$ROWS"
