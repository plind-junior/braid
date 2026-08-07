#!/usr/bin/env bash
# Measure the llama.cpp baseline for the MVP, on this card, in this session.
#
# Produces three things:
#   1. single-stream prefill + decode  (llama-bench)      -- the batch-1 number
#   2. batched decode vs parallel count (llama-batched-bench) -- THE DECISIVE ONE
#   3. host health during the run
#
# (2) settles whether llama.cpp batches GDN recurrent decode. braid's whole
# thesis is that a recurrent scan without a batch axis forces the entire step
# to batch 1. The reference engine demonstrably has that limitation. If
# llama.cpp does NOT, then llama.cpp is the real competitor at concurrency and
# the target number is much higher than the reference engine's flat ~317.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-/root/llama.cpp}
MODEL=${MODEL:?set MODEL to a .gguf path}
BIN="$LLAMA_DIR/build/bin"

echo "=== model: $MODEL ($(du -h "$MODEL" | cut -f1)) ==="
echo "=== llama.cpp $(cd "$LLAMA_DIR" && git rev-parse --short HEAD) ==="
nvidia-smi --query-gpu=name,memory.used,clocks.sm,clocks.mem,power.draw \
  --format=csv,noheader

echo
echo "=== 1. single-stream: pp512 / tg128 (llama-bench) ==="
# Same shape as the reference engine's published sweep: -p 512 -n 128 -r 5 -ngl 99
"$BIN/llama-bench" -m "$MODEL" -p 512 -n 128 -r 5 -ngl 99 -fa 1

echo
echo "=== 2. batched decode vs parallel count (llama-batched-bench) ==="
echo "S_TG t/s is AGGREGATE generation throughput across PL parallel sequences."
echo "Flat in PL  => llama.cpp does not batch this model's recurrent decode."
echo "Rising in PL => it does, and it is the real competitor at concurrency."
"$BIN/llama-batched-bench" -m "$MODEL" -c 8192 -b 2048 -ub 512 \
  -npp 128 -ntg 128 -npl 1,2,4,8,16 -ngl 99 -fa 1

echo
echo "=== host after ==="
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv,noheader
