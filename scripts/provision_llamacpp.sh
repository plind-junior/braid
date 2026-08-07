#!/usr/bin/env bash
# Build llama.cpp on the measurement box and stage the MVP baseline model.
#
# llama.cpp is the MVP's denominator: the deliverable is "x% faster than
# llama.cpp", so its number has to be measured HERE, on this card, on this
# driver, in the same session as ours. The reference engine's published
# llama.cpp column was taken on a different host state and cannot be borrowed.
#
# Unlike that engine (CUDA 13.2+, C++23, gcc 15) llama.cpp builds fine on this
# box's CUDA 12.8 + gcc 11.4, so no toolchain sideload is needed.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

LLAMA_DIR=${LLAMA_DIR:-/root/llama.cpp}
MODEL_DIR=${MODEL_DIR:-/root/models}

apt-get update -qq
apt-get install -y --no-install-recommends git build-essential curl libcurl4-openssl-dev
python3 -m pip install -q --upgrade cmake huggingface_hub

mkdir -p "$MODEL_DIR"

if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
echo "=== llama.cpp build $(git rev-parse --short HEAD) ==="

# CMAKE_CUDA_ARCHITECTURES=120 is plain sm_120 (not 120a): llama.cpp does not
# use arch-conditional instructions, and asking for 120a here only risks a
# ptxas refusal on kernels that do not need it.
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON >/dev/null

cmake --build build --config Release -j "$(nproc)" \
  --target llama-bench llama-server llama-cli 2>&1 | tail -5

echo "=== built ==="
./build/bin/llama-bench --help >/dev/null && echo "llama-bench OK"
./build/bin/llama-server --help >/dev/null && echo "llama-server OK"
