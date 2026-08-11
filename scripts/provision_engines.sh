#!/usr/bin/env bash
# Install vLLM and SGLang on the measurement box, each in its own venv.
#
# Why venvs and not the system interpreter: both engines pin their own torch,
# and braid's JIT kernel path depends on the image's preinstalled
# torch 2.11.0+cu128. A `pip install vllm` into the system interpreter would
# silently replace it and the next `make test-remote` would fail to build a
# kernel for reasons that look nothing like the cause. They also pin
# conflicting flashinfer versions against each other, so they cannot share one
# venv either. `--system-site-packages` is deliberately OFF.
#
# Why not Docker: this box IS a container. Nesting one buys isolation we
# already get from a venv and costs an image pull per engine.
#
# /root/venvs is outside the repo, so `rsync -az --delete` does not eat it and
# a stop/start cycle keeps it on the 300 GB disk.
#
# PYTHON 3.12, NOT THE SYSTEM 3.10. Measured on this box: vLLM 0.27.0 declares
# `Requires-Python >=3.10`, installs happily on the image's 3.10.12, and then
# dies at engine-core init inside its own dependency —
#   flashinfer/comm/fd_exchange.py:55
#     def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:
#   TypeError: 'type' object is not subscriptable
# because `array.array` only became subscriptable in 3.12. The declared floor
# is wrong. Pinning the interpreter is the honest fix; deleting flashinfer
# would disable vLLM's fusion passes and quietly tune the competitor DOWN,
# which is not a comparison worth publishing.
#
# Usage:  ./scripts/remote.sh 'bash scripts/provision_engines.sh [vllm|sglang|both]'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

WHICH=${1:-both}
VENV_ROOT=${VENV_ROOT:-/root/venvs}
MODEL_DIR=${BRAID_MODEL_DIR:-/root/models/Qwen3.5-9B}
# SGLang on sm_120 is a known-poor bet (trtllm_mha is SM100-only, sgl-kernel
# has reported missing SM_120 images). It gets a hard clock, not a rabbit hole.
SGLANG_TIMEOUT=${SGLANG_TIMEOUT:-2700}

PYVER=${PYVER:-3.12}

apt-get update -qq
apt-get install -y --no-install-recommends python3-venv curl

mkdir -p "$VENV_ROOT"

# uv fetches a standalone CPython; the image is Ubuntu 22.04, whose only
# system interpreter is 3.10 and which has no 3.12 in its archive.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
uv python install "$PYVER"

mkvenv() {  # $1 = venv path — recreated if it is on the wrong interpreter
  local have=""
  if [ -x "$1/bin/python" ]; then
    have=$("$1/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  fi
  if [ "$have" != "$PYVER" ]; then
    echo "recreating $1 on python $PYVER (was ${have:-absent})"
    rm -rf "$1"
    uv venv --python "$PYVER" "$1" >/dev/null
  fi
}

# Report the capability the engines actually have to satisfy, from the venv's
# own torch rather than the system one — the whole point is that they differ.
probe() {  # $1 = venv path, $2 = name
  "$1/bin/python" - "$2" <<'PY'
import sys
name = sys.argv[1]
import torch
print(f"{name}: torch {torch.__version__}, "
      f"cc {torch.cuda.get_device_capability()}, "
      f"arch_list {torch.cuda.get_arch_list()}")
PY
}

install_vllm() {
  local v="$VENV_ROOT/vllm"
  mkvenv "$v"
  # >=0.17 is the floor: that is the release that added the Qwen3.5 Gated
  # DeltaNet path (FLA Triton kernels + the hybrid KV cache manager). An
  # earlier vLLM does not serve this architecture at all.
  uv pip install -q --python "$v/bin/python" "vllm>=0.17"
  echo "=== vllm $("$v/bin/python" -c 'import vllm; print(vllm.__version__)') ==="
  probe "$v" vllm
}

install_sglang() {
  local v="$VENV_ROOT/sglang"
  mkvenv "$v"
  if ! timeout "$SGLANG_TIMEOUT" uv pip install -q --python "$v/bin/python" "sglang[all]"; then
    echo "SGLANG_INSTALL_FAILED (timeout ${SGLANG_TIMEOUT}s or resolver error)" >&2
    return 1
  fi
  echo "=== sglang $("$v/bin/python" -c 'import sglang; print(sglang.__version__)') ==="
  probe "$v" sglang
}

case "$WHICH" in
  vllm)   install_vllm ;;
  sglang) install_sglang ;;
  both)
    install_vllm
    # A dead SGLang must not fail the whole provisioning step — vLLM is the
    # arm that matters and the recon is designed to publish SGLang's error as
    # a result in its own right.
    install_sglang || echo "SGLANG_UNAVAILABLE: continuing with vllm only" >&2
    ;;
  *) echo "usage: $0 [vllm|sglang|both]" >&2; exit 2 ;;
esac

[ -d "$MODEL_DIR" ] && echo "model present: $MODEL_DIR" \
  || echo "WARNING: $MODEL_DIR missing — run scripts/stage_model.sh first" >&2
echo "=== provision_engines done ==="
