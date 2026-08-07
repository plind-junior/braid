#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends python3-pip ninja-build ccache rsync
# torch is preinstalled on this image (2.11.0+cu128); do not reinstall it.
python3 -c "import torch" 2>/dev/null \
  || pip3 install torch --index-url https://download.pytorch.org/whl/cu128
# numpy is required: torch 2.11 emits "Failed to initialize NumPy" without it
# and some tensor conversions fail outright.
pip3 install -q pytest numpy
echo "provisioned"
python3 - <<'PY'
import torch
print("torch     :", torch.__version__)
print("archs     :", torch.cuda.get_arch_list())
print("device    :", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
