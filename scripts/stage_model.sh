#!/usr/bin/env bash
# Stage the MVP model: the GGUF llama.cpp will run, and the HF safetensors
# braid will run. Both must be the SAME model or the comparison is void.
#
# Qwen3.5-4B is chosen as the MVP target because it is a DENSE GDN hybrid:
# it exercises the recurrent scan (the thesis) without dragging in the two
# largest engineering forks -- the NVFP4 loader and the 256-expert MoE
# grouped GEMM -- neither of which is on the critical path to a number.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-/root/models}
mkdir -p "$MODEL_DIR"

python3 - <<'PY'
import os, sys
from huggingface_hub import snapshot_download, hf_hub_download

model_dir = os.environ.get("MODEL_DIR", "/root/models")

# GGUF for llama.cpp. Try the quant the reference engine published a number
# against (Q8_0).
gguf_candidates = [
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q8_0.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-UD-Q8_K_XL.gguf"),
]
got_gguf = None
for repo, fname in gguf_candidates:
    try:
        p = hf_hub_download(repo_id=repo, filename=fname, local_dir=model_dir)
        print("GGUF:", p)
        got_gguf = p
        break
    except Exception as e:
        print(f"  miss {repo}/{fname}: {type(e).__name__}: {str(e)[:160]}")

if not got_gguf:
    # Fall back to listing what the repo actually has.
    from huggingface_hub import list_repo_files
    for repo in ["unsloth/Qwen3.5-4B-GGUF"]:
        try:
            files = [f for f in list_repo_files(repo) if f.endswith(".gguf")]
            print(f"available in {repo}:", files[:20])
        except Exception as e:
            print(f"  cannot list {repo}: {type(e).__name__}: {str(e)[:160]}")

# HF safetensors for braid.
for repo in ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B-Instruct"]:
    try:
        p = snapshot_download(
            repo_id=repo,
            local_dir=os.path.join(model_dir, repo.split("/")[-1]),
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py"],
        )
        print("HF:", p)
        break
    except Exception as e:
        print(f"  miss {repo}: {type(e).__name__}: {str(e)[:200]}")
PY

echo "=== staged ==="
ls -la "$MODEL_DIR"
du -sh "$MODEL_DIR"/* 2>/dev/null || true
