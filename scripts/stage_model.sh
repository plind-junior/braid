#!/usr/bin/env bash
# Stage a target model: the GGUF llama.cpp will run, and the HF safetensors
# braid will run. **Both must be the SAME model or the comparison is void**,
# which is why one script fetches both and they are named from one variable.
#
# Defaults to Qwen3.5-9B. The 4B remains one env var away and is still the
# correctness target where fp32 gates have to fit on the card:
#
#     MODEL=Qwen3.5-4B bash scripts/stage_model.sh
#
# Why the 9B is the serving target: it is the smallest member of this family
# published on OpenRouter, and its GDN block is dimensionally IDENTICAL to the
# 4B's (32 value heads / 16 key heads / 128 / 128, conv 4, 24 GDN layers of 32),
# so the decode kernel, the conv kernel and the graph buckets all transfer
# untouched. What differs is outside the scan: hidden 4096 vs 2560, MLP 12288 vs
# 9216, and **untied embeddings** -- the 9B ships a real `lm_head.weight` at the
# top level of the safetensors index rather than under the text tower.
#
# Both are vision-language checkpoints and braid runs the text tower only; the
# GGUF contains the text tower only too, which is what keeps the arms matched.
set -euo pipefail
MODEL=${MODEL:-Qwen3.5-9B}
QUANT=${QUANT:-Q8_0}
MODEL_DIR=${MODEL_DIR:-/root/models}
mkdir -p "$MODEL_DIR"

MODEL="$MODEL" QUANT="$QUANT" MODEL_DIR="$MODEL_DIR" python3 - <<'PY'
import os

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

model = os.environ["MODEL"]
quant = os.environ["QUANT"]
model_dir = os.environ["MODEL_DIR"]

# GGUF for llama.cpp. unsloth for every member of the family, deliberately: a
# head-to-head whose two arms came from different conversion pipelines is
# arguing about the converter, not the engine.
repo = f"unsloth/{model}-GGUF"
fname = f"{model}-{quant}.gguf"
try:
    print("GGUF:", hf_hub_download(repo_id=repo, filename=fname, local_dir=model_dir))
except Exception as e:
    print(f"  miss {repo}/{fname}: {type(e).__name__}: {str(e)[:160]}")
    try:
        files = [f for f in list_repo_files(repo) if f.endswith(".gguf")]
        print(f"available in {repo}:", files[:24])
    except Exception as e2:
        print(f"  cannot list {repo}: {type(e2).__name__}: {str(e2)[:160]}")

# HF safetensors for braid. `*.safetensors` pulls the visual tower too -- the
# index is a single manifest and there is no per-prefix download -- but the
# loader filters it and never faults those pages in, so the cost is disk, not
# VRAM.
try:
    print("HF:", snapshot_download(
        repo_id=f"Qwen/{model}",
        local_dir=os.path.join(model_dir, model),
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json",
                        "*.txt", "*.py"],
    ))
except Exception as e:
    print(f"  miss Qwen/{model}: {type(e).__name__}: {str(e)[:200]}")
PY

echo "=== staged ==="
du -sh "$MODEL_DIR"/* 2>/dev/null || true
df -h /root | tail -1
