"""Per-layer divergence between braid's stack and HF's, on the same input.

If the residual grows smoothly with depth it is accumulated bf16 rounding. If it
steps at one layer, that layer is wrong — and the layer *type* at that index says
which sublayer to look at.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from braid.model.attention import RotaryEmbedding
from braid.model.config import ModelConfig
from braid.model.engine import Engine
from braid.model.loader import load_checkpoint
from braid.model.norm import rms_norm

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
DEV, DT = "cuda", torch.bfloat16


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a @ b / (a.norm() * b.norm())).item()


def main():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from test_full_forward import _load_hf_text_model  # noqa: E402

    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf  # noqa: F401

    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    tcfg = AutoConfig.from_pretrained(MODEL_DIR).get_text_config()
    tcfg._attn_implementation = "sdpa"

    ids = torch.tensor([[151, 9284, 501, 62, 8, 4410, 77, 1201, 33, 990,
                         12, 7788, 45, 2, 9001, 640]], device=DEV)
    T = ids.shape[1]

    dt = torch.float32 if "--fp32" in sys.argv else DT
    print(f"=== dtype {str(dt).split('.')[-1]} ===")

    hf_model = _load_hf_text_model(tcfg)
    if dt is torch.float32:
        hf_model = hf_model.float()
    with torch.no_grad():
        ref = hf_model(input_ids=ids, use_cache=False, output_hidden_states=True)
    ref_hidden = [h.float() for h in ref.hidden_states]  # embeddings + each layer
    ref_final = ref.last_hidden_state.float()
    # HF's LAST hidden_states entry is POST-final-norm, not the last layer's
    # output, so index 32 is not comparable to layer 31. Confirm and drop it.
    post_norm = metrics(ref_hidden[-1], ref_final)[0] < 1e-9
    print(f"  (hf hidden_states[-1] is post-final-norm: {post_norm})")
    if post_norm:
        ref_hidden = ref_hidden[:-1]
    del hf_model
    torch.cuda.empty_cache()

    ck = load_checkpoint(MODEL_DIR, device=DEV)
    if dt is torch.float32:
        ck.tensors = {k: (v.float() if v.is_floating_point() else v)
                      for k, v in ck.tensors.items()}
    eng = Engine.from_checkpoint(ck, device=DEV, dtype=dt)
    with torch.no_grad():
        h = torch.nn.functional.embedding(ids, eng.embed_tokens)
        pos = torch.arange(T, device=DEV)[None]
        cos, sin = RotaryEmbedding(cfg, DEV, dt)(pos)
        print(f"  embed              rel_l2={metrics(h.float(), ref_hidden[0])[0]:.3e}")
        for i, layer in enumerate(eng.layers):
            h = layer(h, cos, sin, cache=None)
            if i + 1 < len(ref_hidden):
                r, c = metrics(h.float(), ref_hidden[i + 1])
                kind = "gdn " if cfg.is_gdn(i) else "attn"
                print(f"  layer {i:2d} [{kind}]     rel_l2={r:.3e}  cosine={c:.9f}")
        hn = rms_norm(h, eng.final_norm, cfg.rms_norm_eps)
        r, c = metrics(hn.float(), ref_final)
        print(f"  final norm         rel_l2={r:.3e}  cosine={c:.9f}")
        mine = torch.nn.functional.linear(hn, eng.lm_head)
        refl = torch.nn.functional.linear(ref_final.to(dt), eng.lm_head)
        r, c = metrics(mine, refl)
        print(f"  logits             rel_l2={r:.3e}  cosine={c:.9f}")
        agree = (mine.argmax(-1) == refl.argmax(-1)).float().mean().item()
        print(f"  greedy argmax agreement: {agree * 100:.1f}%")


if __name__ == "__main__":
    main()
