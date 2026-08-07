"""Is the GDN layer's residual against HF a real defect or scan-ordering noise?

Three discriminators:
  * per-token error — a conv-window bug is localised to t < conv_kernel; fp32
    reordering noise is flat across t.
  * against HF's own *recurrent* rule instead of its chunked one — same
    arithmetic order as braid, so any remaining gap is braid's.
  * error vs T — reordering noise is flat, an accumulating state bug grows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

from braid.model.config import ModelConfig
from braid.model.gdn import GatedDeltaNet
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
L, DEV, DT = 0, "cuda", torch.bfloat16


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a @ b / (a.norm() * b.norm())).item()


def build():
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    tcfg = AutoConfig.from_pretrained(MODEL_DIR).get_text_config()
    tcfg._attn_implementation = "sdpa"
    wm = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]

    def raw(n):
        k = f"model.language_model.layers.{L}.linear_attn.{n}"
        with safe_open(MODEL_DIR / wm[k], framework="pt") as f:
            return f.get_tensor(k)

    with torch.device("meta"):
        m = hf.Qwen3_5GatedDeltaNet(tcfg, layer_idx=L)
    # assign=True + per-tensor dtypes. Building the module in bf16 first and
    # then load_state_dict copies INTO bf16 params, truncating the tensors this
    # checkpoint stores as F32 -- `linear_attn.norm` moves 2.4e-3 and the
    # "reference" becomes a worse model than braid. That artefact accounted for
    # nearly all of the layer's apparent parity gap.
    sd = {}
    for n in ("conv1d.weight", "dt_bias", "A_log", "norm.weight", "out_proj.weight",
              "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
              "in_proj_a.weight"):
        t = raw(n)
        sd[n] = t.to(DEV, torch.float32 if t.dtype == torch.float32 else DT)
    m.load_state_dict(sd, assign=True)
    m.eval()

    ck = load_checkpoint(MODEL_DIR, device=DEV, layers=(L,), include_embeddings=False)
    return cfg, m, GatedDeltaNet(cfg, ck.layer(L))


def main():
    cfg, hf_gdn, mine = build()

    print("=== error vs T (HF chunked rule) ===")
    for T in (1, 2, 4, 8, 16, 24, 64, 128):
        g = torch.Generator(device=DEV).manual_seed(7)
        x = torch.randn(1, T, cfg.hidden_size, generator=g, device=DEV, dtype=DT)
        with torch.no_grad():
            r, c = metrics(mine(x), hf_gdn(x, cache_params=None, attention_mask=None))
        print(f"  T={T:4d}  rel_l2={r:.3e}  cosine={c:.9f}")

    T = 24
    g = torch.Generator(device=DEV).manual_seed(7)
    x = torch.randn(1, T, cfg.hidden_size, generator=g, device=DEV, dtype=DT)
    with torch.no_grad():
        a = mine(x)
        b = hf_gdn(x, cache_params=None, attention_mask=None)
    print(f"\n=== per-token error, T={T} (conv_kernel={cfg.gdn.conv_kernel}) ===")
    for t in range(T):
        r, c = metrics(a[:, t], b[:, t])
        print(f"  t={t:3d}  rel_l2={r:.3e}  cosine={c:.9f}")

    # Swap HF's chunked rule for its own recurrent one: same order as braid.
    def as_chunk(query, key, value, g, beta, initial_state, output_final_state,
                 use_qk_l2norm_in_kernel=False, cu_seqlens=None):
        return hf.torch_recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    hf_gdn.chunk_gated_delta_rule = as_chunk
    print("\n=== vs HF's RECURRENT rule (same arithmetic order as braid) ===")
    for T in (1, 8, 24, 128):
        g2 = torch.Generator(device=DEV).manual_seed(7)
        xx = torch.randn(1, T, cfg.hidden_size, generator=g2, device=DEV, dtype=DT)
        with torch.no_grad():
            r, c = metrics(mine(xx), hf_gdn(xx, cache_params=None, attention_mask=None))
        print(f"  T={T:4d}  rel_l2={r:.3e}  cosine={c:.9f}")

    # And how far apart are HF's OWN two rules? That is the noise floor braid
    # is being measured against.
    print("\n=== HF chunked vs HF recurrent (HF against itself) ===")
    for T in (8, 24, 128):
        g3 = torch.Generator(device=DEV).manual_seed(7)
        xx = torch.randn(1, T, cfg.hidden_size, generator=g3, device=DEV, dtype=DT)
        with torch.no_grad():
            hf_gdn.chunk_gated_delta_rule = as_chunk
            rec = hf_gdn(xx, cache_params=None, attention_mask=None)
            hf_gdn.chunk_gated_delta_rule = hf.torch_chunk_gated_delta_rule
            chunk = hf_gdn(xx, cache_params=None, attention_mask=None)
        r, c = metrics(rec, chunk)
        print(f"  T={T:4d}  rel_l2={r:.3e}  cosine={c:.9f}")


if __name__ == "__main__":
    main()
