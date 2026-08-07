"""Is decode != prefill a cache bug or bf16 accumulation?

Discriminators, per sublayer so the blame is unambiguous:
  * fp32 vs bf16 — a cache bug survives the dtype change, rounding does not.
  * per-token — a conv-window or KV-offset bug is localised; noise is flat.
  * one layer vs 32 — accumulation grows with depth, a bug does not need depth.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from braid.model.attention import Attention, RotaryEmbedding
from braid.model.cache import KVCache, RecurrentCache
from braid.model.config import ModelConfig
from braid.model.gdn import GatedDeltaNet
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
DEV = "cuda"
GDN_L, ATTN_L = 0, 3


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a @ b / (a.norm() * b.norm())).item()


def cast(layer, dt):
    return {k: (v.to(dt) if v.is_floating_point() and v.dtype != torch.float32 else v)
            for k, v in layer.items()}


def main():
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    ck = load_checkpoint(MODEL_DIR, device=DEV, layers=(GDN_L, ATTN_L),
                         include_embeddings=False)
    T = 8

    for dt in (torch.bfloat16, torch.float32):
        name = str(dt).split(".")[-1]
        g = torch.Generator(device=DEV).manual_seed(4)
        x = torch.randn(1, T, cfg.hidden_size, generator=g, device=DEV, dtype=dt)
        print(f"\n=== {name} ===")

        # --- GDN ---
        gdn = GatedDeltaNet(cfg, cast(ck.layer(GDN_L), dt))
        with torch.no_grad():
            bulk = gdn(x)
            c = RecurrentCache(cfg, 1, DEV, dt)
            step = torch.cat([gdn(x[:, t:t + 1], cache=c) for t in range(T)], dim=1)
        r, co = metrics(step, bulk)
        print(f"  gdn  layer {GDN_L}: rel_l2={r:.3e} cosine={co:.9f}")
        for t in range(T):
            rt, _ = metrics(step[:, t], bulk[:, t])
            print(f"      t={t}: {rt:.3e}")

        # --- attention ---
        attn = Attention(cfg, cast(ck.layer(ATTN_L), dt))
        rope = RotaryEmbedding(cfg, DEV, dt)
        pos = torch.arange(T, device=DEV)[None]
        cos, sin = rope(pos)
        with torch.no_grad():
            bulk_a = attn(x, cos, sin)
            kv = KVCache(cfg, 1, T + 2, DEV, dt)
            steps = []
            for t in range(T):
                ct, st = rope(pos[:, t:t + 1])
                steps.append(attn(x[:, t:t + 1], ct, st, cache=kv))
            step_a = torch.cat(steps, dim=1)
        r, co = metrics(step_a, bulk_a)
        print(f"  attn layer {ATTN_L}: rel_l2={r:.3e} cosine={co:.9f}")
        for t in range(T):
            rt, _ = metrics(step_a[:, t], bulk_a[:, t])
            print(f"      t={t}: {rt:.3e}")


if __name__ == "__main__":
    main()
