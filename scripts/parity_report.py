"""Print the parity margins behind `tests/test_attention_parity.py`, and prove
the gate discriminates by re-running it against deliberately broken variants.

A passing tolerance is only evidence if the same tolerance fails when the thing
it guards is wrong. Each ablation below is a mistake this loader/module pair
could plausibly have made.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

from braid.model.attention import Attention, RotaryEmbedding, apply_rotary_pos_emb
from braid.model.config import ModelConfig
from braid.model.loader import load_checkpoint
from braid.model.mlp import MLP
from braid.model.norm import rms_norm

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
ATTN_LAYER, GDN_LAYER = 3, 0
REL_L2_MAX, COSINE_MIN = 5e-3, 0.99999
DEV = "cuda"


def metrics(mine, ref):
    a, b = mine.double().flatten(), ref.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a @ b / (a.norm() * b.norm())).item()


def row(label, mine, ref):
    r, c = metrics(mine, ref)
    ok = "PASS" if (r <= REL_L2_MAX and c >= COSINE_MIN) else "FAIL"
    print(f"  {ok}  {label:<44s} rel_l2={r:.3e}  cosine={c:.9f}")
    return ok == "PASS"


def raw(layer, names):
    import json

    from safetensors import safe_open

    wm = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for n in names:
        k = f"model.language_model.layers.{layer}.{n}"
        with safe_open(MODEL_DIR / wm[k], framework="pt") as f:
            out[n] = f.get_tensor(k)
    return out


def main():
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    ck = load_checkpoint(MODEL_DIR, device=DEV, layers=(GDN_LAYER, ATTN_LAYER),
                         include_embeddings=False)
    print(ck.report.summary(), "\n")

    import copy

    base = AutoConfig.from_pretrained(MODEL_DIR).get_text_config()

    def tcfg_for(impl):
        c = copy.deepcopy(base)
        c._attn_implementation = impl
        return c

    B, T = 2, 32
    # fp32 is compared against HF eager (the literal reference); bf16 against HF
    # sdpa (matching numerics). See tests/test_attention_parity.py.
    for dtype, impl in ((torch.float32, "eager"), (torch.bfloat16, "sdpa"),
                        (torch.bfloat16, "eager")):
        tcfg = tcfg_for(impl)
        name = f"{str(dtype).split('.')[-1]}/hf-{impl}"
        g = torch.Generator(device=DEV).manual_seed(11)
        x = torch.randn(B, T, cfg.hidden_size, generator=g, device=DEV, dtype=dtype)
        pos = torch.arange(T, device=DEV)[None].expand(B, T)
        cos, sin = RotaryEmbedding(cfg, DEV, dtype)(pos)
        causal = torch.full((T, T), torch.finfo(dtype).min, device=DEV, dtype=dtype).triu(1)

        an = [f"self_attn.{n}.weight" for n in
              ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        hfa = hf.Qwen3_5Attention(tcfg, layer_idx=ATTN_LAYER).to(DEV, dtype)
        hfa.load_state_dict({k[len("self_attn."):]: v.to(DEV, dtype)
                             for k, v in raw(ATTN_LAYER, an).items()})
        hfa.eval()
        layer = {k: v.to(dtype) if v.is_floating_point() and v.dim() > 1 else v
                 for k, v in ck.layer(ATTN_LAYER).items()}

        with torch.no_grad():
            ref, _ = hfa(x, position_embeddings=(cos, sin), attention_mask=causal[None, None])
            print(f"[{name}] attention")
            note = ("  <- INFORMATIONAL: HF eager takes softmax in fp32, SDPA does "
                    "not; this gap is a numerical schedule mismatch, not braid"
                    if (dtype is torch.bfloat16 and impl == "eager") else "")
            row("braid", Attention(cfg, layer)(x, cos, sin), ref)
            if note:
                print(note)

            if dtype is torch.float32:
                H, D = cfg.num_attention_heads, cfg.head_dim
                print("       ablations (each MUST fail)")

                # 1. gate split as flat halves instead of per head
                def flat_half(xx):
                    proj = F.linear(xx, layer["self_attn.q_proj"])
                    q, gate = proj[..., :H * D], proj[..., H * D:]
                    return q.view(B, T, H, D), gate

                row("[q|gate] split as flat halves",
                    _attn_variant(cfg, layer, x, cos, sin, split=flat_half), ref)

                # 2. rope over the full 256 dims instead of the first 64
                full_cos, full_sin = _full_rope(cfg, DEV, dtype, pos)
                row("rope over all 256 dims (not 64)",
                    Attention(cfg, layer)(x, full_cos, full_sin), ref)

                # 3. q/k norm without the 1+W fold
                bad = dict(layer)
                bad["self_attn.q_norm"] = layer["self_attn.q_norm"] - 1.0
                bad["self_attn.k_norm"] = layer["self_attn.k_norm"] - 1.0
                row("q/k RMSNorm missing the 1+W offset",
                    Attention(cfg, bad)(x, cos, sin), ref)

                # 4. head_dim inferred as hidden//n_heads = 160
                _report_head_dim_160(cfg, layer, x)

        mn = [f"mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")]
        hfm = hf.Qwen3_5MLP(tcfg, cfg.intermediate_size).to(DEV, dtype)
        hfm.load_state_dict({k[len("mlp."):]: v.to(DEV, dtype)
                             for k, v in raw(GDN_LAYER, mn).items()})
        hfm.eval()
        mlayer = {k: v.to(dtype) if v.is_floating_point() and v.dim() > 1 else v
                  for k, v in ck.layer(GDN_LAYER).items()}
        with torch.no_grad():
            g2 = torch.Generator(device=DEV).manual_seed(23)
            xm = torch.randn(B, T, cfg.hidden_size, generator=g2, device=DEV, dtype=dtype)
            print(f"[{name}] mlp")
            row("braid", MLP(cfg, mlayer)(xm), hfm(xm))

        # final RMSNorm offset — the 13.65 -> 6.82 PPL bug
        if dtype is torch.float32:
            w = raw(ATTN_LAYER, ["input_layernorm.weight"])["input_layernorm.weight"]
            hn = hf.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(DEV, dtype)
            hn.load_state_dict({"weight": w.to(DEV, dtype)})
            xr = torch.randn(4, cfg.hidden_size, device=DEV, dtype=dtype)
            with torch.no_grad():
                print("[fp32] rmsnorm")
                gam = ck[f"layers.{ATTN_LAYER}.input_layernorm"]
                row("braid (fp32 1+W fold)", rms_norm(xr, gam, cfg.rms_norm_eps), hn(xr))
                print("       ablations (each MUST fail)")
                row("1+W folded in bf16", rms_norm(xr, gam.bfloat16().float(),
                                                   cfg.rms_norm_eps), hn(xr))
                row("no 1+W offset at all", rms_norm(xr, gam - 1.0,
                                                     cfg.rms_norm_eps), hn(xr))
        print()


def _attn_variant(cfg, layer, x, cos, sin, split):
    B, T, _ = x.shape
    H, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q, gate = split(x)
    q = rms_norm(q, layer["self_attn.q_norm"], cfg.rms_norm_eps).transpose(1, 2)
    k = rms_norm(F.linear(x, layer["self_attn.k_proj"]).view(B, T, KVH, D),
                 layer["self_attn.k_norm"], cfg.rms_norm_eps).transpose(1, 2)
    v = F.linear(x, layer["self_attn.v_proj"]).view(B, T, KVH, D).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                       scale=cfg.attention_scaling, enable_gqa=True)
    o = o.transpose(1, 2).reshape(B, T, H * D) * torch.sigmoid(gate)
    return F.linear(o, layer["self_attn.o_proj"])


def _full_rope(cfg, dev, dtype, pos):
    dim = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    freqs = pos.float()[..., None] * inv.to(dev)
    emb = torch.cat((freqs, freqs), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _report_head_dim_160(cfg, layer, x):
    """What `hidden_size // num_attention_heads` actually does on this checkpoint.

    It gives 160, and 8192 % 320 != 0, so the very first q reshape raises. The
    trap is real but LOUD here — it is the `q_proj_out` and `o_proj` widths that
    happen to be indivisible. A sibling checkpoint whose widths did divide would
    get the silent version, which is why `from_dict` refuses to infer at all
    rather than relying on a divisibility accident.
    """
    B, T, _ = x.shape
    D = cfg.hidden_size // cfg.num_attention_heads  # 160
    try:
        F.linear(x, layer["self_attn.q_proj"]).view(B, T, -1, 2 * D)
    except RuntimeError as e:
        print(f"  PASS  {'head_dim inferred as 2560//16 = 160':<44s} raises: {e}")
        return
    raise AssertionError("inferred head_dim reshaped cleanly; the trap is now silent here")


if __name__ == "__main__":
    main()
