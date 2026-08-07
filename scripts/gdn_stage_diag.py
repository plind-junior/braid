"""Stage-by-stage divergence between braid's GDN layer and HF's, at T=1.

Both pipelines are written out inline from the two sources so every intermediate
is visible. The first stage whose relative error jumps is the defect; everything
downstream of it inherits.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

from braid.model.config import ModelConfig
from braid.model.loader import load_checkpoint
from braid.model.norm import rms_norm_gated
from braid.reference.gdn_ref import _l2norm, gdn_decode_vectorized

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
L, DEV, DT = 0, "cuda", torch.bfloat16
T = 1


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    n = b.norm()
    return ((a - b).norm() / n).item(), (a @ b / (a.norm() * n)).item()


def cmp(label, mine, ref):
    r, c = metrics(mine, ref)
    flag = "  <-- DIVERGES" if r > 1e-4 else ""
    print(f"  {label:<28s} rel_l2={r:.3e}  cosine={c:.9f}  "
          f"[{str(mine.dtype).split('.')[-1]} vs {str(ref.dtype).split('.')[-1]}]{flag}")


def main():
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    g = cfg.gdn
    tcfg = AutoConfig.from_pretrained(MODEL_DIR).get_text_config()
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
    w = ck.layer(L)

    gen = torch.Generator(device=DEV).manual_seed(7)
    x = torch.randn(1, T, cfg.hidden_size, generator=gen, device=DEV, dtype=DT)

    with torch.no_grad():
        # ---------------- HF ----------------
        h_qkv = m.in_proj_qkv(x).transpose(1, 2)
        h_z = m.in_proj_z(x).reshape(1, T, -1, m.head_v_dim)
        h_b = m.in_proj_b(x)
        h_a = m.in_proj_a(x)
        h_conv = F.silu(m.conv1d(h_qkv)[:, :, :h_qkv.shape[-1]]).transpose(1, 2)
        h_q, h_k, h_v = torch.split(h_conv, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        h_q = h_q.reshape(1, T, -1, m.head_k_dim)
        h_k = h_k.reshape(1, T, -1, m.head_k_dim)
        h_v = h_v.reshape(1, T, -1, m.head_v_dim)
        h_beta = h_b.sigmoid()
        h_g = -m.A_log.float().exp() * F.softplus(h_a.float() + m.dt_bias)
        h_qr = h_q.repeat_interleave(2, dim=2)
        h_kr = h_k.repeat_interleave(2, dim=2)
        h_core, _ = hf.torch_recurrent_gated_delta_rule(
            h_qr, h_kr, h_v, g=h_g, beta=h_beta, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True)
        h_normed = m.norm(h_core.reshape(-1, m.head_v_dim), h_z.reshape(-1, m.head_v_dim))
        h_out = m.out_proj(h_normed.reshape(1, T, -1))

        # ---------------- braid ----------------
        b_qkv = F.linear(x, w["linear_attn.in_proj_qkv"]).transpose(1, 2)
        b_conv_raw = F.conv1d(b_qkv, w["linear_attn.conv1d"].unsqueeze(1), None,
                              padding=g.conv_kernel - 1, groups=g.conv_channels)
        b_conv = F.silu(b_conv_raw[:, :, :T]).transpose(1, 2)
        key_dim = g.n_groups * g.state_size
        b_q, b_k, b_v = torch.split(b_conv, [key_dim, key_dim, g.inner_size], dim=-1)
        b_q = b_q.reshape(1, T, g.n_groups, g.state_size)
        b_k = b_k.reshape(1, T, g.n_groups, g.state_size)
        b_v = b_v.reshape(1, T, g.n_heads, g.head_dim)
        b_a = F.linear(x, w["linear_attn.in_proj_a"])
        b_b = F.linear(x, w["linear_attn.in_proj_b"])
        b_beta = torch.sigmoid(b_b).float()
        b_alpha = torch.exp(w["linear_attn.A"] * F.softplus(b_a.float() + w["linear_attn.dt_bias"]))
        b_qn, b_kn = _l2norm(b_q).float(), _l2norm(b_k).float()
        state = torch.zeros(1, g.n_heads, g.state_size, g.head_dim, device=DEV, dtype=torch.float32)
        b_core = gdn_decode_vectorized(state=state, q=b_qn[:, 0], k=b_kn[:, 0],
                                       v=b_v[:, 0].float(), alpha=b_alpha[:, 0],
                                       beta=b_beta[:, 0], cfg=g, normalize=False)[:, None]
        b_z = F.linear(x, w["linear_attn.in_proj_z"]).reshape(-1, g.head_dim)
        b_normed = rms_norm_gated(b_core.to(DT).reshape(-1, g.head_dim), b_z,
                                  w["linear_attn.norm"], cfg.rms_norm_eps)
        b_out = F.linear(b_normed.reshape(1, T, g.inner_size), w["linear_attn.out_proj"])

    print(f"=== stage-by-stage, T={T} ===")
    cmp("in_proj_qkv", b_qkv, h_qkv)
    cmp("conv+silu", b_conv, h_conv)
    cmp("q (split+reshape)", b_q, h_q)
    cmp("k", b_k, h_k)
    cmp("v", b_v, h_v)
    cmp("beta", b_beta, h_beta.float())
    cmp("decay (alpha vs exp(g))", b_alpha, h_g.float().exp())
    cmp("l2norm(q)", b_qn.repeat_interleave(2, dim=2), _l2norm(h_qr).float())
    cmp("l2norm(k)", b_kn.repeat_interleave(2, dim=2), _l2norm(h_kr).float())
    cmp("scan core_attn_out", b_core.float(), h_core.float())
    cmp("in_proj_z", b_z, h_z.reshape(-1, m.head_v_dim))
    cmp("gated norm", b_normed, h_normed)
    cmp("out_proj (layer output)", b_out, h_out)


if __name__ == "__main__":
    main()
