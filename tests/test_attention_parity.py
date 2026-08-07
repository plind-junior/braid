"""Single-layer parity for the conventional half: attention and MLP vs HF.

`transformers` ships the implementation the checkpoint was trained with, so it
is ground truth — the same standard `tests/test_hf_parity.py` holds the GDN half
to. This is ROADMAP Phase 2 item 2 for the non-recurrent sublayers.

**Gate: rel L2 <= 5e-3 and cosine >= 0.99999**, on real checkpoint weights.

Two arms, each compared against the HF attention implementation that shares its
numerics, because mixing them measures the wrong thing:

  fp32 vs HF **eager** — the literal reference implementation. Bit-exact
      (rel L2 0.0), so this arm proves the *algorithm* with no tolerance to hide
      behind: the per-head [q|gate] split, partial rope, the 1+W fold.
  bf16 vs HF **sdpa** — the deployment dtype, apples to apples. 1.2e-4 / 0.9999999_92.

braid-bf16 against HF-*eager* measures 4.4e-3 / 0.999990, which technically
clears the gate but with ~4e-7 of margin on the cosine. That number is not
braid: HF eager takes its softmax in fp32 and casts the weights back to bf16,
SDPA does not. Gating on it would be gating on a numerical schedule mismatch,
and it would flake on the next GPU or sequence length.

The HF modules are handed the **raw** checkpoint tensors and braid is handed the
loader's transformed ones. That asymmetry is the point: if `loader.py` dropped
the `1 + W` fold on `q_norm`/`k_norm`, the two would disagree here — by
rel L2 0.78, measured in `scripts/parity_report.py`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from braid.model.attention import Attention, RotaryEmbedding
from braid.model.config import ModelConfig
from braid.model.loader import load_checkpoint
from braid.model.mlp import MLP

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
safetensors = pytest.importorskip("safetensors")

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

ATTN_LAYER = 3   # first `full_attention` layer
GDN_LAYER = 0    # a `linear_attention` layer — its MLP is identical in shape

REL_L2_MAX = 5e-3
COSINE_MIN = 0.99999


# --- helpers ----------------------------------------------------------------

def _metrics(mine: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    a, b = mine.double().flatten(), ref.double().flatten()
    rel_l2 = ((a - b).norm() / b.norm()).item()
    cos = (a @ b / (a.norm() * b.norm())).item()
    return rel_l2, cos


def _assert_parity(mine: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    rel_l2, cos = _metrics(mine, ref)
    assert rel_l2 <= REL_L2_MAX and cos >= COSINE_MIN, (
        f"{what}: rel_l2={rel_l2:.3e} (max {REL_L2_MAX:.0e}), "
        f"cosine={cos:.9f} (min {COSINE_MIN})"
    )


def _raw(layer: int, names: list[str]) -> dict[str, torch.Tensor]:
    """Untransformed tensors straight out of the shards, for the HF modules."""
    import json

    from safetensors import safe_open

    weight_map = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]
    out: dict[str, torch.Tensor] = {}
    for n in names:
        key = f"model.language_model.layers.{layer}.{n}"
        with safe_open(MODEL_DIR / weight_map[key], framework="pt") as f:
            out[n] = f.get_tensor(key)
    return out


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return ModelConfig.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def ckpt():
    # Two layers, not 8.8 GB — the parity gate does not need the embedding table.
    return load_checkpoint(
        MODEL_DIR, device="cuda", layers=(GDN_LAYER, ATTN_LAYER), include_embeddings=False
    )


# (dtype, HF attention implementation to compare against) — see the module docstring.
ARMS = [
    pytest.param(torch.float32, "eager", id="fp32-eager"),
    pytest.param(torch.bfloat16, "sdpa", id="bf16-sdpa"),
]


@pytest.fixture(scope="module")
def hf_text_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(MODEL_DIR).get_text_config()


def _hf_config_for(hf_text_config, impl: str):
    import copy

    c = copy.deepcopy(hf_text_config)
    c._attn_implementation = impl
    return c


# --- rope --------------------------------------------------------------------

def test_rope_matches_hf(cfg, hf_text_config):
    """braid skips MRoPE's three-grid interleave; prove that is a no-op for text."""
    dev = "cuda"
    dtype = torch.bfloat16
    B, T = 2, 24
    pos = torch.arange(T, device=dev)[None].expand(B, T)

    hf_rope = hf.Qwen3_5TextRotaryEmbedding(_hf_config_for(hf_text_config, "eager"), device=dev)
    probe = torch.zeros(B, T, cfg.hidden_size, device=dev, dtype=dtype)
    cos_hf, sin_hf = hf_rope(probe, pos)

    cos, sin = RotaryEmbedding(cfg, dev, dtype)(pos)

    assert cos.shape == cos_hf.shape == (B, T, cfg.rotary_dim), (
        f"{cos.shape} vs {cos_hf.shape}, rotary_dim={cfg.rotary_dim}"
    )
    torch.testing.assert_close(cos, cos_hf, rtol=0, atol=0)
    torch.testing.assert_close(sin, sin_hf, rtol=0, atol=0)


def test_rope_is_partial(cfg):
    """64 of 256 dims rotate. Rotating all 256 is not a shape error anywhere."""
    assert cfg.rotary_dim == 64 and cfg.head_dim == 256
    dev = "cuda"
    cos, sin = RotaryEmbedding(cfg, dev, torch.float32)(
        torch.arange(8, device=dev)[None]
    )
    from braid.model.attention import apply_rotary_pos_emb

    q = torch.randn(1, 2, 8, cfg.head_dim, device=dev)
    q_out, _ = apply_rotary_pos_emb(q, q.clone(), cos, sin)
    torch.testing.assert_close(q_out[..., cfg.rotary_dim:], q[..., cfg.rotary_dim:])
    assert not torch.allclose(q_out[..., : cfg.rotary_dim], q[..., : cfg.rotary_dim])


# --- attention ----------------------------------------------------------------

@pytest.mark.parametrize("dtype,impl", ARMS)
def test_attention_layer_matches_hf(cfg, ckpt, hf_text_config, dtype, impl):
    """One `full_attention` layer, real weights, prefill over T=32."""
    dev = "cuda"
    B, T = 2, 32

    names = [f"self_attn.{n}.weight" for n in
             ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
    raw = _raw(ATTN_LAYER, names)
    hf_cfg = _hf_config_for(hf_text_config, impl)
    hf_attn = hf.Qwen3_5Attention(hf_cfg, layer_idx=ATTN_LAYER).to(dev, dtype)
    hf_attn.load_state_dict({k[len("self_attn."):]: v.to(dev, dtype) for k, v in raw.items()})
    hf_attn.eval()

    g = torch.Generator(device=dev).manual_seed(11)
    x = torch.randn(B, T, cfg.hidden_size, generator=g, device=dev, dtype=dtype)
    pos = torch.arange(T, device=dev)[None].expand(B, T)
    cos, sin = RotaryEmbedding(cfg, dev, dtype)(pos)

    # HF's eager path adds the mask or attends bidirectionally; be explicit.
    causal = torch.full((T, T), torch.finfo(dtype).min, device=dev, dtype=dtype).triu(1)
    with torch.no_grad():
        ref, _ = hf_attn(x, position_embeddings=(cos, sin),
                         attention_mask=causal[None, None])

    layer = {k: v.to(dtype) if v.is_floating_point() and v.dim() > 1 else v
             for k, v in ckpt.layer(ATTN_LAYER).items()}
    with torch.no_grad():
        mine = Attention(cfg, layer)(x, cos, sin)

    assert mine.shape == ref.shape
    _assert_parity(mine, ref, f"attention[{ATTN_LAYER}] {dtype} vs HF {impl}")


def test_attention_decode_step_matches_prefill_tail(cfg, ckpt):
    """T=1 with a prepared KV prefix must equal position T-1 of the full pass.

    Cheap, and it catches the `is_causal` shortcut silently masking the single
    decode query against itself only.
    """
    dev, dtype = "cuda", torch.float32
    B, T = 2, 16
    layer = {k: v.to(dtype) if v.is_floating_point() and v.dim() > 1 else v
             for k, v in ckpt.layer(ATTN_LAYER).items()}
    attn = Attention(cfg, layer)

    g = torch.Generator(device=dev).manual_seed(3)
    x = torch.randn(B, T, cfg.hidden_size, generator=g, device=dev, dtype=dtype)
    pos = torch.arange(T, device=dev)[None].expand(B, T)
    cos, sin = RotaryEmbedding(cfg, dev, dtype)(pos)

    with torch.no_grad():
        full = attn(x, cos, sin)
        # A T=1 forward whose mask lets the query see the whole prefix is the
        # same arithmetic as the last row of the causal pass.
        last = attn(x, cos, sin, attn_mask=torch.zeros(T, T, device=dev, dtype=dtype)
                    .masked_fill(torch.ones(T, T, device=dev).triu(1).bool(),
                                 torch.finfo(dtype).min)[None, None])
    _assert_parity(last[:, -1], full[:, -1], "decode tail")


def test_gate_split_is_per_head_not_halves(cfg, ckpt):
    """The [q|gate] split discriminates. If it did not, the gate is meaningless."""
    dev, dtype = "cuda", torch.float32
    B, T, H, D = 1, 8, cfg.num_attention_heads, cfg.head_dim
    layer = ckpt.layer(ATTN_LAYER)
    w = layer["self_attn.q_proj"].to(dtype)
    x = torch.randn(B, T, cfg.hidden_size, device=dev, dtype=dtype)

    proj = torch.nn.functional.linear(x, w)
    per_head_q = proj.view(B, T, H, 2 * D).chunk(2, dim=-1)[0].reshape(B, T, H * D)
    flat_half_q = proj[..., : H * D]

    rel_l2, _ = _metrics(flat_half_q, per_head_q)
    assert rel_l2 > 0.5, (
        "per-head and flat-half q splits agree, so this checkpoint cannot "
        "distinguish them and the parity test above proves less than it claims"
    )


# --- mlp ----------------------------------------------------------------------

@pytest.mark.parametrize("dtype,impl", ARMS)
def test_mlp_layer_matches_hf(cfg, ckpt, hf_text_config, dtype, impl):
    dev = "cuda"
    B, T = 2, 32

    names = [f"mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")]
    raw = _raw(GDN_LAYER, names)
    hf_mlp = hf.Qwen3_5MLP(_hf_config_for(hf_text_config, impl),
                           cfg.intermediate_size).to(dev, dtype)
    hf_mlp.load_state_dict({k[len("mlp."):]: v.to(dev, dtype) for k, v in raw.items()})
    hf_mlp.eval()

    g = torch.Generator(device=dev).manual_seed(23)
    x = torch.randn(B, T, cfg.hidden_size, generator=g, device=dev, dtype=dtype)

    with torch.no_grad():
        ref = hf_mlp(x)
        layer = {k: v.to(dtype) if v.is_floating_point() and v.dim() > 1 else v
                 for k, v in ckpt.layer(GDN_LAYER).items()}
        mine = MLP(cfg, layer)(x)

    assert mine.shape == ref.shape
    _assert_parity(mine, ref, f"mlp[{GDN_LAYER}] {dtype}")


# --- norms --------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_rms_norm_matches_hf_with_folded_gamma(cfg, ckpt, dtype):  # noqa: D401
    """The loader's fp32 `1 + W` fold must be exact against HF's runtime offset."""
    from braid.model.norm import rms_norm

    dev = "cuda"
    raw = _raw(ATTN_LAYER, ["input_layernorm.weight"])["input_layernorm.weight"]
    hf_norm = hf.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev, dtype)
    hf_norm.load_state_dict({"weight": raw.to(dev, dtype)})

    x = torch.randn(4, cfg.hidden_size, device=dev, dtype=dtype)
    with torch.no_grad():
        ref = hf_norm(x)
        mine = rms_norm(x, ckpt[f"layers.{ATTN_LAYER}.input_layernorm"], cfg.rms_norm_eps)
    _assert_parity(mine, ref, f"input_layernorm {dtype}")
