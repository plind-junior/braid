"""Config parsing and checkpoint loading — the traps that are silent, not loud.

Two of these need no GPU and no checkpoint; they run anywhere.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from braid.config import GDNConfig
from braid.model.config import ModelConfig
from braid.model.loader import MTP_PREFIX, VISUAL_PREFIX, braid_name

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
needs_ckpt = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}")


# --- config, no checkpoint required -------------------------------------------

MINIMAL = {
    "text_config": {
        "hidden_size": 2560, "intermediate_size": 9216, "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
        "attn_output_gate": True, "attention_bias": False,
        "rms_norm_eps": 1e-6, "vocab_size": 248320, "tie_word_embeddings": True,
        "hidden_act": "silu", "dtype": "bfloat16", "mlp_only_layers": [],
        "linear_num_value_heads": 32, "linear_num_key_heads": 16,
        "linear_value_head_dim": 128, "linear_key_head_dim": 128,
        "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25,
                            "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    },
    "tie_word_embeddings": True,
    "vision_config": {"hidden_size": 1024, "depth": 24},
}


def test_head_dim_is_never_inferred():
    """16 x 256 = 4096 != 2560, and 2560 // 16 = 160 reshapes without complaint."""
    raw = copy.deepcopy(MINIMAL)
    del raw["text_config"]["head_dim"]
    with pytest.raises(ValueError, match="no explicit head_dim"):
        ModelConfig.from_dict(raw)

    cfg = ModelConfig.from_dict(MINIMAL)
    assert cfg.head_dim == 256
    assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads
    assert cfg.q_dim == 4096 != cfg.hidden_size
    assert cfg.q_proj_out == 8192  # attn_output_gate doubles it
    assert cfg.kv_dim == 1024
    assert cfg.rotary_dim == 64


def test_config_reads_text_tower_not_vision():
    cfg = ModelConfig.from_dict(MINIMAL)
    assert cfg.hidden_size == 2560, "read the visual tower's 1024"


def test_layer_schedule():
    cfg = ModelConfig.from_dict(MINIMAL)
    assert cfg.gdn_layers == (0, 1, 2) and cfg.attention_layers == (3,)
    assert cfg.is_gdn(0) and not cfg.is_gdn(3)

    bad = copy.deepcopy(MINIMAL)
    bad["text_config"]["layer_types"] = ["linear_attention"] * 3
    with pytest.raises(ValueError, match="layer_types has 3 entries"):
        ModelConfig.from_dict(bad)


def test_mrope_section_must_match_rotary_dim():
    bad = copy.deepcopy(MINIMAL)
    bad["text_config"]["rope_parameters"]["mrope_section"] = [11, 11, 11]
    with pytest.raises(ValueError, match="mrope_section"):
        ModelConfig.from_dict(bad)


def test_name_mapping_filters_the_non_text_towers():
    assert braid_name(VISUAL_PREFIX + "blocks.0.attn.qkv.weight") is None
    assert braid_name(MTP_PREFIX + "layers.0.mlp.up_proj.weight") is None
    assert braid_name("model.language_model.embed_tokens.weight") == "embed_tokens"
    assert braid_name("model.language_model.norm.weight") == "norm"
    assert (braid_name("model.language_model.layers.0.linear_attn.A_log")
            == "layers.0.linear_attn.A")
    assert (braid_name("model.language_model.layers.3.self_attn.q_proj.weight")
            == "layers.3.self_attn.q_proj")


# --- against the real checkpoint ------------------------------------------------

@needs_ckpt
def test_checkpoint_config_matches_the_pinned_gdn_shapes():
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    assert cfg.gdn == GDNConfig.qwen35_4b(), (
        f"braid/config.py's pinned shapes drifted from the checkpoint: "
        f"{cfg.gdn} vs {GDNConfig.qwen35_4b()}"
    )
    assert cfg.num_hidden_layers == 32
    assert len(cfg.gdn_layers) == 24 and len(cfg.attention_layers) == 8
    assert cfg.num_key_value_heads == 4, "ARCHITECTURE.md's 2 is the 35B, not this one"


@needs_ckpt
def test_loader_excludes_the_visual_tower_and_mtp_head():
    """The comparison is void if braid loads tensors the GGUF does not contain."""
    from braid.model.loader import load_checkpoint

    ck = load_checkpoint(MODEL_DIR, device="cpu", layers=(0, 3), include_embeddings=False)
    r = ck.report
    assert r.total_in_index == 738
    assert r.text == 426
    assert r.visual_skipped == 297
    assert r.mtp_skipped == 15
    assert r.text + r.visual_skipped + r.mtp_skipped == r.total_in_index
    assert not any("visual" in k or k.startswith("mtp") for k in ck.tensors)


@needs_ckpt
def test_a_log_transform_is_keyed_on_name_not_dtype_or_value():
    """Both published detection heuristics are unsound on this checkpoint.

    `A_log` ships as F32 (so "fp32 means already transformed" is wrong) AND
    entirely negative (so ARCHITECTURE.md:409's "any element >= 0 => raw" is
    also wrong — it would skip the exp and leave the decay ~40x too fast).
    """
    from safetensors import safe_open

    from braid.model.loader import load_checkpoint

    weight_map = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]
    key = "model.language_model.layers.0.linear_attn.A_log"
    with safe_open(MODEL_DIR / weight_map[key], framework="pt") as f:
        raw = f.get_tensor(key)
    assert raw.dtype == torch.float32, "the dtype heuristic's premise"
    assert (raw < 0).all(), (
        "raw A_log now has a non-negative entry, so the value heuristic would "
        "happen to work here and this test no longer pins why it is unsound"
    )

    ck = load_checkpoint(MODEL_DIR, device="cpu", layers=(0,), include_embeddings=False)
    a = ck["layers.0.linear_attn.A"]
    assert ck.report.a_log_transformed == 1
    assert (a < 0).all(), "A must be strictly negative or the state grows"
    torch.testing.assert_close(a, -torch.exp(raw.float()))
    # The untransformed tensor is ~40x larger in magnitude: silent over-decay,
    # not the divergence-to-NaN the docs describe as the tell.
    assert (raw.abs().mean() / a.abs().mean()).item() > 20


@needs_ckpt
def test_norm_offset_is_folded_in_fp32_on_plain_norms_only():
    from safetensors import safe_open

    from braid.model.loader import load_checkpoint

    weight_map = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]

    def raw_of(key):
        with safe_open(MODEL_DIR / weight_map[key], framework="pt") as f:
            return f.get_tensor(key)

    ck = load_checkpoint(MODEL_DIR, device="cpu", layers=(0, 3), include_embeddings=False)

    plain = raw_of("model.language_model.layers.3.input_layernorm.weight")
    got = ck["layers.3.input_layernorm"]
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, 1.0 + plain.float(), rtol=0, atol=0)
    assert not torch.allclose(got, (1.0 + plain.float()).bfloat16().float()), (
        "an fp32 and a bf16 fold are indistinguishable on these values, so this "
        "test would pass either way and pins nothing"
    )

    gated = raw_of("model.language_model.layers.0.linear_attn.norm.weight")
    torch.testing.assert_close(ck["layers.0.linear_attn.norm"], gated.float(), rtol=0, atol=0)


@needs_ckpt
def test_conv1d_is_squeezed_and_shapes_validate():
    from braid.model.loader import load_checkpoint

    ck = load_checkpoint(MODEL_DIR, device="cpu", layers=(0,), include_embeddings=False)
    g = ck.config.gdn
    assert tuple(ck["layers.0.linear_attn.conv1d"].shape) == (g.conv_channels, g.conv_kernel)
    assert g.conv_channels == 8192 and g.inner_size == 4096


@needs_ckpt
@pytest.mark.slow
def test_full_load_ties_lm_head():
    """The whole text tower, once. ~8.8 GB — CPU, so it does not fight the GPU tests."""
    from braid.model.loader import load_checkpoint

    ck = load_checkpoint(MODEL_DIR, device="cpu")
    assert ck.report.tied_lm_head
    assert ck["lm_head"].data_ptr() == ck["embed_tokens"].data_ptr()
    assert ck.report.layers_loaded == tuple(range(32))
    # 426 text tensors, minus 24 A_log names that become A (same count), plus lm_head.
    assert ck.report.loaded == 427
    assert ck.report.a_log_transformed == 24
