"""Config parsing and checkpoint loading — the traps that are silent, not loud.

Two of these need no GPU and no checkpoint; they run anywhere.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from braid.config import GDNConfig
from braid.model.config import ModelConfig
from braid.model.loader import LM_HEAD_PREFIX, MTP_PREFIX, VISUAL_PREFIX, braid_name

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


def test_the_untied_lm_head_is_found_outside_the_text_tower():
    """`lm_head.weight` sits at the TOP level, not under `model.language_model.`.

    Qwen3.5-4B is tied and ships no such tensor, so this path never ran until
    the 9B. It matters that the loader *recognises* the prefix rather than
    merely tolerating it: an unrecognised key raises, and a key silently
    dropped instead would give an untied model a random LM head — fluent
    garbage, the failure mode this codebase keeps having to catch.

    Both the tied and untied paths land in the same flat `lm_head` slot, so
    nothing downstream branches on `tie_word_embeddings`.
    """
    assert braid_name(LM_HEAD_PREFIX + "weight") == "lm_head"
    assert braid_name("lm_head.weight") == "lm_head"
    # Still not a member of the text tower's namespace, and must not collide
    # with a same-named tensor inside it.
    assert braid_name("model.language_model.lm_head.weight") == "lm_head"


def test_untied_config_parses_and_reports_untied():
    """The 9B/27B shape: `tie_word_embeddings` false at the text level."""
    raw = copy.deepcopy(MINIMAL)
    raw["text_config"]["tie_word_embeddings"] = False
    raw["text_config"]["hidden_size"] = 4096
    raw["text_config"]["intermediate_size"] = 12288
    cfg = ModelConfig.from_dict(raw)
    assert cfg.tie_word_embeddings is False
    assert cfg.hidden_size == 4096
    # The GDN half is untouched by any of that -- the whole reason the 9B is a
    # cheap retarget. `n_gdn_layers` is normalised away because MINIMAL is a
    # 4-layer stub; the real 32-layer equality is asserted against the
    # checkpoint on disk in test_checkpoint_config_matches_the_pinned_gdn_shapes.
    assert (replace(cfg.gdn, n_gdn_layers=24)
            == GDNConfig.qwen35_4b() == GDNConfig.qwen35_9b())


# --- against the real checkpoint ------------------------------------------------

@needs_ckpt
def test_checkpoint_config_matches_the_pinned_gdn_shapes():
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    # 4B and 9B are GDN-identical, so one assertion covers both targets; if it
    # ever fails, `braid/config.py` has drifted from the checkpoint on disk.
    assert cfg.gdn == GDNConfig.qwen35_4b() == GDNConfig.qwen35_9b(), (
        f"braid/config.py's pinned shapes drifted from the checkpoint: "
        f"{cfg.gdn} vs {GDNConfig.qwen35_4b()}"
    )
    assert cfg.num_hidden_layers == 32
    assert len(cfg.gdn_layers) == 24 and len(cfg.attention_layers) == 8
    assert cfg.num_key_value_heads == 4, "the 2 in THESIS.md is the 35B, not this one"


@needs_ckpt
def test_loader_excludes_the_visual_tower_and_mtp_head():
    """The comparison is void if braid loads tensors the GGUF does not contain."""
    from braid.model.loader import load_checkpoint

    ck = load_checkpoint(MODEL_DIR, device="cpu", layers=(0, 3), include_embeddings=False)
    r = ck.report

    # Asserted as an ACCOUNTING IDENTITY, not as per-checkpoint constants. The
    # counts differ across the family -- Qwen3.5-4B is 738 = 426 text + 297
    # visual + 15 MTP, Qwen3.5-9B is 775 = 427 + 333 + 15 -- and pinning them
    # makes the suite reject the next target for no reason. What must hold on
    # every member is that nothing is unaccounted for.
    assert r.text + r.visual_skipped + r.mtp_skipped == r.total_in_index
    assert r.unrecognised == (), f"unrecognised prefixes: {r.unrecognised}"
    # Non-vacuous: this checkpoint really does carry both towers, so the filter
    # is doing work rather than passing a file that never had them.
    assert r.visual_skipped > 0 and r.mtp_skipped > 0
    assert not any("visual" in k or k.startswith("mtp") for k in ck.tensors)


@needs_ckpt
def test_a_log_transform_is_keyed_on_name_not_dtype_or_value():
    """Both published detection heuristics are unsound on this checkpoint.

    `A_log` ships as F32 (so "fp32 means already transformed" is wrong) AND
    entirely negative (so the original spec's "any element >= 0 => raw" is
    also wrong — it would skip the exp and leave the decay ~40x too fast).
    See ARCHITECTURE.md §6.
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
def test_full_load_resolves_the_lm_head_either_way():
    """The whole text tower, once. CPU, so it does not fight the GPU tests.

    Both resolutions land in the same flat `lm_head` slot and this asserts
    whichever one the checkpoint calls for: Qwen3.5-4B is tied and has no such
    tensor, so it is aliased onto `embed_tokens`; Qwen3.5-9B ships a real one at
    the top level of the index. An untied checkpoint that silently aliased would
    generate fluently from the wrong head, which is why the alias is asserted by
    pointer identity rather than by shape.
    """
    from braid.model.loader import load_checkpoint

    ck = load_checkpoint(MODEL_DIR, device="cpu")
    cfg, r = ck.config, ck.report
    tied = cfg.tie_word_embeddings

    assert r.tied_lm_head is tied
    assert "lm_head" in ck.tensors
    same = ck["lm_head"].data_ptr() == ck["embed_tokens"].data_ptr()
    assert same is tied, (
        "tied checkpoint must alias embed_tokens; untied must not"
        if tied else "untied checkpoint aliased its lm_head onto embed_tokens")
    assert tuple(ck["lm_head"].shape) == (cfg.vocab_size, cfg.hidden_size)

    assert r.layers_loaded == tuple(range(cfg.num_hidden_layers))
    assert r.a_log_transformed == len(cfg.gdn_layers)
    # A_log -> A is a rename, so it does not change the count; the only tensor
    # that appears beyond the filtered set is a synthesised tied head.
    assert r.loaded == r.text + (1 if tied else 0)
