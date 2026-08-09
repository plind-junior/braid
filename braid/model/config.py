"""Parse the full HF `config.json` for a Qwen3.5 hybrid.

`braid/config.py` holds `GDNConfig`, the shape parameters the scan kernel needs.
This module is the whole model: layer schedule, attention shapes, MLP width,
rope, vocab. It builds a `GDNConfig` for the recurrent half rather than
duplicating it.

Three things about this checkpoint that a generic parser gets wrong:

  1. **The text config is nested.** `Qwen3_5ForConditionalGeneration` is a
     vision-language wrapper; every field below lives under `text_config`, and
     the top level carries a `vision_config` for a tower braid does not run.
     Reading `hidden_size` off the top level yields the *visual* 1024.

  2. **`head_dim` is explicit and matches nothing.** 16 heads x 256 = 4096,
     against `hidden_size` 2560. The near-universal `hidden_size //
     num_attention_heads` fallback gives 160 here. On *this* checkpoint that is
     caught loudly — `q_proj` is `[8192, 2560]` and 8192 % (2 x 160) != 0, so
     the first reshape raises (verified in `scripts/parity_report.py`). But that
     is a divisibility accident, not a safety property: a sibling checkpoint
     whose widths divided would reshape cleanly into fluent garbage. `from_dict`
     refuses to guess rather than relying on the accident.

  3. **`num_key_value_heads` is 4, not 2.** The 2 belongs to the 35B, whose
     shapes are in docs/THESIS.md §3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from braid.config import GDNConfig

LINEAR = "linear_attention"
FULL = "full_attention"


@dataclass(frozen=True)
class ModelConfig:
    """Everything from `text_config`, named as HF names it."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attn_output_gate: bool
    attention_bias: bool

    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool
    hidden_act: str
    dtype: str

    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, ...]
    mrope_interleaved: bool

    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_value_head_dim: int
    linear_key_head_dim: int
    linear_conv_kernel_dim: int

    mlp_only_layers: tuple[int, ...]
    mamba_ssm_dtype: str
    mtp_num_hidden_layers: int

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers is {self.num_hidden_layers}"
            )
        unknown = set(self.layer_types) - {LINEAR, FULL}
        if unknown:
            raise ValueError(f"unknown layer_types: {sorted(unknown)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.mlp_only_layers:
            raise NotImplementedError(
                f"mlp_only_layers={self.mlp_only_layers}; this checkpoint is expected to be "
                "dense (MoE is Phase 5+)"
            )
        # `mrope_section` indexes the half-dim frequency vector, so it must sum
        # to rotary_dim // 2 or the interleave silently walks off the end.
        if sum(self.mrope_section) * 2 != self.rotary_dim:
            raise ValueError(
                f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, "
                f"expected {self.rotary_dim // 2} (= rotary_dim/2)"
            )

    # --- derived attention shapes -------------------------------------------

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def q_proj_out(self) -> int:
        """`attn_output_gate` doubles q_proj: [q | gate] interleaved PER HEAD.

        HF views the projection as `[..., n_heads, 2*head_dim]` and chunks the
        last axis, so the layout is `q_h0 | gate_h0 | q_h1 | gate_h1 | ...`,
        NOT `all_q | all_gate`. Splitting the flat 8192 in half instead gives
        head h the gate of head h/2 — plausible output, wrong model.
        """
        return self.q_dim * (2 if self.attn_output_gate else 1)

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def rotary_dim(self) -> int:
        """64 here: `partial_rotary_factor` 0.25 of a 256-wide head.

        The top 192 dims of q and k are passed through unrotated.
        """
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def attention_scaling(self) -> float:
        return self.head_dim ** -0.5

    # --- layer schedule ------------------------------------------------------

    @property
    def attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == FULL)

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == LINEAR)

    def is_gdn(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == LINEAR

    @property
    def gdn(self) -> GDNConfig:
        """The scan kernel's view of this checkpoint.

        Note the HF name -> GDNConfig name inversion: `linear_num_KEY_heads` is
        `n_groups` and `linear_num_VALUE_heads` is `n_heads`.
        """
        return GDNConfig(
            n_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            state_size=self.linear_key_head_dim,
            n_groups=self.linear_num_key_heads,
            conv_kernel=self.linear_conv_kernel_dim,
            n_gdn_layers=len(self.gdn_layers),
        )

    # --- construction --------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        text = raw.get("text_config", raw)
        rope = text.get("rope_parameters", {})

        if "head_dim" not in text:
            raise ValueError(
                "config.json has no explicit head_dim. Refusing to fall back to "
                "hidden_size // num_attention_heads: on this family that fallback is "
                "wrong by a factor of 1.6 and produces a valid reshape, not an error."
            )
        rope_type = rope.get("rope_type", "default")
        if rope_type != "default":
            raise NotImplementedError(f"rope_type={rope_type!r}; only 'default' is implemented")

        return cls(
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            layer_types=tuple(text["layer_types"]),
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            attn_output_gate=bool(text.get("attn_output_gate", False)),
            attention_bias=bool(text.get("attention_bias", False)),
            rms_norm_eps=text["rms_norm_eps"],
            vocab_size=text["vocab_size"],
            # The tie flag is duplicated at both levels; the text tower's own
            # value is the one that governs its lm_head.
            tie_word_embeddings=bool(text.get("tie_word_embeddings",
                                              raw.get("tie_word_embeddings", False))),
            hidden_act=text.get("hidden_act", "silu"),
            dtype=text.get("dtype", text.get("torch_dtype", "bfloat16")),
            rope_theta=float(rope.get("rope_theta", 10000.0)),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
            mrope_section=tuple(rope.get("mrope_section", ())),
            mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
            mlp_only_layers=tuple(text.get("mlp_only_layers", ())),
            mamba_ssm_dtype=text.get("mamba_ssm_dtype", "float32"),
            mtp_num_hidden_layers=int(text.get("mtp_num_hidden_layers", 0)),
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> ModelConfig:
        p = Path(path)
        if p.is_dir():
            p = p / "config.json"
        with open(p) as f:
            return cls.from_dict(json.load(f))
