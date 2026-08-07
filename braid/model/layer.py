"""One decoder layer: pre-norm, token mixer, residual, pre-norm, MLP, residual.

The mixer is chosen by `layer_types[i]`, not by index arithmetic. The schedule
here happens to be period 4 (`full_attention_interval`), but HF dispatches on
the list and so does braid — a checkpoint that broke the period would otherwise
load cleanly and mix the wrong sublayer into 24 of 32 layers.
"""
from __future__ import annotations

import torch

from braid.model.attention import Attention
from braid.model.cache import KVCache, RecurrentCache
from braid.model.config import ModelConfig
from braid.model.gdn import GatedDeltaNet
from braid.model.mlp import MLP
from braid.model.norm import rms_norm


class DecoderLayer:
    def __init__(self, cfg: ModelConfig, layer_idx: int, weights: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_gdn = cfg.is_gdn(layer_idx)
        self.mixer = GatedDeltaNet(cfg, weights) if self.is_gdn else Attention(cfg, weights)
        self.mlp = MLP(cfg, weights)
        self.input_layernorm = weights["input_layernorm"]
        self.post_attention_layernorm = weights["post_attention_layernorm"]

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | RecurrentCache | None = None,
    ) -> torch.Tensor:
        h = rms_norm(x, self.input_layernorm, self.cfg.rms_norm_eps)
        h = self.mixer(h, cache=cache) if self.is_gdn else self.mixer(h, cos, sin, cache=cache)
        x = x + h

        h = rms_norm(x, self.post_attention_layernorm, self.cfg.rms_norm_eps)
        return x + self.mlp(h)

    __call__ = forward
