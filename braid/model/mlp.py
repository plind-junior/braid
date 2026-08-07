"""SwiGLU MLP. Dense on this checkpoint — `mlp_only_layers` is empty and there
is no expert routing anywhere in the 4B; MoE arrives with the 35B in Phase 5+.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.model.config import ModelConfig


class MLP:
    """One layer's feed-forward. Weights are `[out, in]`."""

    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor]):
        if cfg.hidden_act != "silu":
            raise NotImplementedError(f"hidden_act={cfg.hidden_act!r}")
        self.gate_proj = weights["mlp.gate_proj"]
        self.up_proj = weights["mlp.up_proj"]
        self.down_proj = weights["mlp.down_proj"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            F.silu(F.linear(x, self.gate_proj)) * F.linear(x, self.up_proj),
            self.down_proj,
        )

    __call__ = forward
