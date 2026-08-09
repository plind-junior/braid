"""SwiGLU MLP. Dense on this checkpoint — `mlp_only_layers` is empty and there
is no expert routing anywhere in the 4B; MoE arrives with the 35B in Phase 5+.

This is where FP8 lands first, and the reason is bytes: `gate_proj` and
`up_proj` are 64 of the 265 GEMM launches in a decode step and `down_proj`
another 32, together **4.53 GB of the model's 8.41 GB of weights** — 54%, in
three homogeneous matrices with no fused row groups. Everything else is either
smaller, heterogeneous, or (for `lm_head`) directly on the greedy token.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.model.config import ModelConfig
from braid.model.quant import FP8Weight, fp8_matmul, linear, maybe_quantize, quantize_act


class MLP:
    """One layer's feed-forward. Weights are `[out, in]`."""

    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor],
                 quant: bool = False):
        if cfg.hidden_act != "silu":
            raise NotImplementedError(f"hidden_act={cfg.hidden_act!r}")
        self.gate_proj = maybe_quantize(weights["mlp.gate_proj"], quant)
        self.up_proj = maybe_quantize(weights["mlp.up_proj"], quant)
        self.down_proj = maybe_quantize(weights["mlp.down_proj"], quant)

    @property
    def quantized(self) -> bool:
        return isinstance(self.gate_proj, FP8Weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            return F.linear(
                F.silu(F.linear(x, self.gate_proj)) * F.linear(x, self.up_proj),
                self.down_proj,
            )

        # One activation quantization for both of gate and up. Measured at ~16 us
        # a call against a ~16 us saving on the GEMM itself, so paying it twice
        # here would hand the entire win back.
        *lead, K = x.shape
        xq, sx = quantize_act(x.reshape(-1, K))
        gate = fp8_matmul(xq, sx, self.gate_proj, out_dtype=x.dtype)
        up = fp8_matmul(xq, sx, self.up_proj, out_dtype=x.dtype)
        h = F.silu(gate) * up
        return linear(h, self.down_proj).reshape(*lead, self.down_proj.shape[0])

    __call__ = forward
