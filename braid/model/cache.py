"""Per-sequence state for a B=1 eager forward: KV for the 8 attention layers,
conv + recurrent state for the 24 GDN layers.

This is deliberately the *simple* version — one contiguous KV buffer per layer,
preallocated to `max_len`. Paged KV, the recurrent slot pool and the device-side
`slot_idx` indirection are Phase 3; putting them here would mean writing the
batching machinery before there is a correct B=1 forward to check it against.

The one thing carried forward from Phase 3's design is the **shape**: the
recurrent state is `[B, n_heads, state_size, head_dim]` fp32 with `head_dim`
fastest-varying, which is the per-slot slab layout the decode kernel already
indexes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from braid.model.config import ModelConfig


class KVCache:
    """One attention layer's K/V, preallocated and grown by `append`."""

    def __init__(self, cfg: ModelConfig, batch: int, max_len: int,
                 device: str | torch.device, dtype: torch.dtype):
        shape = (batch, cfg.num_key_value_heads, max_len, cfg.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.length = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """k, v are `[B, KVH, T, D]`. Returns the whole prefix including them."""
        t = k.shape[2]
        if self.length + t > self.k.shape[2]:
            raise ValueError(
                f"KV cache overflow: {self.length} + {t} > {self.k.shape[2]}; "
                "raise max_len"
            )
        self.k[:, :, self.length: self.length + t] = k
        self.v[:, :, self.length: self.length + t] = v
        self.length += t
        return self.k[:, :, : self.length], self.v[:, :, : self.length]


class RecurrentCache:
    """One GDN layer's conv window and delta-rule state.

    `conv` holds the last `conv_kernel` **pre-convolution** inputs, matching HF's
    `causal_conv1d_update` convention. Holding post-conv outputs instead would
    decode fluently and wrongly.
    """

    def __init__(self, cfg: ModelConfig, batch: int,
                 device: str | torch.device, dtype: torch.dtype):
        g = cfg.gdn
        self.conv = torch.zeros(batch, g.conv_channels, g.conv_kernel,
                                device=device, dtype=dtype)
        # fp32 regardless of the activation dtype: `mamba_ssm_dtype` is float32,
        # and FP8 E4M3 state is refuted outright (ARCHITECTURE.md §5).
        self.state = torch.zeros(batch, g.n_heads, g.state_size, g.head_dim,
                                 device=device, dtype=torch.float32)


@dataclass
class Cache:
    """Whole-model state. `layers[i]` is a `KVCache` or a `RecurrentCache`."""

    layers: list[KVCache | RecurrentCache]
    seq_len: int = 0

    @classmethod
    def allocate(cls, cfg: ModelConfig, batch: int, max_len: int,
                 device: str | torch.device, dtype: torch.dtype) -> "Cache":
        return cls(layers=[
            RecurrentCache(cfg, batch, device, dtype) if cfg.is_gdn(i)
            else KVCache(cfg, batch, max_len, device, dtype)
            for i in range(cfg.num_hidden_layers)
        ])

    def bytes(self) -> int:
        n = 0
        for lyr in self.layers:
            if isinstance(lyr, KVCache):
                n += lyr.k.numel() * lyr.k.element_size() * 2
            else:
                n += (lyr.conv.numel() * lyr.conv.element_size()
                      + lyr.state.numel() * lyr.state.element_size())
        return n
