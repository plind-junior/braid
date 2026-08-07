"""Gated GQA attention, BF16, plain torch ops.

No custom kernels here — the scan is the IP, this is the conventional half, and
it exists to be *correct* first. FlashInfer / paged KV replace the SDPA call in
Phase 3; the surrounding arithmetic is what this module pins.

Four places this differs from a textbook Llama block:

  * `head_dim = 256` with 16 heads over `hidden_size = 2560`. Every reshape
    below uses the configured `head_dim`; none derives it.
  * `attn_output_gate` — `q_proj` emits `2 * n_heads * head_dim` and the split
    is **per head**, `[q_h | gate_h]` repeating, not `[all_q | all_gate]`.
  * q/k RMSNorm over the head dim, before rope, with the `1 + W` gamma the
    loader already folded.
  * **Partial rope.** Only the first `rotary_dim = 64` of each 256-wide head is
    rotated; the top 192 pass through. Rotating all 256 is not a shape error.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.model.config import ModelConfig
from braid.model.norm import rms_norm


class RotaryEmbedding:
    """cos/sin for partial rope, matching `Qwen3_5TextRotaryEmbedding`.

    That module is MRoPE: it expands `position_ids` to three grids (temporal,
    height, width) and interleaves their frequencies. For **text**, HF expands a
    2-D `position_ids` by broadcasting the same row three times, so all three
    grids are identical and `apply_interleaved_mrope` selects between equal
    values — a no-op. This class computes the text case directly; the
    equivalence is pinned by `test_rope_matches_hf`, not assumed.
    """

    def __init__(self, cfg: ModelConfig, device: str | torch.device, dtype: torch.dtype):
        dim = cfg.rotary_dim
        exponent = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        self.inv_freq = (1.0 / (cfg.rope_theta ** exponent)).to(device)
        self.dtype = dtype

    def __call__(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids `[B, T]` -> cos, sin `[B, T, rotary_dim]`."""
        freqs = position_ids.float()[..., None] * self.inv_freq  # [B, T, rotary_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k are `[B, H, T, head_dim]`; cos/sin `[B, T, rotary_dim]`."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rd = cos.shape[-1]
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    q_out = torch.cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], dim=-1)
    k_out = torch.cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], dim=-1)
    return q_out, k_out


class Attention:
    """One `full_attention` layer's self-attention. Weights are `[out, in]`."""

    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.q_proj = weights["self_attn.q_proj"]
        self.k_proj = weights["self_attn.k_proj"]
        self.v_proj = weights["self_attn.v_proj"]
        self.o_proj = weights["self_attn.o_proj"]
        self.q_norm = weights["self_attn.q_norm"]
        self.k_norm = weights["self_attn.k_norm"]

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = x.shape
        H, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

        # Per-head [q | gate]: view as [B, T, H, 2D] and split the LAST axis.
        # Chunking the flat 8192 instead pairs head h with the gate of head h/2.
        qg = F.linear(x, self.q_proj).view(B, T, H, 2 * D)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, T, H * D)

        q = rms_norm(q, self.q_norm, cfg.rms_norm_eps).transpose(1, 2)
        k = rms_norm(F.linear(x, self.k_proj).view(B, T, KVH, D),
                     self.k_norm, cfg.rms_norm_eps).transpose(1, 2)
        v = F.linear(x, self.v_proj).view(B, T, KVH, D).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        o = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None and T > 1,
            scale=cfg.attention_scaling,
            enable_gqa=cfg.num_key_value_groups > 1,
        )
        o = o.transpose(1, 2).reshape(B, T, H * D)
        # Gate AFTER attention, BEFORE o_proj.
        o = o * torch.sigmoid(gate)
        return F.linear(o, self.o_proj)

    __call__ = forward
