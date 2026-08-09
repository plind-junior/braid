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

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from braid.model.config import ModelConfig
from braid.model.norm import rms_norm
from braid.model.quant import FP8Weight, linear, maybe_quantize, project

if TYPE_CHECKING:
    from braid.model.cache import KVCache


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


def grouped_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scale: float,
    groups: int,
) -> torch.Tensor:
    """T=1 GQA attention that never materialises the replicated K and V.

    `F.scaled_dot_product_attention(..., enable_gqa=True)` is correct and, as
    braid calls it, expensive in a specific way: it falls to the **math**
    backend, which `repeat_interleave`s K and V up to the full head count and
    then scales the *transposed key*, in fp32.

    **The reason is the explicit `attn_mask`, not `head_dim`.** An earlier
    revision of this comment blamed the head width, and that is measurably
    wrong (`scripts/sdpa_backend_diag.py`): flash accepts `head_dim = 256` on
    this box for every shape braid issues, decode included, and the default
    dispatcher picks `flash_fwd_splitkv_kernel` for T=1. Only `mem_efficient`
    and `cudnn` decline the head width. What flash will not take is an
    arbitrary additive mask — and `decode_step` must pass one, because `kv_len`
    is pinned to `max_len` while rows have different live lengths.

    So flash-decoding is reachable, and reaching it means giving attention
    per-row KV lengths instead of a mask — varlen or a paged kernel, which is
    Phase 3 item 3's block manager. Until then this is the fast correct path.
    Profiled at B=16, `kv_len=512`, per step across the 8 attention layers:

        1.70 ms   aten::copy_  [16, 4, 4, 512, 256]   the 4x replication of K,V
        1.38 ms   aten::mul    [16, 16, 256, 512]     scaling the expanded key
        1.30 ms   aten::bmm                           the arithmetic itself

    So two thirds of decode attention was spent making copies to feed a matmul
    that did not need them. Grouping the *query* instead is free: head `h`
    attends kv head `h // groups`, which is exactly what a `[B, KVH, G, D]`
    reshape of a `[B, H, 1, D]` query already says, and it leaves K and V read
    once at their stored width.

    The scale moves onto `q` rather than `k`. That is not a numerics
    concession here: `head_dim ** -0.5` is `2**-4`, exactly representable, so
    the multiply is lossless — and `q` is 4 KiB against the key's 64 MiB.
    """
    B, H, T, D = q.shape
    if T != 1:
        raise ValueError(f"grouped_decode_attention is the T=1 path, got T={T}")
    KVH = k.shape[1]

    qg = q.reshape(B, KVH, groups, D) * scale          # [B, KVH, G, D]
    scores = qg @ k.transpose(-1, -2)                  # [B, KVH, G, kv_len]
    if attn_mask is not None:
        # `[B, 1, 1, kv_len]` broadcasts over both kv-head and group.
        scores = scores + attn_mask
    # Softmax in the activation dtype, matching the math backend: PyTorch
    # accumulates in fp32 internally and returns bf16 either way, so this is
    # the same arithmetic, not a cheaper one.
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).reshape(B, H, T, D)


class Attention:
    """One `full_attention` layer's self-attention. Weights are `[out, in]`."""

    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor],
                 quant: bool = False):
        self.cfg = cfg
        self.q_proj = maybe_quantize(weights["self_attn.q_proj"], quant)
        self.k_proj = maybe_quantize(weights["self_attn.k_proj"], quant)
        self.v_proj = maybe_quantize(weights["self_attn.v_proj"], quant)
        self.o_proj = maybe_quantize(weights["self_attn.o_proj"], quant)
        # The norms stay in their stored dtype. They are per-head vectors, they
        # are not a GEMM, and this checkpoint deliberately keeps some of them in
        # fp32.
        self.q_norm = weights["self_attn.q_norm"]
        self.k_norm = weights["self_attn.k_norm"]

    @property
    def quantized(self) -> bool:
        return isinstance(self.q_proj, FP8Weight)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        slots: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        kv_len: int | None = None,
        seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = x.shape
        H, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

        # q/k/v all read the same x, so the activation is quantized once for the
        # three of them when this layer is quantized (see `project`).
        qg, kp, vp = project(x, self.q_proj, self.k_proj, self.v_proj)

        # Per-head [q | gate]: view as [B, T, H, 2D] and split the LAST axis.
        # Chunking the flat 8192 instead pairs head h with the gate of head h/2.
        q, gate = qg.view(B, T, H, 2 * D).chunk(2, dim=-1)
        gate = gate.reshape(B, T, H * D)

        q = rms_norm(q, self.q_norm, cfg.rms_norm_eps).transpose(1, 2)
        k = rms_norm(kp.view(B, T, KVH, D),
                     self.k_norm, cfg.rms_norm_eps).transpose(1, 2)
        v = vp.view(B, T, KVH, D).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            if slots is None or positions is None or kv_len is None:
                raise ValueError("a pooled KV cache needs slots, positions and kv_len")
            # SDPA's `is_causal` aligns its mask TOP-LEFT, so it is only correct
            # when q_len == kv_len. A T>1 forward onto a non-empty cache needs an
            # explicit mask; refuse rather than mask wrongly.
            if T > 1 and kv_len != T and attn_mask is None:
                raise NotImplementedError(
                    f"chunked prefill (T={T} onto {kv_len - T} cached tokens) needs an "
                    "explicit attn_mask; is_causal would align the mask top-left. "
                    "Phase 3 item 3."
                )
            cache.write(k, v, slots, positions, seq_lens)
            k, v = cache.read(slots, kv_len)

        if T == 1 and cfg.num_key_value_groups > 1:
            o = grouped_decode_attention(q, k, v, attn_mask, cfg.attention_scaling,
                                         cfg.num_key_value_groups)
        else:
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
        return linear(o, self.o_proj)

    __call__ = forward
