"""The Gated DeltaNet layer — projections, causal conv, gates, scan, gated norm.

The recurrence itself is `braid.reference.gdn_ref.gdn_decode_vectorized`, the
same fp32 oracle the CUDA decode kernel is checked against. **Prefill runs that
one-token step in a loop.** That is slow and deliberate: it makes prefill and
decode the same arithmetic by construction rather than by test, so the classic
"generation drifts after the first token" bug cannot exist here. Phase 5's
ragged chunkwise scan replaces the loop; this phase makes no speed claim.

Gate arithmetic follows **HF, not the reference engine**:

    g    = A * softplus(a_raw + dt_bias)        with A = -exp(A_log), from the loader
    beta = sigmoid(b_raw)

The reference engine additionally clamps `A*dt` at -20 and `b_raw` at +-20
(`gdn.cu:89-96`, mirrored in `gdn_ref.gdn_gates`). Both clamps are no-ops on
real activations — `sigmoid(20)` is 1 to within 2e-9 — and HF is the
implementation the checkpoint was trained with, so parity beats the deviation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.model.cache import RecurrentCache
from braid.model.config import ModelConfig
from braid.model.norm import rms_norm_gated
from braid.reference.gdn_ref import _l2norm, gdn_decode_vectorized


class GatedDeltaNet:
    """One `linear_attention` layer. Weights are `[out, in]`."""

    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.g = cfg.gdn
        w = weights
        self.in_proj_qkv = w["linear_attn.in_proj_qkv"]
        self.in_proj_z = w["linear_attn.in_proj_z"]
        self.in_proj_a = w["linear_attn.in_proj_a"]
        self.in_proj_b = w["linear_attn.in_proj_b"]
        self.out_proj = w["linear_attn.out_proj"]
        self.conv1d = w["linear_attn.conv1d"]          # [C, K], fp/bf16
        self.conv_bias = w.get("linear_attn.conv1d_bias")
        self.A = w["linear_attn.A"]                    # fp32, already -exp(A_log)
        self.dt_bias = w["linear_attn.dt_bias"]        # fp32
        self.norm = w["linear_attn.norm"]              # fp32, gated -> NO 1+W

    # --- convolution ---------------------------------------------------------

    def _conv(self, x: torch.Tensor, cache: RecurrentCache | None) -> torch.Tensor:
        """Depthwise causal conv + SiLU. `x` is `[B, C, T]`.

        SiLU is applied to **all** of Q, K and V, not just V.
        """
        B, C, T = x.shape
        K = self.g.conv_kernel
        w = self.conv1d.unsqueeze(1)  # [C, 1, K]

        if cache is not None and T == 1:
            # Decode: splice the window, then one dot per channel.
            joined = torch.cat([cache.conv, x], dim=-1)
            cache.conv.copy_(joined[:, :, -K:])
            out = F.conv1d(joined, w, self.conv_bias, padding=0, groups=C)
            return F.silu(out[:, :, -1:])

        if cache is not None:
            # The cached window is the last K PRE-conv inputs. For T >= K this
            # pad is negative, i.e. a left truncation — HF's own trick.
            cache.conv.copy_(F.pad(x, (K - T, 0)) if T < K else x[:, :, -K:])
        out = F.conv1d(x, w, self.conv_bias, padding=K - 1, groups=C)
        return F.silu(out[:, :, :T])

    # --- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor, cache: RecurrentCache | None = None) -> torch.Tensor:
        cfg, g = self.cfg, self.g
        B, T, _ = x.shape
        dtype = x.dtype

        qkv = F.linear(x, self.in_proj_qkv).transpose(1, 2)   # [B, C, T]
        qkv = self._conv(qkv, cache).transpose(1, 2)          # [B, T, C]

        # [Q | K | V], Q FIRST. Settled empirically by tests/test_hf_parity.py,
        # not read off the reference engine — the two readings of it disagree and
        # the wrong order is fluent, not fatal.
        key_dim = g.n_groups * g.state_size
        q, k, v = torch.split(qkv, [key_dim, key_dim, g.inner_size], dim=-1)
        q = q.reshape(B, T, g.n_groups, g.state_size)
        k = k.reshape(B, T, g.n_groups, g.state_size)
        v = v.reshape(B, T, g.n_heads, g.head_dim)

        a_raw = F.linear(x, self.in_proj_a)
        b_raw = F.linear(x, self.in_proj_b)

        # beta's sigmoid is taken in the ACTIVATION dtype and only then widened;
        # `g` is computed entirely in fp32. That asymmetry is HF's (`beta =
        # b.sigmoid()` vs `g = -A_log.float().exp() * softplus(a.float() + ...)`)
        # and it is not cosmetic: taking beta's sigmoid in fp32 instead moves the
        # whole layer output by 4.8e-3 relative — one bf16 epsilon, flat across T
        # and flat across tokens, measured in scripts/gdn_layer_diag.py. beta is
        # the delta-rule step size, so its rounding lands straight in the output.
        beta = torch.sigmoid(b_raw).float()                                  # [B,T,H]
        alpha = torch.exp(self.A * F.softplus(a_raw.float() + self.dt_bias))  # [B,T,H]

        # l2norm in the ACTIVATION dtype, then cast — HF's order. Normalising
        # after the cast changes the last bits and compounds over 24 layers.
        qn = _l2norm(q).float()
        kn = _l2norm(k).float()
        vf = v.float()

        state = (cache.state if cache is not None else
                 torch.zeros(B, g.n_heads, g.state_size, g.head_dim,
                             device=x.device, dtype=torch.float32))

        y = torch.empty(B, T, g.n_heads, g.head_dim, device=x.device, dtype=torch.float32)
        for t in range(T):
            y[:, t] = gdn_decode_vectorized(
                state=state, q=qn[:, t], k=kn[:, t], v=vf[:, t],
                alpha=alpha[:, t], beta=beta[:, t], cfg=g, normalize=False,
            )

        # HF casts the scan output back to the activation dtype BEFORE gating.
        core = y.to(dtype).reshape(-1, g.head_dim)
        z = F.linear(x, self.in_proj_z).reshape(-1, g.head_dim)
        core = rms_norm_gated(core, z, self.norm, cfg.rms_norm_eps)
        return F.linear(core.reshape(B, T, g.inner_size), self.out_proj)

    __call__ = forward
