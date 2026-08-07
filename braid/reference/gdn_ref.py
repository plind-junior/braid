"""fp32 reference for the Gated DeltaNet single-token decode step.

Operator order is deliberate and must not be "simplified":

    kv[d]    = sum_s S[s,d] * k_hat[s]          # reduction on the UNDECAYED state
    delta[d] = (v[d] - alpha * kv[d]) * beta
    S'[s,d]  = alpha * S[s,d] + k_hat[s] * delta[d]
    y[d]     = (sum_s S'[s,d] * q_hat[s]) * head_dim ** -0.5

alpha is applied AFTER the reduction over s. Decaying the state first and then
reducing is algebraically identical but not bit-identical in fp32.

In the reference engine the state update and the y accumulation share ONE loop
(gdn.cu:148-174); splitting them into "update, then read" changes the fp32
result. The vectorized form below cannot express that fusion, which is exactly
why the CUDA kernel is checked against it at rtol=2e-5 rather than bitwise.
"""
from __future__ import annotations

import torch

from braid.config import GDNConfig

# ADDITIVE epsilon on the sum of squares, matching HF exactly:
#   transformers.models.qwen3_5.modeling_qwen3_5.l2norm
#     inv_norm = torch.rsqrt((x * x).sum(dim) + eps)   # eps = 1e-6
#
# NOT the reference engine's clamped form `rsqrt(fmax(sum_sq, 1e-12))`. The two
# agree to ~4e-9 for a healthy head (sum_sq ~ 128) and diverge by 10x for a
# near-zero head (sum_sq ~ 1e-8: HF 995 vs 10000). Their comment argues the
# additive form over-clamps near-zero heads -- but HF is the implementation this
# checkpoint was trained with, so HF is ground truth and theirs is the deviation.
_L2_EPS = 1e-6


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """HF-exact L2 norm: rsqrt(sum(x^2) + 1e-6)."""
    inv = torch.rsqrt((x * x).sum(-1, keepdim=True) + _L2_EPS)
    return x * inv


def gdn_decode_naive(
    state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor, cfg: GDNConfig,
) -> torch.Tensor:
    """Explicit-loop oracle. Slow, obviously correct."""
    B = state.shape[0]
    hpg = cfg.heads_per_group
    scale = cfg.head_dim ** -0.5
    qh, kh = _l2norm(q), _l2norm(k)
    y = torch.zeros(B, cfg.n_heads, cfg.head_dim, dtype=torch.float32)

    for b in range(B):
        for h in range(cfg.n_heads):
            g = h // hpg
            a, bt = alpha[b, h].item(), beta[b, h].item()
            for d in range(cfg.head_dim):
                kv = sum(state[b, h, s, d].item() * kh[b, g, s].item() for s in range(cfg.state_size))
                delta = (v[b, h, d].item() - a * kv) * bt
                acc = 0.0
                for s in range(cfg.state_size):
                    new = a * state[b, h, s, d].item() + kh[b, g, s].item() * delta
                    state[b, h, s, d] = new
                    acc += new * qh[b, g, s].item()
                y[b, h, d] = acc * scale
    return y


def gdn_decode_vectorized(
    state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor, cfg: GDNConfig,
) -> torch.Tensor:
    """Batched tensor form of `gdn_decode_naive`. Same contract."""
    hpg = cfg.heads_per_group
    scale = cfg.head_dim ** -0.5

    # [B, n_groups, SS] -> [B, n_heads, SS] by repeating each group hpg times.
    # This is the GROUPED (HF SafeTensors) layout, g = h // hpg. The tiled
    # (GGUF) layout g = h % n_groups is a different permutation of the same
    # index range and produces plausible garbage rather than a crash.
    qh = _l2norm(q).repeat_interleave(hpg, dim=1)
    kh = _l2norm(k).repeat_interleave(hpg, dim=1)

    a = alpha[..., None]                                   # [B, H, 1]
    bt = beta[..., None]                                   # [B, H, 1]

    kv = torch.einsum("bhsd,bhs->bhd", state, kh)          # [B, H, HD]
    delta = (v - a * kv) * bt                              # [B, H, HD]
    state.mul_(a[..., None]).add_(kh[..., None] * delta[:, :, None, :])
    y = torch.einsum("bhsd,bhs->bhd", state, qh) * scale
    return y


def gdn_gates(
    alpha_raw: torch.Tensor, beta_raw: torch.Tensor,
    A: torch.Tensor, dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive (g, beta) from the raw projections, exactly as the reference
    engine does.

    gdn.cu:89-96:
        dt = alpha_raw + dt_bias
        dt = (dt > 20) ? dt : logf(1 + expf(dt))     <- logf(1+expf(.)), NOT log1pf
        g  = expf(max(A * dt, -20))                  <- A is ALREADY -exp(A_log_HF)
        beta = sigmoid(clamp(beta_raw, -20, 20))

    Both clamps matter. `A` must already carry the negative sign: the reference
    engine applies `A = -exp(A_log_HF)` on the host at load, deciding from
    VALUES (any element >= 0 implies raw HF) rather than dtype. Getting the sign
    wrong makes the state grow instead of decay — they logged per-token absmax
    0.04, 0.06, 0.40, 2.51, 110, 31680, inf, then NaN, then one token forever
    (#1282).
    """
    dt = alpha_raw.float() + dt_bias
    dt = torch.where(dt > 20.0, dt, torch.log1p(torch.exp(dt)))
    g = torch.exp(torch.clamp(A * dt, min=-20.0))
    beta = torch.sigmoid(torch.clamp(beta_raw.float(), -20.0, 20.0))
    return g, beta
