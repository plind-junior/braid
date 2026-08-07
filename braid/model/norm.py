"""RMSNorm, in the two forms this model uses.

Both compute in fp32 and cast on the way out, matching HF. The difference is
where the unit offset lives, and mixing them up is worth ~2x perplexity:

  `Qwen3_5RMSNorm`      stores gamma as a DELTA -> effective gamma = 1 + W
  `Qwen3_5RMSNormGated` stores gamma DIRECTLY   -> effective gamma = W

braid folds the `1 + W` at load time (see `loader.py`), so both forms reduce to
one multiply here and the caller cannot apply the offset twice.

The fold is done in fp32. Measured (`scripts/parity_report.py`), against HF:

    fp32 fold          rel L2 0.0        cosine 1.0
    bf16 fold          rel L2 1.86e-3    cosine 0.999998
    no offset at all   rel L2 8.04e-1    cosine 0.976

So the bf16 fold would still *clear* the 5e-3 / 0.99999 gate — it is not the
difference between pass and fail. It is chosen because it costs nothing (these
are 2560 floats per layer) and because it buys back three orders of magnitude of
headroom that the 32-layer forward and the perplexity gate will spend. Dropping
the offset entirely is the failure the gate is actually for; that is the
13.65 -> 6.82 perplexity bug.
"""
from __future__ import annotations

import torch


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """gamma is the EFFECTIVE gamma — already `1 + W` where that applies."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * gamma).type_as(x)


def rms_norm_gated(
    x: torch.Tensor, gate: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    """Normalise FIRST, then gate — the opposite order from the Mamba2 path."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * gamma * torch.nn.functional.silu(gate.float())).type_as(x)
