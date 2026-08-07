"""Which bf16 arm is closer to the truth, and which stage owns the gap?

`test_bf16_stays_inside_the_phase_2_parity_gate` compares the two bf16 arms to
each other, which can only say they differ — not which one moved. Everything
here is compared against the same inputs promoted to fp32, so the disagreement
gets an owner instead of a tolerance bump.

Measured first pass: grouped is ~4.1e-3 from fp32, SDPA math ~2.1e-3 — the
grouped form is the one that moved, and `k * scale` round-trips exactly (the
scale is 2**-4), so scale placement is refuted as the cause. That leaves the
matmul shapes: SDPA reduces at M=1 over 48 batches, grouped at M=4 over 12.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.model.attention import grouped_decode_attention

B, H, KVH, D, L = 3, 16, 4, 256, 96
G, SCALE = H // KVH, D ** -0.5


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def sdpa(q, k, v, mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False,
                                          scale=SCALE, enable_gqa=True)


def stages(q, k, v):
    """The grouped form, stage by stage, in whatever dtype it is handed."""
    qg = q.reshape(B, KVH, G, D) * SCALE
    scores = qg @ k.transpose(-1, -2)
    probs = torch.softmax(scores, dim=-1)
    out = (probs @ v).reshape(B, H, 1, D)
    # Reshaped to SDPA's [B, H, 1, *] so the two arms are comparable elementwise.
    return scores.reshape(B, H, 1, L), probs.reshape(B, H, 1, L), out


def expanded(q, k, v):
    """The same arithmetic at SDPA's shapes: M=1, heads expanded."""
    ke, ve = (t.repeat_interleave(G, dim=1) for t in (k, v))
    scores = (q * SCALE) @ ke.transpose(-1, -2)
    probs = torch.softmax(scores, dim=-1)
    return scores, probs, probs @ ve


def upcast_probs(q, k, v):
    """Same scores, but the softmax output stays fp32 into the second matmul.

    If this lands on SDPA's error rather than ours, the math backend's extra
    accuracy is an internal fp32 promotion — which is also part of what makes
    it read V twice and cost 3.1 ms a step.
    """
    qg = q.reshape(B, KVH, G, D) * SCALE
    scores = qg @ k.transpose(-1, -2)
    probs = torch.softmax(scores.float(), dim=-1)
    return (probs @ v.float()).reshape(B, H, 1, D)


def run(label: str) -> None:
    print(f"\n=== {label}")
    for seed in (3, 4):
        g = torch.Generator(device="cuda").manual_seed(seed)
        qb = torch.randn(B, H, 1, D, generator=g, device="cuda").bfloat16()
        kb = torch.randn(B, KVH, L, D, generator=g, device="cuda").bfloat16()
        vb = torch.randn(B, KVH, L, D, generator=g, device="cuda").bfloat16()

        rs, rp, ro = stages(qb.float(), kb.float(), vb.float())     # fp32 truth
        gs, gp, go = stages(qb, kb, vb)
        es, ep, eo = expanded(qb, kb, vb)
        math_be = sdpa(qb, kb, vb).float()

        print(f"  seed {seed}      {'scores':>10}{'probs':>11}{'out':>11}")
        print(f"    grouped M=4 {rel(gs.float(), rs):>10.3e}"
              f"{rel(gp.float(), rp):>11.3e}{rel(go.float(), ro):>11.3e}")
        print(f"    expand  M=1 {rel(es.float(), rs):>10.3e}"
              f"{rel(ep.float(), rp):>11.3e}{rel(eo.float(), ro):>11.3e}")
        print(f"    sdpa  (out only)                            "
              f"{rel(math_be, ro):>11.3e}")
        print(f"    fp32 probs+V (out only)                     "
              f"{rel(upcast_probs(qb, kb, vb).float(), ro):>11.3e}")


run(f"allow_bf16_reduced_precision_reduction = "
    f"{torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
run("allow_bf16_reduced_precision_reduction = False")
