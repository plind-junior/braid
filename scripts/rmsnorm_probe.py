"""Is `F.rms_norm` a faster spelling of braid's `rms_norm`, and the same one?

The decode profile leaves ~1.68 ms/step of elementwise and 0.35 ms of reduce at
B=16, and the RMSNorm chain is most of it by launch count: `pow -> mean ->
rsqrt -> mul -> mul` is 5 kernels, and a step runs ~105 norms (64 block norms +
16 q/k norms + 24 gated + final). PyTorch ships a fused `F.rms_norm`.

Two questions, and the second one decides it: is it faster, and is it the *same
arithmetic*? Braid computes in fp32 and casts on the way out. A fused kernel
that computes in bf16 instead would be a numerics change wearing a perf change's
clothes -- exactly the trade that got the GDN a/b fusion reverted.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.bench.noise_floor import measure_graphed
from braid.model.norm import rms_norm

D, EPS = 2560, 1e-6


def braid(x, g):
    return rms_norm(x, g, EPS)


def fused_f32(x, g):
    """Same fp32-then-cast contract, spelled as one op."""
    return F.rms_norm(x.float(), (D,), g, EPS).type_as(x)


def fused_native(x, g):
    """Handed the activation dtype directly -- whatever PyTorch does internally."""
    return F.rms_norm(x, (D,), g.to(x.dtype), EPS)


for B in (1, 16):
    print(f"\n=== B={B}, hidden={D}")
    g32 = torch.randn(D, device="cuda", dtype=torch.float32).abs() + 0.5
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(B, 1, D, device="cuda", dtype=dtype)
        ref = braid(x, g32)

        rows = []
        for name, fn in (("braid (5 kernels)", lambda: braid(x, g32)),
                         ("F.rms_norm fp32-in", lambda: fused_f32(x, g32)),
                         ("F.rms_norm native ", lambda: fused_native(x, g32))):
            r = measure_graphed(fn, inner=64, reps=30)
            out = fn()
            exact = torch.equal(out, ref)
            rel = ((out.float() - ref.float()).norm()
                   / ref.float().norm()).item() if not exact else 0.0
            rows.append((name, r.median_s * 1e6, exact, rel))

        print(f"  {str(dtype).replace('torch.', ''):<10}"
              f"{'us/call':>10}{'vs braid':>10}  {'bit-exact':>10}  rel")
        base = rows[0][1]
        for name, us, exact, rel in rows:
            print(f"    {name:<20}{us:>8.2f}{base / us:>9.2f}x  "
                  f"{str(exact):>10}  {rel:.3e}")
