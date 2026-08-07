"""Which SDPA backend actually runs at head_dim=256, per shape?

braid has published, in `attention.py` and `docs/runbooks/decode-profile.md`,
that "every fused SDPA backend on this box declines head_dim=256". An `ncu`
kernel listing shows `flash_fwd_kernel` among the kernels a decode benchmark
launches, which that claim does not allow. This asks each backend directly,
per shape, instead of inferring from one error message.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, KVH, D = 2, 16, 4, 256
SCALE = D ** -0.5

BACKENDS = [
    ("flash", SDPBackend.FLASH_ATTENTION),
    ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
    ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ("math", SDPBackend.MATH),
]

# (label, q_len, kv_len, causal, gqa) -- the shapes braid actually issues.
CASES = [
    ("prefill T=8  causal", 8, 8, True, True),
    ("prefill T=512 causal", 512, 512, True, True),
    ("decode  T=1  masked", 1, 512, False, True),
    ("decode  T=1  no gqa", 1, 512, False, False),
]


def probe(q_len, kv_len, causal, gqa, backend):
    heads_kv = KVH if gqa else H
    q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, heads_kv, kv_len, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, heads_kv, kv_len, D, device="cuda", dtype=torch.bfloat16)
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=SCALE,
            enable_gqa=gqa and heads_kv != H)
    torch.cuda.synchronize()
    return out


print(f"head_dim={D}, bf16, {torch.cuda.get_device_name(0)}, torch {torch.__version__}\n")
print(f"{'shape':<22}" + "".join(f"{n:>16}" for n, _ in BACKENDS))
for label, ql, kl, causal, gqa in CASES:
    row = f"{label:<22}"
    for name, be in BACKENDS:
        try:
            probe(ql, kl, causal, gqa, be)
            row += f"{'OK':>16}"
        except Exception as e:
            msg = str(e).split("\n")[0]
            row += f"{type(e).__name__[:14]:>16}"
    print(row)

print("\nWhat the default dispatcher picks (no sdpa_kernel context):")
for label, ql, kl, causal, gqa in CASES:
    heads_kv = KVH if gqa else H
    q = torch.randn(B, H, ql, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, heads_kv, kl, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, heads_kv, kl, D, device="cuda", dtype=torch.bfloat16)
    fn = lambda: F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=SCALE, enable_gqa=gqa and heads_kv != H)
    fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
    names = sorted({e.key for e in prof.key_averages()
                    if float(e.self_device_time_total) > 0}, )
    picked = [n for n in names if "flash" in n or "fmha" in n or "cudnn" in n
              or "efficient" in n or "attention" in n.lower()]
    print(f"  {label:<22} {picked if picked else 'math (no fused kernel present)'}")
