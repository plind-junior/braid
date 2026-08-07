"""What does swapping the GDN decode step to the CUDA kernels actually change?

The kernels are not a drop-in for the torch path — they are fp32 throughout and
l2-normalise inside the scan, where the torch path follows HF's bf16-then-widen
order. Both are *more* precise than the reference; neither is bit-identical to
it. This measures the difference at one layer (isolated) and through all 32
(amplified), so the swap is justified by numbers rather than by "kernels are
better".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from braid.model.cache import RecurrentCache
from braid.model.config import ModelConfig
from braid.model.engine import Engine
from braid.model.gdn import GatedDeltaNet
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
GDN_L = 0
PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "def fibonacci(n):",
    "The three primary colours are",
    "Photosynthesis is the process by which plants",
    "Q: What is the boiling point of water at sea level?\nA:",
    "Once upon a time,",
    "The following is a list of the largest cities in the world by population:",
]


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a @ b / (a.norm() * b.norm())).item()


def one_layer(dt):
    """One GDN decode step, kernel vs torch, from identical state."""
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    ck = load_checkpoint(MODEL_DIR, device="cuda", layers=(GDN_L,),
                         include_embeddings=False, dtype=dt)
    w = ck.layer(GDN_L)
    B = 8
    torch_mod = GatedDeltaNet(cfg, w, use_kernels=False)
    kern_mod = GatedDeltaNet(cfg, w, use_kernels=True)

    g = torch.Generator(device="cuda").manual_seed(31)
    x = torch.randn(B, 1, cfg.hidden_size, generator=g, device="cuda", dtype=dt)
    slots = torch.arange(B, device="cuda")
    slots32 = slots.to(torch.int32)

    c_t = RecurrentCache(cfg, B, "cuda", dt)
    c_k = RecurrentCache(cfg, B, "cuda", torch.float32)
    # Identical non-zero starting state, so the decay term is exercised.
    st = torch.randn(B, cfg.gdn.n_heads, cfg.gdn.state_size, cfg.gdn.head_dim,
                     generator=g, device="cuda")
    c_t.state.copy_(st)
    c_k.state.copy_(st)

    with torch.no_grad():
        y_t = torch_mod(x, cache=c_t, slots=slots)
        y_k = kern_mod(x, cache=c_k, slots=slots, slots_i32=slots32)
    r, c = metrics(y_k, y_t)
    rs, cs = metrics(c_k.state, c_t.state)
    print(f"  one layer [{str(dt).split('.')[-1]:8s}] out rel_l2={r:.3e} cos={c:.9f} | "
          f"state rel_l2={rs:.3e} cos={cs:.9f}")


def full_stack(dt):
    """Teacher-forced logits over the whole engine, kernel vs torch."""
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    toks = [tk.encode(p).ids for p in PROMPTS]
    B, n = len(toks), 12
    max_len = max(len(t) for t in toks) + n + 1

    ck = load_checkpoint(MODEL_DIR, device="cuda", dtype=dt)
    engines = {"torch": Engine.from_checkpoint(ck, device="cuda", dtype=dt,
                                               use_kernels=False),
               "kernel": Engine.from_checkpoint(ck, device="cuda", dtype=dt,
                                                use_kernels=True)}
    caches, first = {}, None
    for name, eng in engines.items():
        cache = eng.allocate_cache(max_len, max_slots=B)
        outs = []
        for row, p in enumerate(toks):
            cache.reset_slot(row)
            ids = torch.tensor([p], device="cuda")
            outs.append(int(eng.forward(ids, cache.select([row]))[0, -1].argmax()))
        caches[name] = cache
        first = outs  # prefill is the torch path in both, so these agree

    forced = torch.tensor(first, device="cuda")[:, None]
    worst, flips = 0.0, 0
    for _ in range(n):
        lg = {name: engines[name].forward(forced, caches[name].select(list(range(B))))[:, -1]
              for name in engines}
        r, _ = metrics(lg["kernel"], lg["torch"])
        worst = max(worst, r)
        flips += int((lg["kernel"].argmax(-1) != lg["torch"].argmax(-1)).sum())
        forced = lg["torch"].argmax(-1)[:, None]
    print(f"  32 layers [{str(dt).split('.')[-1]:8s}] teacher-forced {n} steps x B={B}: "
          f"worst rel_l2={worst:.3e}, argmax flips {flips}/{n * B}")

    tokens_t = engines["torch"].generate_batch(toks, max_new_tokens=24, temperature=0.0)
    tokens_k = engines["kernel"].generate_batch(toks, max_new_tokens=24, temperature=0.0)
    same = sum(a == b for a, b in zip(tokens_t, tokens_k))
    print(f"                        free-running greedy: {same}/{B} rows identical")


def main():
    print("=== kernel vs torch GDN decode ===")
    for dt in (torch.float32, torch.bfloat16):
        one_layer(dt)
    print()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for dt in (torch.bfloat16,) if only != "--fp32" else (torch.float32,):
        full_stack(dt)


if __name__ == "__main__":
    main()
