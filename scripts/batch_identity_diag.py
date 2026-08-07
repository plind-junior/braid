"""Why does batched greedy decode diverge from B=1?

Two candidates, and they need different fixes:

  a batch-leakage bug  — row b's result depends on the other rows. Survives a
                         dtype change, shows a large logit residual, and is a
                         defect.
  bf16 near-ties       — a B=8 GEMM and a B=1 GEMM accumulate in different
                         orders, so logits differ by ~1e-3; argmax flips only
                         where the top two candidates are within that. Vanishes
                         in fp32 and is a property of the hardware.

Discriminators, in order of decisiveness:
  1. the same comparison in fp32
  2. logit residual at the first batched decode step
  3. the top-2 logit gap at each divergence, against that residual
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from braid.model.engine import Engine
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
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


def main():
    dt = torch.float32 if "--fp32" in sys.argv else torch.bfloat16
    n = 32
    print(f"=== dtype {str(dt).split('.')[-1]} ===")

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    toks = [tk.encode(p).ids for p in PROMPTS]

    ck = load_checkpoint(MODEL_DIR, device="cuda", dtype=dt)
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=dt)

    batched = eng.generate_batch(toks, max_new_tokens=n, temperature=0.0)
    seq = [eng.generate_batch([p], max_new_tokens=n, temperature=0.0)[0] for p in toks]

    print("\n-- token identity --")
    bad = 0
    for r, (b, s) in enumerate(zip(batched, seq)):
        d = next((i for i, (x, y) in enumerate(zip(b, s)) if x != y), None)
        bad += d is not None
        print(f"  row {r}: {'identical' if d is None else f'diverges at {d}/{n}'}")
    print(f"  {len(batched) - bad}/{len(batched)} rows identical")

    # --- logit residual at the first batched decode step ----------------------
    print("\n-- first batched decode step: logits, B=8 vs B=1 --")
    B = len(toks)
    max_len = max(len(t) for t in toks) + n + 1

    cache8 = eng.allocate_cache(max_len, max_slots=B)
    first = []
    for row, p in enumerate(toks):
        cache8.reset_slot(row)
        ids = torch.tensor([p], device="cuda")
        first.append(eng.forward(ids, cache8.select([row]))[0, -1].argmax().item())
    nxt = torch.tensor(first, device="cuda")[:, None]
    lg8 = eng.forward(nxt, cache8.select(list(range(B))))[:, -1]

    for row, p in enumerate(toks):
        c1 = eng.allocate_cache(max_len, max_slots=1)
        c1.reset_slot(0)
        ids = torch.tensor([p], device="cuda")
        eng.forward(ids, c1.select([0]))
        lg1 = eng.forward(torch.tensor([[first[row]]], device="cuda"), c1.select([0]))[:, -1]

        r, c = metrics(lg8[row], lg1[0])
        top2 = lg8[row].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        agree = lg8[row].argmax().item() == lg1[0].argmax().item()
        print(f"  row {row}: rel_l2={r:.3e} cosine={c:.9f} "
              f"top2_gap={gap:.4f} argmax {'==' if agree else '!= <-- FLIP'}")


if __name__ == "__main__":
    main()
