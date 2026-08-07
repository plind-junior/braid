"""What does Phase 3 item 3 actually still need?

Item 3 is "KV block manager + chunked prefill, single sequence per forward",
plus bucketing `kv_len` and removing the gather/scatter. Before building any of
it, this establishes which parts are already reachable, because the code carries
two `NotImplementedError`s whose scope is easy to overstate:

  attention.py   T>1 onto a non-empty cache *with no explicit mask*
  cache.py       multi-token write at B>1  (ragged batched prefill)

`hidden_states` already builds a general `[B, 1, T, kv_len]` mask whenever rows
differ in length or T != kv_len, so the first guard may never fire from the real
path. If single-sequence chunked prefill already works, item 3 is smaller than
it looks and the effort belongs on `kv_len` and the gather.

The gate is the one that matters: a prompt fed in chunks must produce the same
state as the same prompt fed whole.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from braid.model.engine import Engine
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
DT = torch.float32          # fp32: bf16 GEMM-shape noise would mask a real bug

ck = load_checkpoint(MODEL_DIR, device="cuda", dtype=DT)
eng = Engine.from_checkpoint(ck, device="cuda", dtype=DT)

g = torch.Generator(device="cuda").manual_seed(21)
prompt = torch.randint(0, 1000, (1, 24), generator=g, device="cuda")

# --- 1. whole-prompt reference ------------------------------------------------
c1 = eng.allocate_cache(64, max_slots=2)
c1.reset_slot(0)
whole = eng.forward(prompt, c1.select([0]))
print(f"whole prompt   : logits {tuple(whole.shape)}, len={int(c1.lengths[0])}")

# --- 2. the same prompt in chunks ---------------------------------------------
c2 = eng.allocate_cache(64, max_slots=2)
c2.reset_slot(0)
try:
    for lo, hi in ((0, 10), (10, 17), (17, 24)):
        out = eng.forward(prompt[:, lo:hi], c2.select([0]))
    rel = ((out.float() - whole.float()).norm() / whole.float().norm()).item()
    ok = torch.equal(out.argmax(-1), whole.argmax(-1))
    print(f"chunked (B=1)  : WORKS. rel {rel:.3e}, argmax identical: {ok}, "
          f"len={int(c2.lengths[0])}")
except NotImplementedError as e:
    print(f"chunked (B=1)  : REFUSED -- {str(e)[:90]}")

# --- 3. ragged batched prefill (B>1, T>1) -------------------------------------
c3 = eng.allocate_cache(64, max_slots=4)
for s in range(4):
    c3.reset_slot(s)
try:
    eng.forward(torch.randint(0, 1000, (2, 8), generator=g, device="cuda"),
                c3.select([0, 1]))
    print("ragged batched : WORKS")
except NotImplementedError as e:
    print(f"ragged batched : REFUSED -- {str(e)[:90]}")

# --- 4. what decode_step currently costs in wasted KV -------------------------
c4 = eng.allocate_cache(512, max_slots=16)
for s in range(16):
    c4.reset_slot(s)
for row in range(16):
    eng.forward(torch.randint(0, 1000, (1, 8), generator=g, device="cuda"),
                c4.select([row]))
live = int(c4.lengths[:16].max())
print(f"\ndecode_step reads kv_len = max_len = {c4.max_len} while the longest live "
      f"row is {live}\n  -> {c4.max_len / max(live, 1):.1f}x the KV traffic it needs "
      f"at this length")
