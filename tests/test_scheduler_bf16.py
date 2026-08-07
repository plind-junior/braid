"""The bf16 scheduler arms. Split out of `test_scheduler.py` for memory.

`test_scheduler.py` holds a module-scoped **fp32** engine — 16.8 GB — because its
identity gates have to be asserted in fp32, where batch-shape GEMM noise is
1e-6 rather than 1e-2. A bf16 engine is another 8.4 GB, and pytest keeps a module
fixture alive until the module finishes, so the two together plus allocator cache
overran a 31 GB card. `empty_cache()` did not rescue it and neither did
`expandable_segments` (it got to 21.4 GB allocated and still failed); the fix is
not to hold both at once.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from braid.model.engine import Engine
from braid.model.loader import load_checkpoint
from braid.serve.scheduler import Request, Scheduler

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

MAX_LEN = 128


@pytest.fixture(scope="module")
def engine_bf16():
    ck = load_checkpoint(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    return Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16,
                                  use_kernels=True)


def _prompts(n, lo=6, hi=14, seed=41):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randint(0, 1000, (lo + (i * 3) % (hi - lo),), generator=g,
                          device="cuda").tolist() for i in range(n)]


def _alone(engine, prompt, n_new, capacity=1, graphed=False):
    s = Scheduler(engine, capacity=capacity, max_len=MAX_LEN, graphed=graphed)
    return s.run([Request(prompt=prompt, max_new_tokens=n_new)])


# --- the graphed path ---------------------------------------------------------

def test_graphed_scheduler_matches_the_eager_one(engine_bf16):
    """Padding a partial batch must use the scratch slot, not a live one."""
    n_new, prompts = 6, _prompts(3, seed=101)
    eager = Scheduler(engine_bf16, capacity=4, max_len=MAX_LEN, graphed=False)
    graphed = Scheduler(engine_bf16, capacity=4, max_len=MAX_LEN, graphed=True)
    ra = [Request(prompt=p, max_new_tokens=n_new) for p in prompts]
    rb = [Request(prompt=p, max_new_tokens=n_new) for p in prompts]
    out_a, out_b = eager.run(ra), graphed.run(rb)
    for x, y in zip(ra, rb):
        assert out_a[x.id] == out_b[y.id], "graphed and eager schedulers diverged"


def test_bf16_streams_stay_close(engine_bf16):
    """Tripwire, not a gate: bf16 batch-shape noise flips greedy near-ties."""
    n_new, prompts = 8, _prompts(4, seed=71)
    solo = [next(iter(_alone(engine_bf16, p, n_new).values())) for p in prompts]
    sched = Scheduler(engine_bf16, capacity=4, max_len=MAX_LEN, graphed=False)
    reqs = [Request(prompt=p, max_new_tokens=n_new) for p in prompts]
    out = sched.run(reqs)

    agree = sum(a == b for r, want in zip(reqs, solo)
                for a, b in zip(out[r.id], want))
    total = sum(len(w) for w in solo)
    assert agree / total >= 0.5, (
        f"only {agree}/{total} bf16 tokens agree with the solo run — that is "
        f"beyond GEMM-shape noise")
