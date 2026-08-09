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
from braid.model.graph import GraphedDecoder
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

def test_padding_a_partial_batch_leaves_live_slots_alone(engine_bf16):
    """Padding a partial batch must use the scratch slot, not a live one.

    **Asserted on the cache, not on tokens, and that is the point.** A padded
    row still advances whatever slot it names, so the failure this guards is a
    graphed step quietly writing a live sequence's recurrent state and KV — a
    sequence that is not even in the batch. That is an exact, bitwise property
    of the caches, and checking it directly makes the gate independent of dtype.

    This replaces an earlier version that compared the eager and graphed
    schedulers' greedy tokens and asserted equality. It passed for the wrong
    reason. Eager runs the partial batch at B=3 and graphed runs it padded to
    B=4, and a different GEMM batch shape changes bf16's last bits: measured on
    Qwen3.5-9B, the eager-vs-graphed logit residual is 0.06-0.19 while the
    top-2 gap is as low as 0.0625, so at several steps the greedy token is
    decided by rounding rather than by the model. Such a test reports a failure
    for any harmless numerics change anywhere upstream — it flipped when the
    chunk prefill kernel landed, which is a more accurate prefill, not a worse
    one. The file's other bf16 test already says exactly this in its docstring.
    """
    prompts = _prompts(3, seed=101)
    cache = engine_bf16.allocate_cache(MAX_LEN, max_slots=5)
    for s in range(5):
        cache.reset_slot(s)

    # Slots 0-2 are the batch. Slot 3 is a live sequence that is NOT in this
    # batch, and slot 4 is the scratch slot the padded row should name.
    for i, p in enumerate(prompts):
        engine_bf16.forward(torch.tensor(p, device="cuda")[None], cache.select([i]))
    engine_bf16.forward(torch.tensor(_prompts(1, seed=7)[0], device="cuda")[None],
                        cache.select([3]))

    bystander = [layer.state[3].clone() for layer in cache.layers
                 if hasattr(layer, "state")]
    assert bystander, "expected recurrent layers in the cache"

    dec = GraphedDecoder(engine_bf16, cache, buckets=(4,))
    tokens = torch.full((3, 1), 13, dtype=torch.long, device="cuda")
    slots = torch.arange(3, device="cuda")
    scratch = torch.tensor([4], device="cuda")
    for _ in range(6):
        out = dec.step(tokens, slots, pad_slots=scratch)
        tokens = out.argmax(-1).reshape(-1, 1)

    after = [layer.state[3] for layer in cache.layers if hasattr(layer, "state")]
    for i, (before, now) in enumerate(zip(bystander, after)):
        assert torch.equal(before, now), (
            f"layer {i}: the padded row advanced live slot 3's recurrent state")


def test_the_graphed_scheduler_tracks_the_eager_one(engine_bf16):
    """End-to-end sanity on the two scheduler paths, as a tripwire.

    Not token identity: see the note above — eager and graphed run different
    GEMM batch shapes, so in bf16 they disagree wherever the top-2 gap is under
    the shape noise. A real defect (wrong slot, stale KV, a dropped row) does
    not produce near-tie flips, it produces divergence that never recovers.
    """
    n_new, prompts = 6, _prompts(3, seed=101)
    eager = Scheduler(engine_bf16, capacity=4, max_len=MAX_LEN, graphed=False)
    graphed = Scheduler(engine_bf16, capacity=4, max_len=MAX_LEN, graphed=True)
    ra = [Request(prompt=p, max_new_tokens=n_new) for p in prompts]
    rb = [Request(prompt=p, max_new_tokens=n_new) for p in prompts]
    out_a, out_b = eager.run(ra), graphed.run(rb)

    agree = sum(a == b for x, y in zip(ra, rb)
                for a, b in zip(out_a[x.id], out_b[y.id]))
    total = sum(len(out_a[x.id]) for x in ra)
    assert all(len(out_a[x.id]) == n_new for x in ra), "eager dropped tokens"
    assert all(len(out_b[y.id]) == n_new for y in rb), "graphed dropped tokens"
    assert agree / total >= 0.5, (
        f"only {agree}/{total} tokens agree between the eager and graphed "
        f"schedulers — that is beyond bf16 batch-shape noise")


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
