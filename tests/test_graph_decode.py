"""CUDA-graph capture of the decode step. ROADMAP Phase 3 item 2.

Secondary gates the roadmap names, and what each is actually worth:

  replay bit-identical to eager, every bucket   the correctness gate
  a deliberate `.item()` makes capture FAIL     proves the audit is real
  slot reassignment needs no re-capture         the reason for the indirection
  graphs_on / graphs_off >= 1.3                 measured separately, with the
                                                benchmarking contract

"Bit-identical" is the right word here and not hyperbole: a graph replays the
exact same kernels on the exact same addresses, so anything less than equality
means the capture did not record what eager does.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from braid.model.engine import Engine
from braid.model.graph import GraphedDecoder
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

DTYPE = torch.bfloat16
MAX_SLOTS, MAX_LEN = 20, 96
BUCKETS = (1, 2, 4, 8)


@pytest.fixture(scope="module")
def engine():
    ck = load_checkpoint(MODEL_DIR, device="cuda")
    return Engine.from_checkpoint(ck, device="cuda", dtype=DTYPE, use_kernels=True)


def _seeded_cache(engine, n_rows: int):
    """A cache whose slots hold real, differing state — zeros would hide bugs."""
    cache = engine.allocate_cache(MAX_LEN, max_slots=MAX_SLOTS)
    g = torch.Generator(device="cuda").manual_seed(5)
    for slot in range(MAX_SLOTS):
        cache.reset_slot(slot)
    for row in range(n_rows):
        ids = torch.randint(0, 1000, (1, 6 + row), generator=g, device="cuda")
        engine.forward(ids, cache.select([row]))
    return cache


@pytest.fixture(scope="module")
def graphed(engine):
    cache = _seeded_cache(engine, MAX_SLOTS)
    return GraphedDecoder(engine, cache, buckets=BUCKETS), cache


# --- the correctness gate -----------------------------------------------------

@pytest.mark.parametrize("size", BUCKETS)
def test_replay_is_bit_identical_to_eager(engine, graphed, size):
    dec, cache = graphed
    tokens = torch.arange(100, 100 + size, device="cuda")[:, None]
    slots = torch.arange(size, device="cuda")

    # A decode step advances state, conv, KV and lengths — the whole cache has
    # to be rewound between the two arms, not just the lengths.
    before = cache.snapshot()
    eager = engine.decode_step(tokens, cache.select(slots.tolist()))
    after_eager = cache.snapshot()

    cache.restore(before)
    replay = dec.step(tokens, slots)

    torch.testing.assert_close(replay, eager, rtol=0, atol=0)
    for got, want in zip(cache.snapshot(), after_eager):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    cache.restore(before)


def test_every_bucket_was_captured(graphed):
    dec, _ = graphed
    assert sorted(dec.buckets) == sorted(BUCKETS)
    assert dec.bucket_for(3) == 4 and dec.bucket_for(5) == 8 and dec.bucket_for(8) == 8


def test_padding_uses_scratch_slots_and_leaves_live_ones_alone(engine, graphed):
    """A padded row still advances whatever slot it names."""
    dec, cache = graphed
    B, size = 3, 4
    tokens = torch.arange(7, 7 + B, device="cuda")[:, None]
    slots = torch.arange(B, device="cuda")
    scratch = torch.tensor([MAX_SLOTS - 1], device="cuda")

    snap = cache.snapshot()
    before = cache.lengths.clone()
    dec.step(tokens, slots, pad_slots=scratch)
    moved = (cache.lengths != before).nonzero().flatten().tolist()
    assert moved == [0, 1, 2, MAX_SLOTS - 1], f"padding touched {moved}"
    cache.restore(snap)


def test_padding_without_scratch_slots_is_refused(graphed):
    dec, _ = graphed
    with pytest.raises(ValueError, match="pad_slots"):
        dec.step(torch.zeros(3, 1, dtype=torch.long, device="cuda"),
                 torch.arange(3, device="cuda"))


# --- the indirection ----------------------------------------------------------

def test_slot_reassignment_needs_no_recapture(engine, graphed):
    """The claim the whole layout exists for: reassigning slots is a `copy_`."""
    dec, cache = graphed
    size = 4
    tokens = torch.arange(50, 50 + size, device="cuda")[:, None]

    for assignment in ([0, 1, 2, 3], [7, 3, 11, 1], [12, 0, 5, 9]):
        slots = torch.tensor(assignment, device="cuda")
        before = cache.snapshot()
        eager = engine.decode_step(tokens, cache.select(assignment))
        after = cache.snapshot()

        cache.restore(before)
        replay = dec.step(tokens, slots)
        torch.testing.assert_close(replay, eager, rtol=0, atol=0)
        for got, want in zip(cache.snapshot(), after):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        cache.restore(before)


# --- the audit is real --------------------------------------------------------

def test_a_host_sync_makes_capture_fail_loudly(engine):
    """A deliberately inserted `.item()` must break capture, not slip through.

    Without this the no-sync claim is unfalsifiable: a suite that only ever
    captures working code cannot tell whether capture would have caught a sync.
    """
    cache = engine.allocate_cache(MAX_LEN, max_slots=2)
    cache.reset_slot(0)
    ids = torch.randint(0, 1000, (1, 5), device="cuda")
    engine.forward(ids, cache.select([0]))

    original = engine.decode_step

    def sync_step(tokens, c):
        out = original(tokens, c)
        # The exact thing hidden_states does and decode_step avoids.
        if int(c.lengths.max().item()) >= 0:
            pass
        return out

    engine.decode_step = sync_step
    try:
        with pytest.raises(RuntimeError):
            GraphedDecoder(engine, cache, buckets=(1,), warmup=1)
    finally:
        engine.decode_step = original


def test_capture_rejects_a_bucket_larger_than_the_pool(engine):
    cache = engine.allocate_cache(MAX_LEN, max_slots=2)
    with pytest.raises(ValueError, match="no bucket fits"):
        GraphedDecoder(engine, cache, buckets=(8,), warmup=1)


# --- generation through the graph ---------------------------------------------

def test_graphed_generation_matches_eager(engine):
    """Many replays in a row, against the same steps run eagerly."""
    n, B = 12, 4
    cache = engine.allocate_cache(MAX_LEN, max_slots=MAX_SLOTS)
    for slot in range(MAX_SLOTS):
        cache.reset_slot(slot)
    g = torch.Generator(device="cuda").manual_seed(17)
    prompts = [torch.randint(0, 1000, (1, 5 + i), generator=g, device="cuda")
               for i in range(B)]
    for row, ids in enumerate(prompts):
        engine.forward(ids, cache.select([row]))

    snapshot = cache.snapshot()

    slots = torch.arange(B, device="cuda")
    tokens = torch.full((B, 1), 42, dtype=torch.long, device="cuda")
    eager_out = []
    for _ in range(n):
        lg = engine.decode_step(tokens, cache.select(slots.tolist()))
        tokens = lg.argmax(-1)[:, None]
        eager_out.append(tokens.clone())

    # Rewind and repeat through the graph. Capture itself is snapshot-safe, but
    # rewind again after it so both arms start from the same place.
    cache.restore(snapshot)
    dec = GraphedDecoder(engine, cache, buckets=(B,))
    cache.restore(snapshot)

    tokens = torch.full((B, 1), 42, dtype=torch.long, device="cuda")
    for step in range(n):
        lg = dec.step(tokens, slots)
        tokens = lg.argmax(-1)[:, None]
        assert torch.equal(tokens, eager_out[step]), f"diverged at step {step}"
