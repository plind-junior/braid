"""Continuous batching. ROADMAP Phase 4 item 1.

The gate is the roadmap's own and it is the strongest correctness test in the
build: **a request's output must not depend on what else was running.** Here it
is stronger than Phase 3's, because sequences join and leave *mid-flight* — a
stream admitted at tick 40 shares its batch with streams at every stage of
generation, and its slot may have been someone else's a moment earlier.

Asserted in **fp32**. In bf16 a B=8 GEMM and a B=1 GEMM pick different tiles and
accumulate in different orders, which moves logits ~1e-2 and flips greedy argmax
on near-ties — measured in Phase 3, not assumed here. That is hardware, not a
scheduler bug, and gating on it would test cuBLAS. The bf16 arm is exercised by
`tests/test_scheduler_bf16.py`, as a tripwire rather than a gate — in its own
module because an fp32 engine (16.8 GB) and a bf16 one (8.4 GB) do not fit
beside each other on a 31 GB card, and pytest holds a module fixture until the
module ends.

Three failures this file is built to catch, each named by the roadmap:

  a slot recycled without a reset      the previous occupant's state leaks
  a sampling parameter read from row 0 invisible at c=1, wrong at c=8
  cancel that frees only one pool      the pool fills up silently
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from conftest import fp32_engine

from braid.serve.scheduler import Request, Scheduler

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

MAX_LEN = 128


@pytest.fixture(scope="module")
def engine_f32():
    return fp32_engine(MODEL_DIR)


def _prompts(n, lo=6, hi=14, seed=41):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randint(0, 1000, (lo + (i * 3) % (hi - lo),), generator=g,
                          device="cuda").tolist() for i in range(n)]


def _alone(engine, prompt, n_new, capacity=1, graphed=False):
    s = Scheduler(engine, capacity=capacity, max_len=MAX_LEN, graphed=graphed)
    return s.run([Request(prompt=prompt, max_new_tokens=n_new)])


# --- the gate -----------------------------------------------------------------

@pytest.mark.parametrize("capacity", [2, 4])
def test_concurrent_streams_match_running_each_alone(engine_f32, capacity):
    n_new = 6
    prompts = _prompts(capacity * 2)
    solo = [next(iter(_alone(engine_f32, p, n_new).values())) for p in prompts]

    sched = Scheduler(engine_f32, capacity=capacity, max_len=MAX_LEN, graphed=False)
    reqs = [Request(prompt=p, max_new_tokens=n_new) for p in prompts]
    together = sched.run(reqs)

    for req, want in zip(reqs, solo):
        assert together[req.id] == want, (
            f"{req.id} diverged at capacity {capacity}:\n"
            f"  together {together[req.id]}\n  alone    {want}")


def test_the_identity_gate_is_not_vacuous(engine_f32):
    """If the pool never decodes more than one row, the gate above proves
    nothing. Assert the concurrency it claims to test actually happened."""
    sched = Scheduler(engine_f32, capacity=4, max_len=MAX_LEN, graphed=False)
    sched.run([Request(prompt=p, max_new_tokens=6) for p in _prompts(8, seed=41)])
    assert sched.max_decode_batch >= 4, (
        f"peak decode batch was {sched.max_decode_batch}; the pool never filled, "
        f"so the concurrency tests are not testing concurrency")


def test_prefill_actually_batches(engine_f32):
    """Same argument as above, for the other half of the tick. Prefill used to
    run one row per forward at a flat 262 tok/s; if it still did, the identity
    gate would be passing over a loop rather than over a ragged batch."""
    sched = Scheduler(engine_f32, capacity=4, max_len=MAX_LEN, graphed=False)
    sched.run([Request(prompt=p, max_new_tokens=4) for p in _prompts(8, seed=41)])
    assert sched.max_prefill_batch >= 4, (
        f"peak prefill batch was {sched.max_prefill_batch}; rows are still being "
        f"prefilled one at a time")


def test_a_long_prompt_is_still_chunked(engine_f32):
    """`prefill_budget` caps total tokens per tick, but the first row is always
    admitted — otherwise a prompt longer than the budget would never start."""
    sched = Scheduler(engine_f32, capacity=2, max_len=MAX_LEN, graphed=False,
                      prefill_chunk=8, prefill_budget=8)
    long_prompt = _prompts(1, lo=40, hi=41, seed=2)[0]
    out = sched.run([Request(prompt=long_prompt, max_new_tokens=3)])
    assert len(next(iter(out.values()))) == 3
    assert sched.prefill_tokens == len(long_prompt)


def test_streams_joining_midflight_are_unaffected(engine_f32):
    """The continuous part: admit late, into a pool that is already busy."""
    n_new = 6
    early, late = _prompts(2, seed=5), _prompts(2, seed=9)
    solo = {i: next(iter(_alone(engine_f32, p, n_new).values()))
            for i, p in enumerate(late)}

    sched = Scheduler(engine_f32, capacity=4, max_len=MAX_LEN, graphed=False)
    out: dict[str, list[int]] = {}
    for p in early:
        r = Request(prompt=p, max_new_tokens=n_new + 8)
        out[sched.submit(r)] = []
    for _ in range(4):                       # let the early ones get going
        for up in sched.step():
            out.setdefault(up.id, []).extend(up.tokens)

    late_reqs = [Request(prompt=p, max_new_tokens=n_new) for p in late]
    for r in late_reqs:
        out[sched.submit(r)] = []
    while not sched.idle:
        for up in sched.step():
            out.setdefault(up.id, []).extend(up.tokens)

    for i, r in enumerate(late_reqs):
        assert out[r.id] == solo[i], f"late joiner {r.id} was perturbed"


def test_a_recycled_slot_does_not_leak_the_previous_occupant(engine_f32):
    """The reference engine's documented leak, as a test.

    A slot is deliberately forced: capacity 1, so the second request can only
    run in the slot the first just vacated. Its output must match a cold run.
    """
    n_new = 6
    first, second = _prompts(1, seed=2)[0], _prompts(1, seed=77)[0]
    cold = next(iter(_alone(engine_f32, second, n_new).values()))

    sched = Scheduler(engine_f32, capacity=1, max_len=MAX_LEN, graphed=False)
    sched.run([Request(prompt=first, max_new_tokens=n_new)])
    assert sched.free_slots == 1, "the finished request did not release its slot"
    warm = sched.run([Request(prompt=second, max_new_tokens=n_new)])
    assert next(iter(warm.values())) == cold, "recycled slot leaked prior state"


def test_cancel_releases_the_slot_and_the_next_request_is_clean(engine_f32):
    """Disconnect mid-generation: both pools must come back."""
    n_new, prompts = 6, _prompts(2, seed=13)
    cold = next(iter(_alone(engine_f32, prompts[1], n_new).values()))

    sched = Scheduler(engine_f32, capacity=1, max_len=MAX_LEN, graphed=False)
    rid = sched.submit(Request(prompt=prompts[0], max_new_tokens=64))
    for _ in range(3):
        sched.step()
    assert sched.free_slots == 0 and rid in sched.live_ids

    assert sched.cancel(rid) is True
    assert sched.free_slots == 1, "cancel did not release the slot"
    assert sched.idle

    after = sched.run([Request(prompt=prompts[1], max_new_tokens=n_new)])
    assert next(iter(after.values())) == cold, "cancelled slot leaked into the next"


def test_cancelling_an_unknown_or_queued_request(engine_f32):
    sched = Scheduler(engine_f32, capacity=1, max_len=MAX_LEN, graphed=False)
    assert sched.cancel("nope") is False
    a = sched.submit(Request(prompt=[1, 2, 3], max_new_tokens=4))
    b = sched.submit(Request(prompt=[4, 5, 6], max_new_tokens=4))
    assert sched.cancel(b) is True          # still queued, never admitted
    assert sched.cancel(a) is True
    assert sched.idle


# --- per-row sampling ---------------------------------------------------------

def test_sampling_parameters_are_per_row(engine_f32):
    """A greedy stream must stay greedy while a hot one runs beside it.

    If temperature were read from row 0, the greedy request would drift; if it
    were read from the greedy row, the hot one would collapse onto argmax.
    """
    n_new, prompts = 8, _prompts(2, seed=23)
    greedy_alone = next(iter(_alone(engine_f32, prompts[0], n_new).values()))

    sched = Scheduler(engine_f32, capacity=2, max_len=MAX_LEN, graphed=False)
    g = Request(prompt=prompts[0], max_new_tokens=n_new, temperature=0.0)
    h = Request(prompt=prompts[1], max_new_tokens=n_new, temperature=1.5,
                top_p=0.95, seed=7)
    out = sched.run([g, h])

    assert out[g.id] == greedy_alone, "the greedy stream picked up its neighbour's heat"
    hot_greedy = next(iter(_alone(engine_f32, prompts[1], n_new).values()))
    assert out[h.id] != hot_greedy, "the sampled stream came out greedy"


def test_a_seeded_stream_is_reproducible(engine_f32):
    prompt = _prompts(1, seed=31)[0]
    runs = []
    for _ in range(2):
        s = Scheduler(engine_f32, capacity=2, max_len=MAX_LEN, graphed=False)
        r = Request(prompt=prompt, max_new_tokens=8, temperature=1.2, seed=99)
        runs.append(s.run([r])[r.id])
    assert runs[0] == runs[1]


# --- lifecycle ----------------------------------------------------------------

def test_more_requests_than_slots_all_complete(engine_f32):
    n_new = 4
    reqs = [Request(prompt=p, max_new_tokens=n_new) for p in _prompts(7, seed=61)]
    sched = Scheduler(engine_f32, capacity=2, max_len=MAX_LEN, graphed=False)
    out = sched.run(reqs)
    assert all(len(out[r.id]) == n_new for r in reqs)
    assert sched.free_slots == 2 and sched.idle


def test_eos_stops_a_stream_and_frees_its_slot(engine_f32):
    prompt = _prompts(1, seed=17)[0]
    probe = Scheduler(engine_f32, capacity=1, max_len=MAX_LEN, graphed=False)
    r0 = Request(prompt=prompt, max_new_tokens=4)
    first = probe.run([r0])[r0.id][0]

    sched = Scheduler(engine_f32, capacity=1, max_len=MAX_LEN, graphed=False)
    r = Request(prompt=prompt, max_new_tokens=8, eos_token_id=first)
    out = sched.run([r])
    assert out[r.id] == [], "the stream should have stopped on its first token"
    assert sched.free_slots == 1 and sched.idle


def test_a_request_that_cannot_fit_is_refused_at_submit(engine_f32):
    sched = Scheduler(engine_f32, capacity=1, max_len=MAX_LEN, graphed=False)
    with pytest.raises(ValueError, match="the pool holds"):
        sched.submit(Request(prompt=[1] * (MAX_LEN - 2), max_new_tokens=16))


def test_an_empty_prompt_is_refused(engine_f32):
    with pytest.raises(ValueError, match="empty prompt"):
        Request(prompt=[], max_new_tokens=4)


def test_long_prompts_are_prefilled_in_chunks(engine_f32):
    """Chunking must not change the answer — the bug fixed on 2026-08-08."""
    prompt = _prompts(1, seed=8)[0] * 6            # ~48-84 tokens
    one_shot = Scheduler(engine_f32, capacity=1, max_len=MAX_LEN, graphed=False,
                         prefill_chunk=4096)
    chunked = Scheduler(engine_f32, capacity=1, max_len=MAX_LEN, graphed=False,
                        prefill_chunk=7)
    a = Request(prompt=prompt, max_new_tokens=6)
    b = Request(prompt=prompt, max_new_tokens=6)
    assert chunked.run([b])[b.id] == one_shot.run([a])[a.id]
