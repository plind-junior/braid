"""Ragged batched prefill — ROADMAP Phase 5 item 5.

Rows of a prefill batch sit at different positions and carry different numbers
of real tokens. braid runs them as a right-padded `[B, T]` rectangle plus a
`seq_lens[B]` vector; the claim this suite exists to hold is that **a row's
result does not depend on what it was batched with, or on how wide the padding
made the rectangle.**

Two gates, and the second is the sharper one:

1. *Parity* — a ragged batch matches prefilling each row alone, in fp32 to
   machine precision, in the logits **and** in the cache the next token reads.
   Logits alone are one token's worth; a conv window or a recurrent state
   advanced one step too far stays invisible for exactly one step, which is the
   failure mode `tests/test_chunked_prefill.py` was written after.

2. *Padding is inert* — running the same batch with a different pad token id
   must be **bit-identical** on the real rows. That is a stronger statement than
   a tolerance and it is the one that can actually be made: right padding sits
   outside every real token's causal receptive field, pad steps are an exact
   identity on the recurrent state (`alpha = 1, beta = 0`), and padded KV
   columns are written to a sink position that `read` never returns. If any of
   those three is wrong the pad id leaks and `torch.equal` sees it. A tolerance
   would not — the leak from a single pad token is small, fluent, and about the
   size of ordinary fp32 noise.

fp32 throughout, for the reason the rest of the suite is: in bf16 the ragged arm
and the solo arm issue differently shaped GEMMs and accumulate ~1e-2 apart,
which is two orders above anything here.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from conftest import fp32_engine

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

MAX_LEN, MAX_SLOTS = 96, 8


@pytest.fixture(scope="module")
def engine():
    return fp32_engine(MODEL_DIR)


def _fresh(engine):
    c = engine.allocate_cache(MAX_LEN, max_slots=MAX_SLOTS)
    for s in range(MAX_SLOTS):
        c.reset_slot(s)
    return c


def _prompts(lens, seed=17):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randint(0, 1000, (n,), generator=g, device="cuda").tolist()
            for n in lens]


def _pad(prompts, pad_id=0):
    """Right-pad to the longest -> `(ids [B, T], seq_lens [B])`."""
    width = max(len(p) for p in prompts)
    ids = torch.tensor([p + [pad_id] * (width - len(p)) for p in prompts],
                       dtype=torch.long, device="cuda")
    lens = torch.tensor([len(p) for p in prompts], device="cuda")
    return ids, lens


def _solo(engine, prompt, slot=0):
    """Prefill one prompt into its own fresh cache. `(logits [vocab], cache)`."""
    c = _fresh(engine)
    ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
    return engine.forward(ids, c.select([slot]))[0, -1], c


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# --- gate 1: a ragged batch matches running each row alone --------------------

LEN_SETS = [
    (12, 7, 20, 3),      # the general case: four different lengths
    (5, 5, 5),           # a true rectangle, taken through the ragged path
    (24, 1),             # one row is a single token beside a long one
    (2, 19),             # short first, so the widest row is not row 0
]


@pytest.mark.parametrize("lens", LEN_SETS)
def test_ragged_batch_matches_each_row_alone(engine, lens):
    prompts = _prompts(lens)
    want = [_solo(engine, p)[0] for p in prompts]

    cache = _fresh(engine)
    ids, seq_lens = _pad(prompts)
    got = engine.forward(ids, cache.select(list(range(len(prompts)))),
                         seq_lens=seq_lens)[:, -1]

    for b, w in enumerate(want):
        rel = _rel(got[b], w)
        assert rel < 1e-5, f"row {b} (len {lens[b]}) differs by {rel:.3e} in fp32"
        assert int(got[b].argmax()) == int(w.argmax())
    assert cache.lengths[:len(lens)].tolist() == list(lens)


# Relative L2 per cache tensor, ragged arm vs solo arm. **Not** an elementwise
# tolerance: the two arms issue the same GEMMs at different M (80 rows against
# 12), so cuBLAS picks different tiles and accumulates in a different order, and
# the residual lands on the *large* elements. An `atol` tight enough to be
# meaningful near zero then fails on a V element of magnitude 33 that is correct
# to six figures. Measured worst over 32 layers x 4 rows: 5.2e-6, at layer 22.
# The gate sits an order above that and four orders below the 1.6e-1 that
# chunked prefill was wrong by before it was caught.
CACHE_REL_MAX = 5e-5


@pytest.mark.parametrize("lens", [(12, 7, 20, 3), (24, 1)])
def test_the_cache_matches_too_not_just_the_logits(engine, lens):
    """What the *next* token reads. A conv window that kept pad columns, or a
    state advanced past a short row's prompt, is silent for exactly one step.

    This is the *loose* gate of the two in this file, deliberately — it compares
    across differently shaped GEMMs and so can only see gross error. The claims
    specific to raggedness are gated bitwise below.
    """
    prompts = _prompts(lens)
    solo = [_solo(engine, p, slot=b)[1] for b, p in enumerate(prompts)]

    cache = _fresh(engine)
    ids, seq_lens = _pad(prompts)
    engine.forward(ids, cache.select(list(range(len(prompts)))), seq_lens=seq_lens)

    worst = 0.0
    for b in range(len(lens)):
        n = lens[b]
        for i, lyr in enumerate(cache.layers):
            ref = solo[b].layers[i]
            if hasattr(lyr, "state"):
                pairs = [("state", lyr.state[b], ref.state[b]),
                         ("conv", lyr.conv[b], ref.conv[b])]
            else:
                pairs = [("K", lyr.k[b, :, :n], ref.k[b, :, :n]),
                         ("V", lyr.v[b, :, :n], ref.v[b, :, :n])]
            for name, got, want in pairs:
                rel = _rel(got, want)
                worst = max(worst, rel)
                assert rel < CACHE_REL_MAX, (
                    f"row {b} (len {n}): {name} of layer {i} differs by {rel:.3e}")
    print(f"\n  lens {lens}: worst cache rel_l2 {worst:.3e}")


def test_ragged_chunks_onto_a_non_empty_cache(engine):
    """Rows at different *positions* as well as different lengths.

    The case with the most ways to be wrong: the conv window has to be spliced
    per row and re-saved from a per-row offset, and `is_causal` is no longer
    available because the rows no longer share a start.
    """
    prompts = _prompts((22, 15, 9), seed=31)
    want = [_solo(engine, p)[0] for p in prompts]

    cache = _fresh(engine)
    view = cache.select([0, 1, 2])
    # Feed each prompt in pieces, ragged at every step. The per-row cut points
    # differ so no two rows are ever the same length or at the same position.
    cuts = [(0, 7, 13, 22), (0, 5, 11, 15), (0, 4, 6, 9)]
    for step in range(3):
        piece = [p[c[step]:c[step + 1]] for p, c in zip(prompts, cuts)]
        ids, seq_lens = _pad(piece)
        got = engine.forward(ids, view, seq_lens=seq_lens)[:, -1]

    for b, w in enumerate(want):
        rel = _rel(got[b], w)
        assert rel < 1e-5, f"row {b} differs by {rel:.3e} after ragged chunking"
        assert int(got[b].argmax()) == int(w.argmax())
    assert cache.lengths[:3].tolist() == [22, 15, 9]


# --- gate 2: the padding is inert, bit-for-bit --------------------------------

def test_the_pad_token_id_cannot_reach_a_real_row(engine):
    """Bit-identical under a different pad id, which is the whole safety claim.

    Three independent mechanisms have to hold for this to pass, and each fails
    fluently on its own: a causal conv that reads forward, a pad step that moves
    the recurrent state, and a padded KV column landing on a live position.
    """
    prompts = _prompts((13, 6, 18, 2), seed=77)
    runs = []
    for pad_id in (0, 999):
        cache = _fresh(engine)
        ids, seq_lens = _pad(prompts, pad_id=pad_id)
        logits = engine.forward(ids, cache.select([0, 1, 2, 3]), seq_lens=seq_lens)
        runs.append((logits[:, -1].clone(), cache))

    (a, ca), (b, cb) = runs
    assert torch.equal(a, b), (
        f"the pad token id changed a real row's logits by "
        f"{_rel(a, b):.3e} — padding is leaking")
    for i, (la, lb) in enumerate(zip(ca.layers, cb.layers)):
        if hasattr(la, "state"):
            assert torch.equal(la.state, lb.state), f"layer {i} state leaked pad"
            assert torch.equal(la.conv, lb.conv), f"layer {i} conv window leaked pad"
        else:
            for n, (ka, kb) in enumerate(((la.k, lb.k), (la.v, lb.v))):
                for row, ln in enumerate((13, 6, 18, 2)):
                    assert torch.equal(ka[row, :, :ln], kb[row, :, :ln]), (
                        f"layer {i} {'KV'[n]} row {row} leaked pad")


def test_a_wider_rectangle_does_not_move_a_row(engine):
    """The same row, padded to different widths: 0, 6 and 30 identity steps.

    Separates "padding is masked" from "padding happens to be harmless at this
    width". Unlike the test above this one **cannot** be asserted bitwise, and
    the reason is worth stating because it looks like a weaker test and is not:
    widening the rectangle changes the GEMMs' M, cuBLAS picks its tiles and
    split-k from M, and a different split-k changes the reduction order along K
    for *every* row including the real one. Measured at 1.0e-6 on the logits.
    Holding the shape fixed and varying only the pad values — which is what the
    test above does — leaves the tiling alone and is bit-exact.

    What it still discriminates is the thing it is for. An unmasked pad step
    multiplies the state by `alpha < 1` and adds a delta-rule update; thirty of
    them move the state by order 1, not by 1e-6.
    """
    prompt = _prompts((14,), seed=5)[0]
    outs, states = [], []
    for width in (14, 20, 44):
        cache = _fresh(engine)
        ids = torch.tensor([prompt + [0] * (width - len(prompt))],
                           dtype=torch.long, device="cuda")
        lens = torch.tensor([len(prompt)], device="cuda")
        outs.append(engine.forward(ids, cache.select([0]), seq_lens=lens)[0, -1])
        states.append(cache.layers[0].state[0].clone())
        # This one *is* exact, and it is the bookkeeping half of the claim.
        assert int(cache.lengths[0]) == len(prompt)

    for i in (1, 2):
        lg, st = _rel(outs[i], outs[0]), _rel(states[i], states[0])
        assert lg < 1e-4, f"width {(20, 44)[i - 1]} moved the logits by {lg:.3e}"
        assert st < 1e-4, f"width {(20, 44)[i - 1]} advanced the state by {st:.3e}"
        print(f"\n  width {(20, 44)[i - 1]} vs 14: logits {lg:.3e}, state {st:.3e}")


# --- the batched path is what generate_batch and the scheduler now take -------

def test_generate_batch_prefills_ragged(engine):
    """End to end, through the public API, with prompts of different lengths."""
    prompts = _prompts((11, 4, 17), seed=3)
    together = engine.generate_batch(prompts, max_new_tokens=6)
    alone = [engine.generate_batch([p], max_new_tokens=6)[0] for p in prompts]
    for b, (t, a) in enumerate(zip(together, alone)):
        assert t == a, f"row {b} diverged: batched {t} vs alone {a}"
