"""Chunked prefill — Phase 3 item 3, single sequence per forward.

A prompt fed in pieces must leave the cache in the same state as the same prompt
fed whole. Phase 4's scheduler needs this: a long prompt that must be admitted
in one forward blocks every decoding stream for the duration.

**This suite exists because that was broken and nothing caught it.** The GDN
conv's T>1 path left-padded with zeros and ignored the cached window, so a chunk
landing on a non-empty cache convolved its first `conv_kernel - 1` tokens as if
the sequence started there. Measured at **1.6e-1 relative on the logits, in
fp32** — and the greedy token still agreed, so a token-identity test would have
passed it. For a *fresh* sequence the window is zeros and zero-padding is
correct, which is why every existing prefill test was green.

The gate is fp32 to machine precision, deliberately. In bf16 the two arms issue
differently shaped GEMMs and the noise floor is ~1e-2, which is two orders above
the bug this suite is for.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from braid.model.engine import Engine
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

MAX_LEN, MAX_SLOTS = 96, 4


@pytest.fixture(scope="module")
def engine():
    ck = load_checkpoint(MODEL_DIR, device="cuda", dtype=torch.float32)
    return Engine.from_checkpoint(ck, device="cuda", dtype=torch.float32)


def _fresh(engine):
    c = engine.allocate_cache(MAX_LEN, max_slots=MAX_SLOTS)
    for s in range(MAX_SLOTS):
        c.reset_slot(s)
    return c


def _run(engine, prompt, splits):
    c = _fresh(engine)
    out = None
    for lo, hi in zip((0,) + splits, splits + (prompt.shape[1],)):
        if hi > lo:
            out = engine.forward(prompt[:, lo:hi], c.select([0]))
    return out, c


def _prompt(n=24, seed=21):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, 1000, (1, n), generator=g, device="cuda")


# --- the gate -----------------------------------------------------------------

@pytest.mark.parametrize("splits", [
    (10, 17),        # chunks longer than conv_kernel
    (4, 8, 12, 16),  # chunks exactly conv_kernel
    (1, 2, 3),       # chunks SHORTER than conv_kernel — the window-save path
    (23,),           # a 1-token tail, i.e. prefill then decode
    (12,),
])
def test_chunked_prefill_matches_whole_prefill(engine, splits):
    prompt = _prompt()
    whole, cw = _run(engine, prompt, ())
    part, cp = _run(engine, prompt, splits)

    rel = ((part.float() - whole.float()).norm() / whole.float().norm()).item()
    assert rel < 1e-5, f"splits {splits}: logits differ by {rel:.3e} in fp32"
    assert torch.equal(part.argmax(-1), whole.argmax(-1))
    assert int(cp.lengths[0]) == int(cw.lengths[0]) == prompt.shape[1]


@pytest.mark.parametrize("splits", [(10, 17), (1, 2, 3)])
def test_the_cache_itself_matches_not_just_the_logits(engine, splits):
    """Logits are one token's worth. The conv window and recurrent state are
    what the *next* token reads, and a bug can hide in them for one step."""
    prompt = _prompt()
    _, cw = _run(engine, prompt, ())
    _, cp = _run(engine, prompt, splits)
    for i, (a, b) in enumerate(zip(cp.snapshot(), cw.snapshot())):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5,
                                   msg=f"cache tensor {i} diverged")


def test_a_chunk_shorter_than_the_conv_kernel_keeps_its_history(engine):
    """The regression that was live: the window save zero-padded when T < K,
    dropping however much real history the pad displaced."""
    K = engine.config.gdn.conv_kernel
    prompt = _prompt(n=K * 3)
    whole, _ = _run(engine, prompt, ())
    # Split so that every chunk after the first is shorter than the kernel.
    splits = tuple(range(K, prompt.shape[1], max(1, K - 1)))
    part, _ = _run(engine, prompt, splits)
    rel = ((part.float() - whole.float()).norm() / whole.float().norm()).item()
    assert rel < 1e-5, f"short chunks (K={K}) differ by {rel:.3e}"


def test_prefill_then_decode_continues_the_same_sequence(engine):
    """The path generation actually takes: a prompt, then single tokens."""
    prompt = _prompt(n=20)
    whole, cw = _run(engine, prompt[:, :16], ())
    tok = whole.argmax(-1)[:, -1:]
    for _ in range(4):
        lg = engine.forward(tok, cw.select([0]))
        tok = lg.argmax(-1)[:, -1:]

    ref, cr = _run(engine, prompt[:, :16], ())
    tok2 = ref.argmax(-1)[:, -1:]
    for _ in range(4):
        lg2 = engine.decode_step(tok2, cr.select([0]))
        tok2 = lg2.argmax(-1)[:, None]
    assert torch.equal(tok.flatten(), tok2.flatten())


# --- what is still refused, and refused loudly --------------------------------

def test_a_ragged_batch_without_lengths_is_refused(engine):
    """`seq_lens` is how a caller says the rows are ragged. Without it a `[B, T]`
    forward is a genuine rectangle and every row advances by T — correct, but
    only if that is what the caller meant. It is `seq_lens` describing a batch
    with no cache to address that is meaningless, and refused."""
    ids = torch.randint(0, 1000, (2, 8), device="cuda")
    lens = torch.tensor([8, 5], device="cuda")
    with pytest.raises(ValueError, match="needs a cache"):
        engine.forward(ids, None, seq_lens=lens)


def test_seq_lens_must_match_the_batch(engine):
    c = _fresh(engine)
    ids = torch.randint(0, 1000, (2, 8), device="cuda")
    with pytest.raises(ValueError, match="seq_lens must be"):
        engine.forward(ids, c.select([0, 1]), seq_lens=torch.tensor([8], device="cuda"))
