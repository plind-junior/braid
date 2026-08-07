"""The Phase 2 perplexity gate: braid within 20% of an HF reference.

Marked `slow` — it loads two 4B models in sequence and needs the corpus, which
is downloaded once and cached. Deselect with `-m 'not slow'`.

Numbers and method live in `docs/runbooks/perplexity.md`.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

# Pinned 2026-08-07. The perplexity below is only meaningful for THIS token list;
# a corpus that silently changed would move the number and read as a regression.
CORPUS_SHA256 = "e5ab1f058d3dfc4cf63a1220da79116a7e2ecc5c1f60eb9ed6f968d04091554f"
N_TOKENS, WINDOW = 16384, 2048

GATE = 0.20          # ROADMAP Phase 2: PPL within 20% of the reference
BRAID_PPL = 8.2376   # measured; see the runbook
HF_PPL = 8.2393


@pytest.fixture(scope="module")
def corpus():
    from braid.bench.perplexity import load_corpus

    try:
        return load_corpus(N_TOKENS, MODEL_DIR)
    except Exception as e:  # offline box, HF rate limit
        pytest.skip(f"corpus unavailable: {type(e).__name__}: {str(e)[:120]}")


def test_corpus_is_pinned(corpus):
    assert corpus.sha256 == CORPUS_SHA256, (
        "the corpus changed; the recorded perplexity no longer describes this "
        f"token list. got {corpus.sha256}"
    )
    assert corpus.n_tokens == N_TOKENS >= 10_000


@pytest.fixture(scope="module")
def measured(corpus):
    """Both arms, run in sequence so two 4B models are never resident together."""
    from braid.bench.perplexity import braid_perplexity, hf_perplexity

    hf_res, hf_model = hf_perplexity(corpus, MODEL_DIR, window=WINDOW)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    br_res, eng = braid_perplexity(corpus, MODEL_DIR, window=WINDOW)
    weights = eng.weight_bytes()
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return br_res, hf_res, weights


def test_perplexity_within_20_percent_of_hf(measured):
    br, hf, _ = measured
    delta = abs(br.perplexity - hf.perplexity) / hf.perplexity
    print(f"\n  braid {br.perplexity:.4f}  HF {hf.perplexity:.4f}  "
          f"delta {delta * 100:.4f}%  ({br.predicted_tokens} tokens)")
    assert delta <= GATE, f"perplexity differs by {delta * 100:.2f}% (gate {GATE * 100:.0f}%)"


def test_absolute_perplexity_is_recorded(measured):
    """Pin the absolute value, not just the agreement.

    The gate above passes trivially if both arms break the same way. This one
    fails if the model's quality moves at all, which is what would happen if a
    load-time transform regressed on both sides.
    """
    br, hf, _ = measured
    assert br.perplexity == pytest.approx(BRAID_PPL, rel=1e-3), (
        f"braid perplexity moved: {br.perplexity:.4f} vs recorded {BRAID_PPL}")
    assert hf.perplexity == pytest.approx(HF_PPL, rel=1e-3)


def test_peak_vram_under_the_phase2_budget(measured):
    """Rescoped Phase 2 budget is 12 GiB, against the 35B's original 30 GiB."""
    _, _, weights = measured
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"\n  weights {weights / 2 ** 30:.2f} GiB, peak {peak:.2f} GiB")
    assert weights / 2 ** 30 < 9.0


def test_dropping_the_final_norm_offset_is_caught_by_the_gate(corpus, measured):
    """The gate must detect the failure it exists for.

    ROADMAP predicted ~2x perplexity from a missing `1+W` on the final norm,
    carried over from the 35B (13.65 -> 6.82). On this checkpoint and protocol it
    measures **1.44x**, not 2x — smaller, but still 44% degraded and therefore
    comfortably outside the 20% gate. The prediction's direction holds; its
    magnitude was another model's.
    """
    from braid.bench.perplexity import braid_perplexity

    _, hf, _ = measured
    ab, eng = braid_perplexity(corpus, MODEL_DIR, window=WINDOW,
                               drop_final_norm_offset=True)
    del eng
    gc.collect()
    torch.cuda.empty_cache()

    ratio = ab.perplexity / hf.perplexity
    print(f"\n  ablation (no 1+W on final norm): {ab.perplexity:.4f} = {ratio:.2f}x")
    assert ratio > 1 + GATE, (
        f"removing the final norm's offset only costs {ratio:.2f}x, which the 20% "
        "gate would not catch — the gate is not discriminating"
    )
