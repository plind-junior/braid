"""The batch axis in **bf16** — the tripwires, and the slot indirection.

Split from `test_batched_decode.py`, which holds the fp32 gate. The two cannot
share a process at Qwen3.5-9B and the reason is arithmetic rather than
incidental: an fp32 copy of this checkpoint's embeddings alone is 8.14 GiB
(248,320 x 4,096, untied, x4 bytes x2 tensors), so an fp32 engine and a bf16 one
do not both fit on a 31.4 GiB card however far the fp32 stack is truncated.
`scripts/test_isolated.sh` gives each module its own process; the autouse
reclaim in `conftest.py` covers the case where the whole suite runs in one.

What lives here is everything asserted in bf16:

  1. **Teacher-forced logit agreement.** Feed a B=8 decode and eight B=1 decodes
     the *same* tokens, so sampling cannot amplify, and bound the per-step
     residual. A real leak moves this immediately; cuBLAS picking a different
     tile for a different M does not. Free-running bf16 identity is *reported,
     not asserted* — gating on it would be gating on tile heuristics.
  2. **The indirection.** A non-identity slot permutation must change nothing,
     and a slot handed to a new sequence must not carry the old one's state.
     Those are what make one captured CUDA graph valid across reassignment.
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

DTYPE = torch.bfloat16

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


@pytest.fixture(scope="module")
def engine():
    ck = load_checkpoint(MODEL_DIR, device="cuda")
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=DTYPE)
    del ck
    return eng


@pytest.fixture(scope="module")
def tokens():
    tok = pytest.importorskip("tokenizers")
    t = tok.Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    return [t.encode(p).ids for p in PROMPTS]


def _metrics(mine: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    a, b = mine.double().flatten(), ref.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a @ b / (a.norm() * b.norm())).item()


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _report(batched, sequential, n) -> str:
    lines = []
    for r, (bb, ss) in enumerate(zip(batched, sequential)):
        d = _first_divergence(bb, ss)
        lines.append(f"    row {r}: {'identical' if d is None else f'diverges at {d}/{n}'}")
    return "\n".join(lines)


# --- bf16: teacher-forced logit agreement --------------------------------------

# Per-step relative residual between a B=8 decode and eight B=1 decodes fed the
# same tokens. Measured ~1e-2, matching Phase 2's B=1 decode-vs-prefill drift.
# A tripwire on cuBLAS tile selection, not a correctness threshold.
BF16_BATCH_DRIFT_MAX = 3e-2


def test_bf16_teacher_forced_logits_agree(engine, tokens):
    """Same tokens into both paths, so sampling cannot amplify the difference.

    This is where a genuine leak shows up: if row b's logits depended on its
    neighbours, feeding identical inputs would still diverge, and it would
    diverge by far more than tile-selection noise.
    """
    n, B = 12, len(tokens)
    max_len = max(len(t) for t in tokens) + n + 1

    # Prefill both arms identically, one sequence at a time.
    cache8 = engine.allocate_cache(max_len, max_slots=B)
    singles = []
    forced = []
    for row, p in enumerate(tokens):
        cache8.reset_slot(row)
        ids = torch.tensor([p], device="cuda")
        lg = engine.forward(ids, cache8.select([row]))[0, -1]
        forced.append(int(lg.argmax()))

        c1 = engine.allocate_cache(max_len, max_slots=1)
        c1.reset_slot(0)
        engine.forward(ids, c1.select([0]))
        singles.append(c1)

    batch_view = cache8.select(list(range(B)))
    worst = 0.0
    flips = 0
    for step in range(n):
        tok = torch.tensor(forced, device="cuda")[:, None]
        lg8 = engine.forward(tok, batch_view)[:, -1]
        lg1 = torch.cat([engine.forward(tok[r:r + 1], singles[r].select([0]))[:, -1]
                         for r in range(B)], dim=0)

        r, _ = _metrics(lg8, lg1)
        worst = max(worst, r)
        flips += int((lg8.argmax(-1) != lg1.argmax(-1)).sum())
        # Teacher forcing: both arms advance on the SAME token, chosen by B=1.
        forced = lg1.argmax(-1).tolist()

    print(f"\n  bf16 teacher-forced, {n} steps x B={B}: worst rel_l2={worst:.3e}, "
          f"argmax flips {flips}/{n * B}")
    assert worst <= BF16_BATCH_DRIFT_MAX, (
        f"batched logits drift {worst:.3e} from B=1 — beyond tile-selection noise")


def test_bf16_free_running_divergence_is_reported(engine, tokens):
    """Not a gate. Records how often greedy decode flips in the deployment dtype.

    Asserting identity here would be asserting that cuBLAS picks the same tiles
    at M=1 and M=8, which is not a property braid controls or should depend on.
    """
    n = 32
    batched = engine.generate_batch(tokens, max_new_tokens=n, temperature=0.0)
    sequential = [engine.generate_batch([p], max_new_tokens=n, temperature=0.0)[0]
                  for p in tokens]
    same = sum(b == s for b, s in zip(batched, sequential))
    print(f"\n  bf16 free-running B=8 vs 8x B=1, {n} tokens: "
          f"{same}/{len(tokens)} rows identical\n{_report(batched, sequential, n)}")
    assert same >= 1, "no row survived — that is a leak, not near-ties"


# --- the indirection ----------------------------------------------------------

def test_non_identity_slot_assignment_changes_nothing(engine, tokens):
    """Sequences must follow their slot, not their row index.

    An engine that reads `pool[row]` instead of `pool[slot_idx[row]]` passes
    every test above, because there the two are equal.
    """
    n, sub = 24, tokens[:4]
    identity = engine.generate_batch(sub, max_new_tokens=n, temperature=0.0)
    shuffled = engine.generate_batch(sub, max_new_tokens=n, temperature=0.0,
                                     slots=[5, 0, 7, 2])
    assert shuffled == identity, "output depends on which pool slot a sequence got"


def test_reused_slot_does_not_leak_the_previous_sequence(engine, tokens):
    """A slot handed to a new sequence must not carry the old one's state.

    This is the failure the reference engine shipped until it added a dedicated
    reset: the leak is *fluent*, because the new sequence is simply conditioned
    on someone else's prompt.
    """
    n = 24
    first = engine.generate_batch([tokens[1]], max_new_tokens=n, temperature=0.0,
                                  slots=[3])
    # Same slot, different prompt, in a fresh run.
    second = engine.generate_batch([tokens[2]], max_new_tokens=n, temperature=0.0,
                                   slots=[3])
    clean = engine.generate_batch([tokens[2]], max_new_tokens=n, temperature=0.0,
                                  slots=[0])
    assert second == clean, "slot 3 leaked the previous occupant's state"
    assert second != first, "two different prompts produced identical output"


def test_rows_are_independent(engine, tokens):
    """Changing row 0's prompt must not move any other row's output."""
    n, sub = 16, tokens[:4]
    base = engine.generate_batch(sub, max_new_tokens=n, temperature=0.0)
    altered = engine.generate_batch([tokens[5]] + sub[1:], max_new_tokens=n,
                                    temperature=0.0)
    assert altered[1:] == base[1:], "row 0's prompt leaked into other rows"


def test_batch_rejects_a_repeated_slot(engine, tokens):
    with pytest.raises(ValueError, match="reuses a slot"):
        engine.generate_batch(tokens[:3], max_new_tokens=4, slots=[1, 1, 2])


def test_kernel_path_still_generates_coherently(engine, tokens):
    """A kernel that is numerically close but wired wrong still produces text."""
    tok = pytest.importorskip("tokenizers")
    tk = tok.Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    assert engine.source is not None, "fixture engine should alias its weights"
    eng = Engine.from_checkpoint(engine.source, device="cuda", dtype=DTYPE,
                                 use_kernels=True)
    out = eng.generate_batch([tokens[0]], max_new_tokens=6, temperature=0.0)[0]
    text = tk.decode(out, skip_special_tokens=False)
    print(f"\n  kernel path: 'The capital of France is' -> {text!r}")
    assert "Paris" in text
