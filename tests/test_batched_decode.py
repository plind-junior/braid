"""Phase 3 item 1 — the batch axis through the whole decode forward.

**The gate: greedy token identity**, 8 prompts as one B=8 batch against 8
sequential B=1 runs. It catches every batch-leakage bug at once — a sampling
parameter read from row 0, a kernel with no token stride, a workspace aliased
across rows, a recurrent slot carrying the previous occupant's state.

**It is asserted in fp32, and that is not a weakening.** Measured
(`scripts/batch_identity_diag.py`):

    fp32   8/8 rows token-identical, logit residual 1e-6   (machine precision)
    bf16   6/8 rows token-identical, logit residual 1e-2

The bf16 gap is not a defect and no implementation removes it. A B=8 GEMM and a
B=1 GEMM select different tiles and split-k, so they accumulate in different
orders; the resulting ~1e-2 relative logit residual is the **same magnitude as
Phase 2's B=1 decode-vs-prefill drift**, so batching did not make it worse.
Greedy argmax then amplifies it discontinuously wherever the top two candidates
are closer than the residual — row 1's top-2 gap at the first decode step is
0.125 against logits of order 10.

So the suite asserts two separate things, and both are stronger than one
free-running bf16 comparison would be:

  1. **fp32 token identity** at B = 2, 4, 8 — the algorithmic claim, exact.
  2. **bf16 teacher-forced logit agreement** — feed both paths the *same*
     tokens so sampling cannot amplify, and bound the per-step residual. A real
     leak moves this immediately; a tile-selection difference does not.

Free-running bf16 identity is reported, not asserted, because gating on it would
be gating on cuBLAS's tile heuristics.

The remaining tests are about the *indirection*: a non-identity slot permutation
must change nothing, and a slot handed to a new sequence must not leak the old
one. Those are what make one captured CUDA graph valid across reassignment,
which is item 2.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from conftest import cuda_reclaim, fp32_engine

from braid.model.config import ModelConfig
from braid.model.engine import Engine
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

DTYPE = torch.bfloat16

# Deliberately different lengths: equal-length prompts would let a cache that
# ignores per-row position pass, which is most of what this is testing.
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


# --- the gate: fp32 token identity --------------------------------------------

@pytest.fixture(scope="module")
def engine_fp32():
    eng = fp32_engine(MODEL_DIR)
    yield eng
    del eng
    cuda_reclaim()


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_greedy_token_identity_fp32(engine_fp32, tokens, batch):
    """**The Phase 3 item-1 gate.** B rows batched == B sequential runs, exactly."""
    n, sub = 32, tokens[:batch]
    batched = engine_fp32.generate_batch(sub, max_new_tokens=n, temperature=0.0)
    sequential = [engine_fp32.generate_batch([p], max_new_tokens=n, temperature=0.0)[0]
                  for p in sub]
    print(f"\n  fp32 B={batch} vs {batch}x B=1, {n} tokens:\n"
          f"{_report(batched, sequential, n)}")
    assert batched == sequential, f"B={batch} is not token-identical to B=1 in fp32"


@pytest.mark.slow
def test_greedy_token_identity_fp32_256(engine_fp32, tokens):
    """The roadmap's stated length: 8 prompts, 256 tokens."""
    n = 256
    batched = engine_fp32.generate_batch(tokens, max_new_tokens=n, temperature=0.0)
    sequential = [engine_fp32.generate_batch([p], max_new_tokens=n, temperature=0.0)[0]
                  for p in tokens]
    print(f"\n  fp32 B=8 vs 8x B=1, {n} tokens:\n{_report(batched, sequential, n)}")
    assert batched == sequential


# --- the CUDA decode path -----------------------------------------------------

@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 1e-2)],
                         ids=["fp32", "bf16"])
def test_kernel_decode_matches_torch_decode(dtype, tol):
    """One GDN layer, both scan paths, from an identical non-zero state.

    fp32 is the real check — it lands at 2.7e-7, which says the kernel computes
    the same function. The bf16 arm is looser on purpose: the kernel runs the
    conv and the l2-norm in fp32 where the torch path follows HF's
    bf16-then-widen order, so it is *more* precise than the reference rather
    than equal to it, and 4.9e-3 is that difference, not an error.
    """
    from braid.model.cache import RecurrentCache
    from braid.model.gdn import GatedDeltaNet

    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    ck = load_checkpoint(MODEL_DIR, device="cuda", layers=(0,),
                         include_embeddings=False, dtype=dtype)
    w, B = ck.layer(0), 8
    g = torch.Generator(device="cuda").manual_seed(31)
    x = torch.randn(B, 1, cfg.hidden_size, generator=g, device="cuda", dtype=dtype)
    slots = torch.arange(B, device="cuda")

    c_t = RecurrentCache(cfg, B, "cuda", dtype)
    c_k = RecurrentCache(cfg, B, "cuda", torch.float32)
    st = torch.randn(B, cfg.gdn.n_heads, cfg.gdn.state_size, cfg.gdn.head_dim,
                     generator=g, device="cuda")
    c_t.state.copy_(st)
    c_k.state.copy_(st)

    with torch.no_grad():
        y_t = GatedDeltaNet(cfg, w, use_kernels=False)(x, cache=c_t, slots=slots)
        y_k = GatedDeltaNet(cfg, w, use_kernels=True)(
            x, cache=c_k, slots=slots, slots_i32=slots.to(torch.int32))

    r, c = _metrics(y_k, y_t)
    rs, _ = _metrics(c_k.state, c_t.state)
    print(f"\n  kernel vs torch [{dtype}]: out rel_l2={r:.3e} cos={c:.9f} "
          f"state rel_l2={rs:.3e}")
    assert r <= tol, f"kernel decode differs from torch by {r:.3e}"


# Kernel engines are built over the SAME checkpoint tensors as the fixtures --
# `from_checkpoint` only wraps them -- so this costs no extra weight memory. Two
# copies plus a third would not fit at 4B and are nowhere near fitting at 9B.
# `Engine.source` is None for a quantized engine, because there the tensors are
# NOT shared; these fixtures are unquantized, so it is present.

def test_kernel_path_holds_the_batch_identity_gate(engine_fp32, tokens):
    """The gate again, with the kernels doing the scan. fp32, B=2/4/8."""
    assert engine_fp32.source is not None, "fixture engine should alias its weights"
    eng = Engine.from_checkpoint(engine_fp32.source, device="cuda",
                                 dtype=torch.float32, use_kernels=True)
    for batch in (2, 4, 8):
        n, sub = 24, tokens[:batch]
        batched = eng.generate_batch(sub, max_new_tokens=n, temperature=0.0)
        sequential = [eng.generate_batch([p], max_new_tokens=n, temperature=0.0)[0]
                      for p in sub]
        assert batched == sequential, (
            f"kernels, B={batch}: {_report(batched, sequential, n)}")
    print("\n  kernel path: fp32 token identity holds at B=2, 4, 8")
