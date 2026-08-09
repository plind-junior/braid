"""Shared fixtures and the one piece of resource hygiene the suite needs.

Several modules hold a **module-scoped** engine — `test_scheduler` keeps an fp32
one (16.8 GB) for its identity gates and a bf16 one beside it, `test_server` and
`test_graph_decode` keep bf16 engines. pytest holds a module fixture until that
module finishes, so by the time a later module builds its engine the allocator
is sitting on many GB of *reserved but unallocated* blocks from earlier ones.

That is not a leak and `torch.cuda.memory_allocated` looks fine; it is caching.
It still OOMs the next big load, which is what happened at 155 passed / 2 errors
with 14.71 GB reserved and 237 MiB free. Returning the cached blocks between
modules is enough — no fixture has to be restructured, and none of them are
wrong.
"""
from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path

import pytest
import torch

# Fraction of the card an fp32 engine's *weights* may occupy. The rest is caches,
# activations, graph pools and the allocator's cached blocks. 0.55 is chosen so
# Qwen3.5-4B (16.8 GiB fp32, 0.54) keeps its full 32-layer stack and Qwen3.5-9B
# (35.8 GiB, 1.14) does not — see `fp32_engine`.
FP32_WEIGHT_BUDGET = 0.55


def weight_bytes(cfg, n_layers: int | None = None, itemsize: int = 4) -> int:
    """Weight bytes for the embeddings plus the first `n_layers` layers.

    Derived from the config rather than measured, because the point is to decide
    how much to load *before* loading it.
    """
    if n_layers is None:
        n_layers = cfg.num_hidden_layers
    h, g = cfg.hidden_size, cfg.gdn
    emb = cfg.vocab_size * h * (1 if cfg.tie_word_embeddings else 2)
    mlp = 3 * h * cfg.intermediate_size
    attn = cfg.q_proj_out * h + 2 * cfg.kv_dim * h + h * cfg.q_dim + 2 * cfg.head_dim
    gdn = (g.conv_channels * h + g.inner_size * h + 2 * g.n_heads * h
           + h * g.inner_size + g.conv_channels * g.conv_kernel + 3 * g.n_heads)
    n_gdn = sum(1 for i in range(n_layers) if cfg.is_gdn(i))
    params = (emb + h + n_gdn * (gdn + mlp) + (n_layers - n_gdn) * (attn + mlp)
              + n_layers * 2 * h)
    return params * itemsize


def layer_budget(cfg, itemsize: int = 4, reserve_gib: float = 0.0) -> int:
    """How many leading layers of `cfg` fit on this card at `itemsize` bytes.

    `reserve_gib` is memory the caller has already committed elsewhere and that
    this engine may not use — in practice another module-scoped engine. pytest
    holds a module fixture until the module ends, so a module carrying both a
    bf16 engine and a second one needs them sized *together*, not each against
    an empty card.

    Returns `cfg.num_hidden_layers` when the whole stack fits.
    """
    if not torch.cuda.is_available():
        return cfg.num_hidden_layers
    cuda_reclaim()
    # Against **free** memory, not total. A second engine is built while the
    # first is still resident *and holding its caches and graph pool* — in
    # `test_graph_decode` that is 16.7 GiB of weights plus ~1.2 of pools, and a
    # budget computed from the card's total size cheerfully hands out memory
    # that is already spoken for. `mem_get_info` after a reclaim is the only
    # number that knows about all of it.
    free, total = torch.cuda.mem_get_info()
    # 0.70 leaves room for what this engine allocates *after* its weights: the
    # cache pool, the graph's private pool, and decode activations.
    cap = min(total * FP32_WEIGHT_BUDGET, free * 0.70 - reserve_gib * 2 ** 30)
    for n in range(cfg.num_hidden_layers, 0, -1):
        types = cfg.layer_types[:n]
        # A truncated stack that lost one of the two mixer types would quietly
        # stop testing half the engine.
        if "full_attention" not in types or "linear_attention" not in types:
            break
        if weight_bytes(cfg, n, itemsize) <= cap:
            return n
    raise RuntimeError(
        f"no {itemsize}-byte stack with both mixer types fits in "
        f"{cap / 2 ** 30:.1f} GiB for {cfg.num_hidden_layers} layers")


def fp32_layer_budget(cfg, reserve_gib: float = 0.0) -> int:
    return layer_budget(cfg, 4, reserve_gib)


def budgeted_checkpoint(model_dir, dtype, reserve_gib: float = 0.0, label: str = ""):
    """A checkpoint truncated to as many leading layers as the card allows.

    Says so on stdout when it truncates, because a gate that quietly tests less
    of the model than its name claims is worse than one that fails.
    """
    from braid.model.config import ModelConfig
    from braid.model.loader import load_checkpoint

    model_dir = Path(model_dir)
    cfg = ModelConfig.from_pretrained(model_dir)
    n = layer_budget(cfg, dtype.itemsize, reserve_gib)

    cuda_reclaim()
    if n == cfg.num_hidden_layers:
        return load_checkpoint(model_dir, device="cuda", dtype=dtype)
    ck = load_checkpoint(model_dir, device="cuda", dtype=dtype,
                         layers=tuple(range(n)))
    print(f"\n  [{label or 'budget'}] {model_dir.name}: stack truncated to {n} of "
          f"{cfg.num_hidden_layers} layers "
          f"({weight_bytes(cfg, n, dtype.itemsize) / 2 ** 30:.1f} GiB) — the full "
          f"stack is {weight_bytes(cfg, None, dtype.itemsize) / 2 ** 30:.1f} GiB")
    return replace(ck, config=replace(ck.config, num_hidden_layers=n,
                                      layer_types=cfg.layer_types[:n]))


def fp32_engine(model_dir, reserve_gib: float = 0.0, **engine_kwargs):
    """An fp32 engine, truncated to as many leading layers as the card allows.

    **The fp32 identity gates compare braid against braid**, not against HF:
    batch leakage, padding inertness, chunk equivalence, concurrent-vs-solo.
    Every one of them asserts a property of the cache and scan *plumbing*, which
    a stack carrying both mixer types exercises in full — depth adds accumulated
    rounding, not new code paths. So where the whole stack does not fit, the
    honest move is to shorten it and say so, not to drop the gate or move it to
    bf16. **bf16 cannot carry these gates at all**: a B=1 and a B=8 GEMM select
    different tiles and accumulate in different orders, which moves logits ~1e-2
    and flips greedy argmax on near-ties (ROADMAP Phase 3).

    Qwen3.5-4B keeps all 32 layers (16.8 GiB). Qwen3.5-9B is 35.8 GiB against a
    31.4 GiB card and truncates. `Engine` reads `num_hidden_layers` and
    `layer_types`, so a truncated config *is* a shorter model — nothing else has
    to know.
    """
    from braid.model.engine import Engine

    ck = budgeted_checkpoint(model_dir, torch.float32, reserve_gib, "fp32 gate")
    return Engine.from_checkpoint(ck, device="cuda", dtype=torch.float32,
                                  **engine_kwargs)


def budgeted_engine(model_dir, dtype=torch.bfloat16, reserve_gib: float = 0.0,
                    label: str = "", **engine_kwargs):
    """An engine at `dtype`, truncated to what fits beside `reserve_gib`.

    For gates that must build a **second** engine while a module-scoped one is
    still alive. At Qwen3.5-4B two bf16 engines are 7.8 GiB each and the card
    absorbs it; at Qwen3.5-9B they are 16.7 each against 31.4 and it does not.
    """
    from braid.model.engine import Engine

    ck = budgeted_checkpoint(model_dir, dtype, reserve_gib, label)
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=dtype, **engine_kwargs)
    del ck
    cuda_reclaim()
    return eng


def cuda_reclaim() -> None:
    """Return cached blocks to the driver. Call before a large allocation.

    `torch.cuda.empty_cache()` at a module boundary is not enough when one
    module holds two engines: the churn that fills the cache happens *between*
    them, so the reclaim has to sit at the second load, not at the edges.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="module", autouse=True)
def _reclaim_between_modules():
    """Hand cached blocks back to the driver before and after each module."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def assert_greedy(mine, ref, what):
    """Greedy tokens must agree **wherever the arithmetic can resolve the choice**.

    A flat `torch.equal(argmax)` is not a well-posed gate on a low-precision arm,
    and this suite asserted it in two places before anyone checked. Both were
    measured, and both were measuring reduction order rather than the model:

      * `test_decode_matches_prefill` (Qwen3.5-9B, bf16): at position 1 the top
        two logits are tokens 847 and 11 and they are **the same bf16 number** —
        top-2 gap exactly 0.00000. Reading the same hidden states through an fp32
        head resolves the gap to 0.019 and both arms then agree everywhere.
      * `test_graphed_scheduler_matches_the_eager_one` (same model): the eager
        and graphed schedulers run the partial batch at B=3 and B=4, and the
        measured logit residual between them is 0.06-0.19 while the top-2 gap
        goes as low as 0.0625. It flipped when a *more accurate* prefill landed.

    The rule this encodes: a difference is a defect only where the two arms could
    tell the two tokens apart. `gap > residual` means resolvable, so a difference
    there is real. `gap <= residual` means the arithmetic never distinguished
    them, and the position is reported rather than asserted on.

    This keeps everything the flat assertion caught — a miswired layer moves
    logits by ~1e-1, hundreds of times the residual, and lands on positions with
    ordinary gaps — and drops only the part that was measuring tie-breaks.

    `mine` and `ref` are `[..., vocab]`; leading dims are flattened, so this
    takes `[T, V]` prefill logits and `[B, V]` decode logits alike. Returns the
    number of unresolvable positions, so a caller can assert the gate is not
    vacuous if it wants to.
    """
    m = mine.reshape(-1, mine.shape[-1]).float()
    r = ref.reshape(-1, ref.shape[-1]).float()
    top2 = r.topk(2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    resid = (m - r).abs().amax(dim=-1)
    differ = m.argmax(-1) != r.argmax(-1)
    hard = differ & (gap > resid)
    soft = differ & ~hard

    if bool(soft.any()):
        p = soft.nonzero().flatten().tolist()
        print(f"\n  {what}: {len(p)} unresolvable position(s) {p} — top-2 gap "
              f"{[round(g, 5) for g in gap[p].tolist()]} within residual "
              f"{[round(x, 5) for x in resid[p].tolist()]}")
    assert not bool(hard.any()), (
        f"{what}: greedy tokens differ at resolvable position(s) "
        f"{hard.nonzero().flatten().tolist()} — gap "
        f"{gap[hard].tolist()} exceeds residual {resid[hard].tolist()}")
    return int(soft.sum())
