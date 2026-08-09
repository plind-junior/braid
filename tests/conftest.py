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


def fp32_layer_budget(cfg, reserve_gib: float = 0.0) -> int:
    """How many leading layers of `cfg` fit in fp32 on this card.

    `reserve_gib` is memory the caller has already committed elsewhere and that
    this engine may not use — in practice another module-scoped engine. pytest
    holds a module fixture until the module ends, so a module carrying both a
    bf16 engine and an fp32 one needs them sized *together*, not each against an
    empty card.

    Returns `cfg.num_hidden_layers` when the whole stack fits.
    """
    if not torch.cuda.is_available():
        return cfg.num_hidden_layers
    total = torch.cuda.get_device_properties(0).total_memory
    cap = min(total * FP32_WEIGHT_BUDGET, total * 0.94 - reserve_gib * 2 ** 30)
    for n in range(cfg.num_hidden_layers, 0, -1):
        types = cfg.layer_types[:n]
        # A truncated stack that lost one of the two mixer types would quietly
        # stop testing half the engine.
        if "full_attention" not in types or "linear_attention" not in types:
            break
        if weight_bytes(cfg, n) <= cap:
            return n
    raise RuntimeError(
        f"no fp32 stack with both mixer types fits in "
        f"{cap / 2 ** 30:.1f} GiB for {cfg.num_hidden_layers} layers")


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
    from braid.model.config import ModelConfig
    from braid.model.engine import Engine
    from braid.model.loader import load_checkpoint

    model_dir = Path(model_dir)
    cfg = ModelConfig.from_pretrained(model_dir)
    n = fp32_layer_budget(cfg, reserve_gib)

    cuda_reclaim()
    if n == cfg.num_hidden_layers:
        ck = load_checkpoint(model_dir, device="cuda", dtype=torch.float32)
    else:
        ck = load_checkpoint(model_dir, device="cuda", dtype=torch.float32,
                             layers=tuple(range(n)))
        ck = replace(ck, config=replace(ck.config, num_hidden_layers=n,
                                        layer_types=cfg.layer_types[:n]))
        print(f"\n  [fp32 gate] {model_dir.name}: stack truncated to {n} of "
              f"{cfg.num_hidden_layers} layers "
              f"({weight_bytes(cfg, n) / 2 ** 30:.1f} GiB fp32) — the full "
              f"stack is {weight_bytes(cfg) / 2 ** 30:.1f} GiB")
    return Engine.from_checkpoint(ck, device="cuda", dtype=torch.float32,
                                  **engine_kwargs)


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
