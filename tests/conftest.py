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

import pytest
import torch


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
