"""FP8 projections under CUDA-graph capture, one group at a time.

**Capture safety is a per-group property and the MLP does not settle it.** Every
fp8 projection adds a device-side amax and a scaled cast to the decode path.
Those allocate, and an allocation whose size depends on a device value — or any
host sync — is a capture failure. `mlp` has been captured since Phase 3; `attn`
and `gdn` put the same pattern on the paths that also carry the KV write and the
recurrent scan, and `head` puts it on the lm_head GEMM.

Bit-identical replay is the gate rather than a tolerance, because a graph that
captured the *wrong* scale would still run and would still produce plausible
text. Equality is the only assertion that can tell the difference.

This lives in its own module rather than beside `test_graph_decode.py` because
it builds a **second** engine, and that module holds a bf16 one plus a
twenty-slot cache and four captured graphs for its whole lifetime. At Qwen3.5-4B
the card absorbs the pair; at Qwen3.5-9B the second engine gets 1.4 GiB and the
test fails for want of memory rather than for want of correctness. A module of
its own gets a process of its own under `scripts/test_isolated.sh`, and the
autouse reclaim in `conftest.py` hands the memory back at the module boundary
even when the whole suite runs in one process.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
from conftest import budgeted_engine

from braid.model.graph import GraphedDecoder
from braid.model.quant import GROUPS

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

DTYPE = torch.bfloat16
MAX_LEN = 96
B = 4


@pytest.mark.parametrize("quant", [*GROUPS, "all"])
def test_every_quant_group_survives_capture_and_replays_identically(quant):
    eng = cache = dec = None
    try:
        # Truncated if the card demands it, and it says so when it does. Capture
        # safety is a per-layer property, so a shorter stack carrying both mixer
        # types exercises it in full.
        eng = budgeted_engine(MODEL_DIR, DTYPE, label=f"capture quant={quant}",
                              use_kernels=True, quant=quant)
        assert eng.quant, f"{quant} resolved to no groups"

        cache = eng.allocate_cache(MAX_LEN, max_slots=B + 1)
        g = torch.Generator(device="cuda").manual_seed(5)
        for slot in range(B + 1):
            cache.reset_slot(slot)
        # Real, differing state per row — zeros would hide a slot-indexing bug.
        for row in range(B):
            eng.forward(torch.randint(0, 1000, (1, 6 + row), generator=g,
                                      device="cuda"), cache.select([row]))

        snapshot = cache.snapshot()
        tokens = torch.arange(100, 100 + B, device="cuda")[:, None]
        slots = torch.arange(B, device="cuda")

        eager = eng.decode_step(tokens, cache.select(slots.tolist())).clone()
        cache.restore(snapshot)
        dec = GraphedDecoder(eng, cache, buckets=(B,))
        cache.restore(snapshot)
        replay = dec.step(tokens, slots)

        assert torch.equal(eager, replay), (
            f"quant={quant}: replay differs from eager by "
            f"{(eager.float() - replay.float()).abs().max():.3e}")
    finally:
        # All of it, not just the engine: a graph holds a private allocator pool
        # that `empty_cache` cannot return while the graph is reachable, and a
        # cache left in a local is ~1 GiB of recurrent state.
        del eng, cache, dec
        gc.collect()
        torch.cuda.empty_cache()
