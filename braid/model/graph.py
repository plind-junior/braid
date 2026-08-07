"""CUDA-graph capture of the decode step. ROADMAP Phase 3 item 2.

One graph per batch bucket {1, 2, 4, 8, 16}; a batch of 5 replays the 8-bucket
with the unused rows padded. Every graph is captured against **static input
buffers** — token ids and the slot assignment — which the caller copies into
before replay. The cache pools are captured by address and never reallocated.

**The slot buffer is the whole point.** Because `slot_idx` is read from device
memory inside the kernels, changing which sequences occupy which slots is a
`copy_` into that buffer, not a re-capture. The reference engine re-captures on
every reassignment at a documented 10–20 ms
(`engine_scheduler.cpp:1968-1981`); braid already measured the replay path at
10.3 µs (`tests/test_slot_indirection.py`). This class is where that becomes
true of the whole model rather than of one kernel.

**Padding is not free of meaning.** Padded rows are real sequences as far as the
arithmetic is concerned — they occupy slots, advance those slots' lengths, and
write KV. They must therefore point at *scratch* slots that no live sequence
owns, or a padded row will corrupt a real one's state.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from braid.model.cache import Cache
from braid.model.engine import Engine

DEFAULT_BUCKETS = (1, 2, 4, 8, 16)


@dataclass
class _Bucket:
    graph: torch.cuda.CUDAGraph
    tokens: torch.Tensor      # [B, 1] int64 — static input
    slots: torch.Tensor       # [B]    int64 — static input
    slots_i32: torch.Tensor   # [B]    int32 — static input, same values
    logits: torch.Tensor      # [B, vocab] — static output
    size: int


class GraphedDecoder:
    """Captures `Engine.decode_step` per batch bucket and replays it."""

    def __init__(self, engine: Engine, cache: Cache,
                 buckets: tuple[int, ...] = DEFAULT_BUCKETS,
                 warmup: int = 3):
        usable = [b for b in sorted(buckets) if b <= cache.max_slots]
        if not usable:
            raise ValueError(
                f"no bucket fits: buckets={buckets}, cache has {cache.max_slots} slots")
        self.engine = engine
        self.cache = cache
        self.warmup = warmup
        self.buckets: dict[int, _Bucket] = {}
        for b in usable:
            self.buckets[b] = self._capture(b)

    # --- capture --------------------------------------------------------------

    def _capture(self, size: int) -> _Bucket:
        eng, cache = self.engine, self.cache
        dev = eng.device

        tokens = torch.zeros(size, 1, dtype=torch.long, device=dev)
        slots = torch.arange(size, dtype=torch.long, device=dev)
        slots_i32 = slots.to(torch.int32)
        view = Cache(layers=cache.layers, slots=slots, lengths=cache.lengths,
                     max_slots=cache.max_slots, max_len=cache.max_len,
                     slots_i32=slots_i32)

        # Warm up on a side stream first. Capture records whatever the ops do,
        # and cuBLAS/cuDNN pick algorithms and allocate workspaces on first call
        # — doing that inside the capture bakes in a one-off and can fail
        # outright.
        #
        # Warmup and capture are real decode steps: they advance state, conv, KV
        # and lengths. The whole cache is snapshotted and restored around them,
        # so capturing leaves no mark on the sequences already in the pool.
        saved = cache.snapshot()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                eng.decode_step(tokens, view)
        torch.cuda.current_stream().wait_stream(side)
        cache.restore(saved)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = eng.decode_step(tokens, view)
        cache.restore(saved)
        return _Bucket(graph=graph, tokens=tokens, slots=slots, slots_i32=slots_i32,
                       logits=logits, size=size)

    # --- replay ---------------------------------------------------------------

    def bucket_for(self, batch: int) -> int:
        for b in sorted(self.buckets):
            if b >= batch:
                return b
        raise ValueError(f"batch {batch} exceeds the largest bucket "
                         f"{max(self.buckets)}")

    def step(self, tokens: torch.Tensor, slots: torch.Tensor,
             pad_slots: torch.Tensor | None = None) -> torch.Tensor:
        """`tokens[B,1]`, `slots[B]` -> logits `[B, vocab]`.

        `pad_slots` supplies scratch slots for the padded rows of the bucket. A
        padded row still advances whatever slot it names, so pointing it at a
        live sequence would corrupt that sequence.
        """
        B = tokens.shape[0]
        size = self.bucket_for(B)
        buck = self.buckets[size]

        buck.tokens[:B].copy_(tokens)
        buck.slots[:B].copy_(slots)
        if B < size:
            if pad_slots is None:
                raise ValueError(
                    f"batch {B} pads bucket {size}; pass pad_slots for the "
                    f"{size - B} scratch rows, or the padding will advance live slots")
            buck.slots[B:].copy_(pad_slots[: size - B])
            buck.tokens[B:].zero_()
        buck.slots_i32.copy_(buck.slots)

        buck.graph.replay()
        return buck.logits[:B]
