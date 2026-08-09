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

# The ladder stops at 64 and not at 16, and the change was forced by
# measurement. Phase 1 item 4 held that "batch buckets stop at 16 -- c=32 does
# not fit in VRAM and is throughput-pointless", on the reasoning that the linear
# per-sequence state term overtakes the fixed weight sweep at B=14-18. All three
# clauses were wrong: B=64 peaks at 20.6 GB of 32.6 on Qwen3.5-4B, B=16->32 is
# 1.93x and B=32->64 is 1.43x, and the crossover is at B~83. Worse, **B>=32 is
# where braid overtakes llama.cpp at all** -- a ceiling of 16 kept every bucket
# in the range where braid loses.
#
# Buckets are only ever *captured* up to `cache.max_slots`, so a small pool
# still costs nothing; this widens what a large pool is allowed to use.
# Stops at 128, not 64. The decode sweep reaches B=128 on this card once the
# state pool is fp16 (26.0 GiB peak, measured), and braid's margin over
# llama.cpp is still widening there — +84% at B=64, +137% at B=128 — so a served
# ceiling of 64 was leaving the best part of the curve unreachable through the
# scheduler. Callers filter this by their own capacity (`b <= capacity`) — and
# the Scheduler additionally appends an off-rung capacity as its own top rung,
# because a filtered ladder that stops BELOW capacity is a mid-serve ValueError
# the first time every slot decodes in one tick. The extra rungs cost nothing
# to anyone serving fewer rows and no test captures a graph it did not already
# capture.
DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 96, 128)
# `kv_len` is the second capture axis. A graph's shapes are fixed, so the KV
# extent attention reads has to be a constant -- and pinning it to `max_len`
# means every step reads the whole buffer however short the live sequences are:
# measured at 64x the needed traffic for 8-token prompts in a 512 pool. Any
# `kv_len` at or above every live row's length is *correct* (masked keys
# contribute exp(-inf)=0), so these are a bandwidth ladder, not a semantics one.
DEFAULT_KV_BUCKETS = (128, 256, 512)


@dataclass
class _Bucket:
    graph: torch.cuda.CUDAGraph
    tokens: torch.Tensor      # [B, 1] int64 — static input
    slots: torch.Tensor       # [B]    int64 — static input
    slots_i32: torch.Tensor   # [B]    int32 — static input, same values
    logits: torch.Tensor      # [B, vocab] — static output
    size: int
    kv_len: int


class GraphedDecoder:
    """Captures `Engine.decode_step` per batch bucket and replays it."""

    def __init__(self, engine: Engine, cache: Cache,
                 buckets: tuple[int, ...] = DEFAULT_BUCKETS,
                 kv_buckets: tuple[int, ...] | None = None,
                 warmup: int = 3):
        usable = [b for b in sorted(buckets) if b <= cache.max_slots]
        if not usable:
            raise ValueError(
                f"no bucket fits: buckets={buckets}, cache has {cache.max_slots} slots")
        if kv_buckets is None:
            kv_buckets = tuple(k for k in DEFAULT_KV_BUCKETS if k < cache.max_len)
        # `max_len` is always capturable, so the ladder always has a top rung
        # that can serve any live length -- `kv_bucket_for` can never fail.
        self.kv_buckets = tuple(sorted({*(k for k in kv_buckets
                                          if 0 < k <= cache.max_len),
                                        cache.max_len}))
        self.engine = engine
        self.cache = cache
        self.warmup = warmup
        self.buckets: dict[tuple[int, int], _Bucket] = {}
        for b in usable:
            for kv in self.kv_buckets:
                self.buckets[(b, kv)] = self._capture(b, kv)
        self.sizes = tuple(usable)

    # --- capture --------------------------------------------------------------

    def _capture(self, size: int, kv_len: int) -> _Bucket:
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
                eng.decode_step(tokens, view, kv_len=kv_len)
        torch.cuda.current_stream().wait_stream(side)
        cache.restore(saved)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = eng.decode_step(tokens, view, kv_len=kv_len)
        cache.restore(saved)
        return _Bucket(graph=graph, tokens=tokens, slots=slots, slots_i32=slots_i32,
                       logits=logits, size=size, kv_len=kv_len)

    # --- replay ---------------------------------------------------------------

    def bucket_for(self, batch: int) -> int:
        for b in self.sizes:
            if b >= batch:
                return b
        raise ValueError(f"batch {batch} exceeds the largest bucket "
                         f"{max(self.sizes)}")

    def kv_bucket_for(self, live_len: int) -> int:
        """Smallest captured `kv_len` that covers every live row.

        `live_len` is the largest committed length in the batch **plus the token
        about to be written** — a row at length L writes at index L and then
        attends keys 0..L, so it needs kv_len >= L + 1. Undershooting does not
        degrade quality, it silently drops the newest key.
        """
        for k in self.kv_buckets:
            if k >= live_len:
                return k
        raise ValueError(f"live length {live_len} exceeds max_len "
                         f"{max(self.kv_buckets)}")

    def step(self, tokens: torch.Tensor, slots: torch.Tensor,
             pad_slots: torch.Tensor | None = None,
             live_len: int | None = None) -> torch.Tensor:
        """`tokens[B,1]`, `slots[B]` -> logits `[B, vocab]`.

        `pad_slots` supplies scratch slots for the padded rows of the bucket. A
        padded row still advances whatever slot it names, so pointing it at a
        live sequence would corrupt that sequence.

        `live_len` is the largest committed length in the batch **plus one**;
        it selects the KV bucket. Omitting it uses the full `max_len` graph,
        which is always correct and reads the whole buffer. The caller supplies
        it because reading it from the cache here would be a device->host sync,
        and a scheduler knows its own sequence lengths already.
        """
        B = tokens.shape[0]
        size = self.bucket_for(B)
        kv = (self.kv_buckets[-1] if live_len is None
              else self.kv_bucket_for(live_len))
        buck = self.buckets[(size, kv)]

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
