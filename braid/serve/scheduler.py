"""Continuous batching over the slot pool. ROADMAP Phase 4 item 1.

**Admission-only capacity control, no preemption.** A request is admitted when a
slot is free and then runs to completion; nothing is evicted mid-generation.
That is a deliberate simplification and it is the reason `Scheduler` can be this
small — preemption would need the recurrent state swapped out, and unlike KV a
GDN state cannot be recomputed from the tokens, only replayed.

Three things this file exists to get right, each of which is a bug the roadmap
names by hand:

1. **Release on cancel must release *both* pools.** A client disconnecting
   mid-generation frees its KV extent and its recurrent slot. The reference
   engine leaked slots on exactly this path until it added a dedicated reset,
   and the failure is silent: the pool just fills up.

2. **Sampling parameters are per row.** `Engine._sample` takes one temperature
   for the whole batch, which is correct for `generate_batch` and wrong here —
   concurrent requests do not share a temperature. A sampler that reads its
   parameters, or its RNG draw, from row 0 is invisible at concurrency 1 and
   silently wrong at 8.

3. **A recycled slot must not carry its previous occupant's state.** The
   recurrent state is zeroed on admission. The KV pool deliberately is *not*:
   stale keys past the new sequence's length are masked to `exp(-inf) = 0`, so
   zeroing 16 MiB per admission would buy nothing. That is a claim about the
   mask, so `tests/test_scheduler.py` tests it rather than trusting it.

**Chunked prefill is what keeps admission from stalling decode.** A 2,000-token
prompt admitted in one forward is ~2,000 tokens of work while every live stream
waits. Each tick spends at most `prefill_chunk` tokens on one sequence, so a
long prompt costs the pool a bounded slice per step instead of all of it. That
path was measurably broken until 2026-08-08 (`docs/runbooks/chunked-prefill-kv.md`);
it is load-bearing here.
"""
from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from braid.model.cache import Cache
from braid.model.engine import Engine
from braid.model.graph import GraphedDecoder

_ids = itertools.count(1)


@dataclass
class Request:
    """One generation request. `temperature <= 0` means greedy."""

    prompt: list[int]
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    eos_token_id: int | None = None
    id: str = field(default_factory=lambda: f"r{next(_ids)}")

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError(f"request {self.id} has an empty prompt")
        if self.max_new_tokens < 1:
            raise ValueError(f"request {self.id}: max_new_tokens must be >= 1")


@dataclass
class Update:
    """What happened to one request during one `step()`."""

    id: str
    tokens: list[int] = field(default_factory=list)
    finished: bool = False
    reason: str | None = None          # "eos" | "length" | "cancelled"


@dataclass
class _Live:
    req: Request
    slot: int
    fed: int = 0                       # prompt tokens committed so far
    out: list[int] = field(default_factory=list)
    last: int | None = None            # token to feed at the next decode
    gen: torch.Generator | None = None

    @property
    def prefilling(self) -> bool:
        return self.fed < len(self.req.prompt)

    @property
    def length(self) -> int:
        return self.fed + len(self.out)


class Scheduler:
    """Continuous batching. Drive it by calling `step()` until `idle`.

    `capacity` is how many requests may be resident. The pool is allocated with
    one extra slot: `GraphedDecoder` pads a partial batch up to its bucket, and
    a padded row still advances whatever slot it names, so the padding needs a
    scratch slot that no request can ever be admitted into.
    """

    def __init__(self, engine: Engine, capacity: int = 8, max_len: int = 2048,
                 graphed: bool = True, buckets: tuple[int, ...] | None = None,
                 prefill_chunk: int = 256):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.engine = engine
        self.capacity = capacity
        self.max_len = max_len
        self.prefill_chunk = prefill_chunk
        self.cache = engine.allocate_cache(max_len, max_slots=capacity + 1)
        self.scratch = capacity                      # never admitted into
        for s in range(capacity + 1):
            self.cache.reset_slot(s)

        # Observed concurrency, so a test can prove the identity gate is not
        # vacuous: if only one row ever decodes at a time, "concurrent streams
        # match solo runs" is a tautology.
        self.max_decode_batch = 0
        self.decode_ticks = 0
        # Prefill runs the decode recurrence in a Python loop over T
        # (`gdn.py`: "slow and deliberate", Phase 5 replaces it), so at realistic
        # prompt lengths it can dominate a serving run outright. Accounted here
        # rather than inferred, because "aggregate tok/s is low" does not say
        # which half is slow.
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.prefill_tokens = 0

        self._free: deque[int] = deque(range(capacity))
        self._queue: deque[Request] = deque()
        self._live: dict[str, _Live] = {}
        self._scratch_t = torch.tensor([self.scratch], device=engine.device)

        self.decoder = None
        if graphed:
            usable = buckets or tuple(b for b in (1, 2, 4, 8, 16) if b <= capacity)
            self.decoder = GraphedDecoder(engine, self.cache, buckets=usable)
            for s in range(capacity + 1):
                self.cache.reset_slot(s)

    def reset_stats(self) -> None:
        """Zero the counters. A benchmark that warms up on the same scheduler
        must call this, or prefill time from the discarded run is charged to the
        timed one -- which reads as a prefill share above 100%."""
        self.max_decode_batch = 0
        self.decode_ticks = 0
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.prefill_tokens = 0

    # --- queue ----------------------------------------------------------------

    @property
    def idle(self) -> bool:
        return not self._queue and not self._live

    @property
    def live_ids(self) -> list[str]:
        return list(self._live)

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def submit(self, req: Request) -> str:
        if len(req.prompt) + req.max_new_tokens > self.max_len:
            raise ValueError(
                f"request {req.id} needs {len(req.prompt) + req.max_new_tokens} "
                f"tokens of KV; the pool holds {self.max_len}")
        self._queue.append(req)
        return req.id

    def cancel(self, req_id: str) -> bool:
        """Release a request's slot. Both pools, or the pool leaks.

        Returns False for an id that is neither queued nor live, which is the
        normal race when a client disconnects as its last token is emitted.
        """
        live = self._live.pop(req_id, None)
        if live is not None:
            self._release(live)
            return True
        for i, q in enumerate(self._queue):
            if q.id == req_id:
                del self._queue[i]
                return True
        return False

    def _release(self, live: _Live) -> None:
        # `reset_slot` zeroes the recurrent state, the conv window and the
        # length. It deliberately does not zero KV -- see the module docstring.
        self.cache.reset_slot(live.slot)
        self._free.append(live.slot)

    # --- one tick -------------------------------------------------------------

    @torch.no_grad()
    def step(self) -> list[Update]:
        """Admit, prefill a bounded slice, then decode every ready row."""
        updates: dict[str, Update] = {}

        while self._free and self._queue:
            req = self._queue.popleft()
            slot = self._free.popleft()
            self.cache.reset_slot(slot)
            gen = None
            if req.temperature > 0.0 and req.seed is not None:
                gen = torch.Generator(device=self.engine.device)
                gen.manual_seed(req.seed)
            self._live[req.id] = _Live(req=req, slot=slot, gen=gen)

        # One sequence per forward, a bounded number of tokens per tick.
        for live in list(self._live.values()):
            if live.prefilling:
                t0 = time.perf_counter()
                n = live.fed
                self._prefill_chunk(live, updates)
                self.prefill_s += time.perf_counter() - t0
                self.prefill_tokens += live.fed - n
                break

        ready = [l for l in self._live.values() if not l.prefilling and l.last is not None]
        if ready:
            self.max_decode_batch = max(self.max_decode_batch, len(ready))
            self.decode_ticks += 1
            t0 = time.perf_counter()
            self._decode(ready, updates)
            self.decode_s += time.perf_counter() - t0
        return list(updates.values())

    def _prefill_chunk(self, live: _Live, updates: dict[str, Update]) -> None:
        prompt = live.req.prompt
        hi = min(live.fed + self.prefill_chunk, len(prompt))
        ids = torch.tensor([prompt[live.fed:hi]], dtype=torch.long,
                           device=self.engine.device)
        logits = self.engine.forward(ids, self.cache.select([live.slot]))
        live.fed = hi
        if not live.prefilling:
            # The last chunk's final logits are the first generated token.
            tok = self._sample_one(logits[0, -1], live)
            self._emit(live, tok, updates)

    def _decode(self, ready: list[_Live], updates: dict[str, Update]) -> None:
        eng = self.engine
        slots = torch.tensor([l.slot for l in ready], device=eng.device)
        tokens = torch.tensor([[l.last] for l in ready], dtype=torch.long,
                              device=eng.device)
        live_len = max(l.length for l in ready) + 1

        if self.decoder is not None:
            logits = self.decoder.step(tokens, slots, pad_slots=self._scratch_t,
                                       live_len=live_len)
        else:
            view = self.cache.select(slots)
            logits = eng.decode_step(tokens, view, kv_len=self._kv_len(live_len))

        for i, tok in enumerate(self._sample_rows(logits, ready)):
            self._emit(ready[i], tok, updates)

    def _kv_len(self, live_len: int) -> int:
        if self.decoder is not None:
            return self.decoder.kv_bucket_for(live_len)
        return min(self.max_len, max(1, live_len))

    # --- sampling -------------------------------------------------------------

    def _sample_rows(self, logits: torch.Tensor, ready: list[_Live]) -> list[int]:
        """`[B, vocab]` -> one token per row, each with **its own** parameters.

        Greedy rows go through one vectorized `argmax`; sampled rows are drawn
        individually because `torch.multinomial` takes a single generator and a
        per-request seed is the only way a stream is reproducible. B is at most
        the batch bucket, and only non-greedy rows pay the loop.
        """
        out = logits.argmax(-1).tolist()
        for i, live in enumerate(ready):
            if live.req.temperature > 0.0:
                out[i] = self._sample_one(logits[i], live)
        return out

    def _sample_one(self, row: torch.Tensor, live: _Live) -> int:
        req = live.req
        if req.temperature <= 0.0:
            return int(row.argmax(-1))
        probs = torch.softmax(row.float() / req.temperature, dim=-1)
        if req.top_p < 1.0:
            srt, idx = probs.sort(descending=True)
            keep = (srt.cumsum(0) - srt) < req.top_p
            srt = torch.where(keep, srt, torch.zeros_like(srt))
            probs = torch.zeros_like(probs).scatter_(0, idx, srt)
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1, generator=live.gen)[0])

    # --- completion -----------------------------------------------------------

    def _emit(self, live: _Live, tok: int, updates: dict[str, Update]) -> None:
        up = updates.setdefault(live.req.id, Update(id=live.req.id))
        req = live.req

        if tok == req.eos_token_id:
            up.finished, up.reason = True, "eos"
            self._finish(live)
            return

        live.out.append(tok)
        live.last = tok
        up.tokens.append(tok)

        if len(live.out) >= req.max_new_tokens:
            up.finished, up.reason = True, "length"
            self._finish(live)
        elif live.length + 1 > self.max_len:
            # Admission checks this, but a caller can hand-build a request that
            # slips through; running past the pool would corrupt the next slot.
            up.finished, up.reason = True, "length"
            self._finish(live)

    def _finish(self, live: _Live) -> None:
        self._live.pop(live.req.id, None)
        self._release(live)

    # --- convenience ----------------------------------------------------------

    def run(self, requests: list[Request], max_ticks: int = 100_000
            ) -> dict[str, list[int]]:
        """Submit everything and drive to completion. For tests and benches."""
        out: dict[str, list[int]] = {r.id: [] for r in requests}
        for r in requests:
            self.submit(r)
        for _ in range(max_ticks):
            if self.idle:
                return out
            for up in self.step():
                out[up.id].extend(up.tokens)
        raise RuntimeError(f"scheduler did not drain in {max_ticks} ticks")
