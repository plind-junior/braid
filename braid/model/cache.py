"""Per-sequence state, addressed by **slot** rather than by batch row.

Phase 3 item 1. The change from Phase 2 is the indirection: state lives in a
pool of `max_slots` entries and a device-resident `slot_idx[batch]` says which
entry each row of the current batch owns.

**Why the indirection is the point.** Baking a base pointer per sequence means a
captured CUDA graph is only valid for one assignment of sequences to slots, so
every admission or eviction forces a re-capture — the reference engine documents
that at 10–20 ms (`engine_scheduler.cpp:1968-1981`). Reading `slot_idx` from
device memory inside the kernel makes one captured graph valid for **every**
assignment, forever; braid already measured the replay at 10.3 µs across three
different slot assignments (`tests/test_slot_indirection.py`). Graph capture is
item 2, but the layout that permits it has to be here.

The recurrent pool's shape is the one the Phase 1 CUDA kernels already take:
`[max_slots, n_heads, state_size, head_dim]` fp32, `head_dim` fastest-varying,
with `conv` as `[max_slots, conv_channels, conv_kernel]`.

**KV is pooled but not paged.** Each slot owns a contiguous `max_len` run. That
wastes capacity for short sequences and is exactly what Phase 3 item 3's block
manager replaces; it is not a layout braid intends to serve on.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from braid.model.config import ModelConfig


class KVCache:
    """One attention layer's K/V pool, `[max_slots, kv_heads, max_len + 1, head_dim]`.

    **The `+ 1` is a sink.** Ragged batched prefill hands every layer a padded
    `[B, T]` rectangle in which row b carries only `seq_lens[b] <= T` real
    tokens, and the padded columns have to be written *somewhere*: routing them
    to a reserved position that `read` never returns is branch-free and costs
    one token-slot per row. Clamping them onto a live position instead would
    stamp garbage over real keys, and dropping them would need a data-dependent
    shape.
    """

    def __init__(self, cfg: ModelConfig, max_slots: int, max_len: int,
                 device: str | torch.device, dtype: torch.dtype):
        shape = (max_slots, cfg.num_key_value_heads, max_len + 1, cfg.head_dim)
        # Zeroed, not empty: masked-out positions are still *read* by SDPA, and
        # uninitialised memory can hold inf/NaN that survives a -inf mask as NaN.
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.sink = max_len

    def write(self, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor,
              positions: torch.Tensor, seq_lens: torch.Tensor | None = None) -> None:
        """`k, v` are `[B, kv_heads, T, head_dim]`; `positions[b]` is row b's start.

        `seq_lens[b]` is how many of the T columns row b actually owns; `None`
        means all T, which is the rectangular case. Columns past a row's own
        length go to the sink.

        Callers are responsible for the overflow check — `Engine.hidden_states`
        already reads `positions.max()` on the host and raises there, so
        repeating it here would buy a second sync per attention layer and no
        extra safety.
        """
        B, _, T, _ = k.shape
        if T == 1:
            # Scatter: both index tensors are [B], so this picks one (slot, pos)
            # pair per row rather than a cross product.
            self.k[slots, :, positions] = k[:, :, 0]
            self.v[slots, :, positions] = v[:, :, 0]
            return

        col = torch.arange(T, device=k.device)
        pos = positions[:, None] + col                          # [B, T]
        if seq_lens is not None:
            pos = torch.where(col[None] < seq_lens[:, None], pos, self.sink)
        rows = slots[:, None].expand(B, T)
        # Two advanced indices with a slice between them: the indexed axes move
        # to the front, so the right-hand side is [B, T, kv_heads, head_dim].
        self.k[rows, :, pos] = k.transpose(1, 2)
        self.v[rows, :, pos] = v.transpose(1, 2)

    def read(self, slots: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`[B, kv_heads, length, head_dim]` for the rows named by `slots`.

        Slicing before gathering matters: `index_select` on the full pool would
        copy `max_len` per row regardless of how much of it is live. `length` is
        bounded by `max_len`, so the slice can never reach the sink.
        """
        return (self.k[:, :, :length].index_select(0, slots),
                self.v[:, :, :length].index_select(0, slots))


class RecurrentCache:
    """One GDN layer's conv window and delta-rule state, pooled by slot.

    `conv` holds the last `conv_kernel` **pre-convolution** inputs, matching HF's
    `causal_conv1d_update` convention. Holding post-conv outputs instead would
    decode fluently and wrongly.
    """

    def __init__(self, cfg: ModelConfig, max_slots: int,
                 device: str | torch.device, dtype: torch.dtype,
                 state_dtype: torch.dtype = torch.float32):
        g = cfg.gdn
        # The CUDA conv kernel requires fp32; the torch path uses the activation
        # dtype so its numerics stay exactly what Phase 2 measured.
        self.conv = torch.zeros(max_slots, g.conv_channels, g.conv_kernel,
                                device=device, dtype=dtype)
        # **The state is the largest per-sequence term and it is half the decode
        # step's traffic at B=64.** 48 MiB per sequence on Qwen3.5-9B (24 GDN
        # layers x 32 heads x 128 x 128 x 4 B), read and written every step: at
        # B=64 that is 6.14 GiB against 7.40 GiB of fp8 weights. Storing it in
        # 16 bits halves the term that caps the curve.
        #
        # fp32 remains the default because it is what `mamba_ssm_dtype` is and
        # what every parity gate is calibrated against. **Only the storage
        # narrows** — the recurrence is computed in fp32 in both the torch path
        # and the kernel, so what a 16-bit state costs is one rounding per step,
        # not a lower-precision scan.
        if state_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"state_dtype {state_dtype} is not one of "
                             "fp32 / fp16 / bf16")
        self.state = torch.zeros(max_slots, g.n_heads, g.state_size, g.head_dim,
                                 device=device, dtype=state_dtype)

    def reset(self, slot: int) -> None:
        """Zero one slot. A slot handed to a new sequence carrying the previous
        occupant's state produces fluent output conditioned on someone else's
        prompt — the failure mode a leak test has to catch."""
        self.conv[slot].zero_()
        self.state[slot].zero_()


@dataclass
class Cache:
    """Whole-model state plus the batch's slot assignment and per-row lengths."""

    layers: list[KVCache | RecurrentCache]
    slots: torch.Tensor      # int64 [B] — pool entry owned by each batch row
    lengths: torch.Tensor    # int64 [max_slots] — tokens committed, indexed by SLOT
    max_slots: int
    max_len: int
    # The CUDA kernels take `slot_idx` as int32. Held alongside rather than cast
    # per call: a cast allocates, and an allocation inside a captured graph is a
    # capture failure.
    slots_i32: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.slots_i32 is None:
            self.slots_i32 = self.slots.to(torch.int32)

    @classmethod
    def allocate(cls, cfg: ModelConfig, max_slots: int, max_len: int,
                 device: str | torch.device, dtype: torch.dtype,
                 conv_dtype: torch.dtype | None = None,
                 state_dtype: torch.dtype = torch.float32) -> Cache:
        return cls(
            layers=[
                RecurrentCache(cfg, max_slots, device, conv_dtype or dtype,
                               state_dtype)
                if cfg.is_gdn(i)
                else KVCache(cfg, max_slots, max_len, device, dtype)
                for i in range(cfg.num_hidden_layers)
            ],
            slots=torch.arange(max_slots, device=device),
            lengths=torch.zeros(max_slots, dtype=torch.int64, device=device),
            max_slots=max_slots,
            max_len=max_len,
        )

    def reset_slot(self, slot: int) -> None:
        for lyr in self.layers:
            if isinstance(lyr, RecurrentCache):
                lyr.reset(slot)
        self.lengths[slot] = 0

    def select(self, slots: list[int] | torch.Tensor) -> Cache:
        """A view of this cache addressing `slots` as batch rows 0..B-1.

        Shares every pool tensor — only the assignment changes. This is what a
        scheduler hands to a forward, and what makes one captured graph valid
        across reassignment.
        """
        idx = (slots if isinstance(slots, torch.Tensor)
               else torch.tensor(slots, dtype=torch.int64, device=self.slots.device))
        idx = idx.to(self.slots.device)
        return Cache(layers=self.layers, slots=idx, lengths=self.lengths,
                     max_slots=self.max_slots, max_len=self.max_len,
                     slots_i32=idx.to(torch.int32))

    def snapshot(self) -> list[torch.Tensor]:
        """Copy every mutable pool. A decode step advances state, conv, KV *and*
        lengths, so rewinding only the lengths leaves the caller comparing two
        different starting states — which reads as a catastrophic mismatch and
        is really a bookkeeping error."""
        out = [self.lengths.clone()]
        for lyr in self.layers:
            if isinstance(lyr, KVCache):
                out += [lyr.k.clone(), lyr.v.clone()]
            else:
                out += [lyr.conv.clone(), lyr.state.clone()]
        return out

    def restore(self, snap: list[torch.Tensor]) -> None:
        it = iter(snap)
        self.lengths.copy_(next(it))
        for lyr in self.layers:
            if isinstance(lyr, KVCache):
                lyr.k.copy_(next(it))
                lyr.v.copy_(next(it))
            else:
                lyr.conv.copy_(next(it))
                lyr.state.copy_(next(it))

    @property
    def batch_lengths(self) -> torch.Tensor:
        """Committed token count for each row of the *current* assignment."""
        return self.lengths.index_select(0, self.slots)

    def bytes(self) -> int:
        n = 0
        for lyr in self.layers:
            if isinstance(lyr, KVCache):
                n += lyr.k.numel() * lyr.k.element_size() * 2
            else:
                n += (lyr.conv.numel() * lyr.conv.element_size()
                      + lyr.state.numel() * lyr.state.element_size())
        return n
