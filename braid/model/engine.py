"""The engine: a correct B=1 eager forward over the whole 32-layer hybrid.

ROADMAP Phase 2 item 3. **Correctness only — no speed claim is made or measured
here.** The GDN prefill is a per-token Python loop (`gdn.py`), attention runs on
the SDPA math backend because `head_dim=256` disqualifies every fused kernel,
and nothing is captured into a CUDA graph. All three are Phase 3's problem.

Scope, against the roadmap as originally written for the 35B:

  32 layers, not 40      24 GDN + 8 gated attention, period 4
  dense MLP, not MoE     `mlp_only_layers: []` on this checkpoint
  BF16, not NVFP4        so there is no dequantisation step at all
  tied LM head           there is no `lm_head` tensor in the file
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from braid.model.attention import RotaryEmbedding
from braid.model.cache import Cache
from braid.model.config import ModelConfig
from braid.model.layer import DecoderLayer
from braid.model.loader import Checkpoint, load_checkpoint
from braid.model.norm import rms_norm


@dataclass
class Engine:
    config: ModelConfig
    layers: list[DecoderLayer]
    embed_tokens: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor
    rope: RotaryEmbedding
    device: torch.device
    dtype: torch.dtype
    checkpoint: Checkpoint
    use_kernels: bool = False

    # --- construction --------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, ckpt: Checkpoint, device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.bfloat16,
                        use_kernels: bool = False) -> "Engine":
        cfg = ckpt.config
        layers = [DecoderLayer(cfg, i, ckpt.layer(i), use_kernels=use_kernels)
                  for i in range(cfg.num_hidden_layers)]
        return cls(
            config=cfg,
            layers=layers,
            embed_tokens=ckpt["embed_tokens"],
            final_norm=ckpt["norm"],
            lm_head=ckpt["lm_head"],
            rope=RotaryEmbedding(cfg, device, dtype),
            device=torch.device(device),
            dtype=dtype,
            checkpoint=ckpt,
            use_kernels=use_kernels,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.bfloat16,
                        use_kernels: bool = False) -> "Engine":
        return cls.from_checkpoint(load_checkpoint(path, device=device, dtype=dtype),
                                   device, dtype, use_kernels)

    def allocate_cache(self, max_len: int, max_slots: int = 1) -> Cache:
        # The conv kernel requires an fp32 window; the torch path keeps the
        # activation dtype so its numerics stay what Phase 2 measured.
        return Cache.allocate(self.config, max_slots, max_len, self.device, self.dtype,
                              conv_dtype=torch.float32 if self.use_kernels else None)

    # --- forward -------------------------------------------------------------

    @torch.no_grad()
    def hidden_states(
        self,
        input_ids: torch.Tensor,
        cache: Cache | None = None,
        last_only: bool = True,
    ) -> torch.Tensor:
        """`input_ids` `[B, T]` -> post-final-norm hidden states `[B, T_out, hidden]`.

        Separate from `forward` because the LM head is the expensive part to
        materialise: the vocab is 248,320 wide, so logits cost ~1 MB per token in
        bf16 and 2 GB for a 2,048-token window in fp32. Perplexity applies the
        head in slices over *these*; generation only ever needs the last row.

        With a cache, rows may be at **different lengths** — the batch is a set
        of independent sequences, not a padded rectangle. `positions` is
        therefore per row, and so is rope.
        """
        B, T = input_ids.shape
        slots = positions = attn_mask = slots_i32 = None
        kv_len = None

        if cache is not None:
            if cache.slots.numel() < B:
                raise ValueError(f"cache assigns {cache.slots.numel()} slots for B={B}")
            slots = cache.slots[:B]
            slots_i32 = cache.slots_i32[:B]
            positions = cache.lengths.index_select(0, slots)
            kv_len = int(positions.max().item()) + T
            if kv_len > cache.max_len:
                raise ValueError(f"KV overflow: {kv_len} > max_len {cache.max_len}")
            attn_mask = self._decode_mask(positions, T, kv_len)
            pos = positions[:, None] + torch.arange(T, device=self.device)[None]
        else:
            pos = torch.arange(T, device=self.device)[None].expand(B, T)

        h = F.embedding(input_ids, self.embed_tokens)
        cos, sin = self.rope(pos)

        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin,
                      cache=cache.layers[i] if cache is not None else None,
                      slots=slots, positions=positions, kv_len=kv_len,
                      attn_mask=attn_mask, slots_i32=slots_i32)

        if cache is not None:
            cache.lengths.index_copy_(0, slots, positions + T)

        if last_only:
            h = h[:, -1:]
        return rms_norm(h, self.final_norm, self.config.rms_norm_eps)

    def _decode_mask(self, positions: torch.Tensor, T: int,
                     kv_len: int) -> torch.Tensor | None:
        """`[B, 1, T, kv_len]` additive mask, or `None` when `is_causal` suffices.

        Two things have to be masked at once and they are easy to conflate:
        **causality** (query t may not see key t+1) and **occupancy** (row b's
        KV beyond its own length is another sequence's business, or zeros).
        When every row is at the same length and T == kv_len, SDPA's `is_causal`
        covers both; otherwise it covers neither correctly, because its mask is
        aligned top-left.
        """
        same_length = bool((positions == positions[0]).all())
        if same_length and T == kv_len:
            return None
        key = torch.arange(kv_len, device=self.device)
        # query q of row b sits at absolute position positions[b] + q
        q_abs = positions[:, None] + torch.arange(T, device=self.device)[None]   # [B, T]
        allowed = key[None, None, :] <= q_abs[:, :, None]                        # [B, T, kv]
        mask = torch.zeros(allowed.shape, device=self.device, dtype=self.dtype)
        return mask.masked_fill_(~allowed, torch.finfo(self.dtype).min)[:, None]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Cache | None = None,
        last_only: bool = True,
    ) -> torch.Tensor:
        """`input_ids` is `[B, T]` -> logits `[B, T_out, vocab]`."""
        return F.linear(self.hidden_states(input_ids, cache, last_only), self.lm_head)

    # --- the graph-safe decode step ------------------------------------------

    @torch.no_grad()
    def decode_step(self, tokens: torch.Tensor, cache: Cache) -> torch.Tensor:
        """One token per row -> logits `[B, vocab]`. **No host sync, static shapes.**

        `hidden_states` is the general path and cannot be captured: it reads
        `positions.max().item()` to size the KV slice and takes a Python `bool`
        of a device tensor to decide whether a mask is needed. Both are host
        syncs, and a host sync during capture is a capture failure.

        Two things are traded away to remove them:

        * **`kv_len` is fixed at `cache.max_len`** rather than the live maximum,
          because a shape that depends on device state cannot be a captured
          shape. Masked keys contribute `exp(-inf) = 0`, so the result is
          unchanged; the cost is reading the whole KV buffer every step. That is
          the argument for bucketing `kv_len` as well as batch, which item 3's
          block manager makes natural.
        * **The mask is always built**, never skipped by a data-dependent branch.
        """
        B = tokens.shape[0]
        if cache.slots.numel() < B:
            raise ValueError(f"cache assigns {cache.slots.numel()} slots for B={B}")
        slots, slots_i32 = cache.slots[:B], cache.slots_i32[:B]
        kv_len = cache.max_len

        positions = cache.lengths.index_select(0, slots)          # [B]
        key = torch.arange(kv_len, device=self.device)
        allowed = key[None, :] <= positions[:, None]              # [B, kv_len]
        mask = torch.zeros(B, kv_len, device=self.device, dtype=self.dtype)
        mask = mask.masked_fill_(~allowed, torch.finfo(self.dtype).min)[:, None, None]

        h = F.embedding(tokens, self.embed_tokens)
        cos, sin = self.rope(positions[:, None])

        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, cache=cache.layers[i], slots=slots,
                      positions=positions, kv_len=kv_len, attn_mask=mask,
                      slots_i32=slots_i32)

        cache.lengths.index_copy_(0, slots, positions + 1)
        h = rms_norm(h, self.final_norm, self.config.rms_norm_eps)
        return F.linear(h, self.lm_head)[:, 0]

    # --- generation ----------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        eos_token_id: int | None = None,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Single sequence. `[1, T] -> [1, n_new]`. See `generate_batch` for B>1."""
        if input_ids.shape[0] != 1:
            raise ValueError("generate takes one sequence; use generate_batch")
        out = self.generate_batch([input_ids[0].tolist()], max_new_tokens, temperature,
                                  top_p, seed, eos_token_id, max_len)
        return torch.tensor([out[0]], device=self.device, dtype=torch.long)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[list[int]],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        eos_token_id: int | None = None,
        max_len: int | None = None,
        slots: list[int] | None = None,
    ) -> list[list[int]]:
        """Prefill each prompt on its own, then decode all of them together.

        **Prefill is per sequence, decode is batched**, which is what a real
        continuous-batching engine does and is also what the two halves of this
        model can each support today: the GDN scan carries a per-sequence
        recurrence, so a padded rectangle would feed pad tokens through it and
        corrupt the state unless separately masked. Ragged batched prefill is
        Phase 3 item 3.

        `slots` assigns pool entries explicitly. Passing a non-identity
        permutation is the test that the indirection is real rather than an
        arange that happens to work.
        """
        B = len(prompts)
        longest = max(len(p) for p in prompts)
        max_len = max_len or longest + max_new_tokens + 1
        n_slots = max(B, (max(slots) + 1) if slots else 0)
        cache = self.allocate_cache(max_len, max_slots=n_slots)
        assign = list(range(B)) if slots is None else list(slots)
        if len(set(assign)) != B:
            raise ValueError(f"slot assignment {assign} reuses a slot")

        gen = (torch.Generator(device=self.device).manual_seed(seed)
               if seed is not None else None)

        # --- prefill, one sequence at a time into its own slot ---------------
        last_logits = torch.empty(B, self.config.vocab_size,
                                  device=self.device, dtype=self.dtype)
        for row, (prompt, slot) in enumerate(zip(prompts, assign)):
            cache.reset_slot(slot)
            one = cache.select([slot])
            ids = torch.tensor([prompt], device=self.device, dtype=torch.long)
            last_logits[row] = self.forward(ids, one)[0, -1]

        # --- decode, all rows together ---------------------------------------
        batch = cache.select(assign)
        out: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_new_tokens):
            nxt = self._sample(last_logits, temperature, top_p, gen)   # [B]
            for r, tok in enumerate(nxt.tolist()):
                if not done[r]:
                    out[r].append(tok)
                    if eos_token_id is not None and tok == eos_token_id:
                        done[r] = True
            if all(done):
                break
            last_logits = self.forward(nxt[:, None], batch)[:, -1]
        return out

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float,
                gen: torch.Generator | None) -> torch.Tensor:
        """`[B, vocab] -> [B]`. Every reduction is per row.

        A sampler that reads its parameters — or its RNG draw — from row 0 is one
        of the batch-leakage bugs the token-identity gate exists to catch, and it
        is invisible at B=1.
        """
        if temperature <= 0.0:
            return logits.argmax(-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1.0:
            srt, idx = probs.sort(dim=-1, descending=True)
            keep = (srt.cumsum(-1) - srt) < top_p
            srt = torch.where(keep, srt, torch.zeros_like(srt))
            probs = torch.zeros_like(probs).scatter_(-1, idx, srt)
            probs = probs / probs.sum(-1, keepdim=True)
        return torch.multinomial(probs, 1, generator=gen)[:, 0]

    # --- reporting -----------------------------------------------------------

    def weight_bytes(self) -> int:
        seen, total = set(), 0
        for t in self.checkpoint.tensors.values():
            if t.data_ptr() in seen:
                continue  # tied lm_head aliases embed_tokens
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
        return total
