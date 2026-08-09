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

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from braid.model.attention import RotaryEmbedding
from braid.model.cache import Cache
from braid.model.config import ModelConfig
from braid.model.layer import DecoderLayer
from braid.model.loader import Checkpoint, load_checkpoint
from braid.model.norm import rms_norm
from braid.model.quant import (
    FP8Weight,
    fused_state,
    linear,
    maybe_quantize,
    parse_groups,
    warm_fused,
)


def checkpoint_bytes(ckpt: Checkpoint) -> int:
    """Distinct bytes held by a checkpoint's tensors."""
    seen, total = set(), 0
    for t in ckpt.tensors.values():
        if t.data_ptr() in seen:
            continue  # a tied lm_head aliases embed_tokens
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total


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
    # The checkpoint's own footprint, captured as a number so that reporting it
    # does not require holding the tensors. See `source`.
    ckpt_bytes: int
    # The checkpoint this engine was built from, **or None if anything was
    # quantized**. An unquantized engine aliases the checkpoint's tensors, so
    # keeping it costs nothing and lets a caller build a sibling engine over the
    # same weights (`use_kernels=True`, say) for free. A *quantized* engine does
    # not alias them — `maybe_quantize` allocates fp8 copies and the bf16
    # originals would stay reachable through this field forever, leaving a "half
    # the bytes" engine resident at 14.78 + 7.40 GiB and OOMing the B=64 capture
    # on a 31 GiB card. So it is dropped exactly when it would be a duplicate.
    source: Checkpoint | None = None
    use_kernels: bool = False
    quant: frozenset[str] = field(default_factory=frozenset)
    # Storage type for the recurrent state pool. See `RecurrentCache`: only the
    # storage narrows, the scan is fp32 either way.
    state_dtype: torch.dtype = torch.float32

    # --- construction --------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, ckpt: Checkpoint, device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.bfloat16,
                        use_kernels: bool = False,
                        quant: str | Iterable[str] | None = None,
                        state_dtype: torch.dtype = torch.float32) -> Engine:
        cfg = ckpt.config
        groups = parse_groups(quant)
        # Built here, not on first use. `quantize_act` runs inside the captured
        # decode step, and the first call is the one that would JIT-compile the
        # extension — building inside a graph capture is not a thing to find out
        # about at capture time. Only when something is actually quantized: an
        # unquantized engine never calls it and should not pay a build.
        if groups and torch.cuda.is_available():
            warm_fused()
        layers = [DecoderLayer(cfg, i, ckpt.layer(i), use_kernels=use_kernels,
                               quant=groups)
                  for i in range(cfg.num_hidden_layers)]
        return cls(
            config=cfg,
            layers=layers,
            embed_tokens=ckpt["embed_tokens"],
            final_norm=ckpt["norm"],
            # `embed_tokens` is gathered, never swept, so it is left alone even
            # under "all" — quantizing it would cost accuracy for no bandwidth.
            # `lm_head` is a full GEMM every step and is the one that pays.
            lm_head=maybe_quantize(ckpt["lm_head"], "head" in groups),
            rope=RotaryEmbedding(cfg, device, dtype),
            device=torch.device(device),
            dtype=dtype,
            ckpt_bytes=checkpoint_bytes(ckpt),
            source=None if groups else ckpt,
            use_kernels=use_kernels,
            quant=groups,
            state_dtype=state_dtype,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.bfloat16,
                        use_kernels: bool = False,
                        quant: str | Iterable[str] | None = None,
                        state_dtype: torch.dtype = torch.float32) -> Engine:
        return cls.from_checkpoint(load_checkpoint(path, device=device, dtype=dtype),
                                   device, dtype, use_kernels, quant, state_dtype)

    def set_chunk_prefill(self, enabled: bool) -> int:
        """Turn the `gdn_prefill` chunk scan on or off across every GDN layer.

        **This exists to be measured, not configured.** Serving always wants it
        on; the bench needs it off to say what it was worth. The two arms are
        still run as separate processes, because the measurement contract asks
        for medians over independent processes — what the toggle buys is that
        those processes differ in exactly one attribute and share every line of
        construction, rather than being two code paths that drifted apart.

        Off falls back to the Python loop over columns, which is the same
        arithmetic the torch path has always run. Returns the number of layers
        touched: a toggle that silently matched nothing (an engine built with
        `use_kernels=False`, say) would otherwise report a 0% effect and look
        like a refutation of the kernel rather than of the measurement.
        """
        n = 0
        for layer in self.layers:
            mixer = getattr(layer, "mixer", None)
            if hasattr(mixer, "chunk_prefill"):
                # `use_kernels=False` has no kernel module to call, so enabling
                # the chunk scan there is not a slower arm, it is an AttributeError.
                mixer.chunk_prefill = enabled and mixer.use_kernels
                n += 1
        return n

    def set_raw_gates(self, enabled: bool) -> int:
        """Compute alpha/beta inside the GDN kernels, or in torch.

        The other single-attribute A/B lever, same contract as
        `set_chunk_prefill`: serving always wants it on, the bench turns it off
        to price it, and the return value exists so an arm that silently
        matched nothing cannot masquerade as a null result. The two paths are
        asserted bit-identical (tests/test_gdn_raw_gates.py), so this is a
        launch-count switch, never a numerics one.
        """
        n = 0
        for layer in self.layers:
            mixer = getattr(layer, "mixer", None)
            if hasattr(mixer, "raw_gates"):
                mixer.raw_gates = enabled and mixer.use_kernels
                n += 1
        return n

    def allocate_cache(self, max_len: int, max_slots: int = 1,
                       state_dtype: torch.dtype | None = None) -> Cache:
        # The conv kernel requires an fp32 window; the torch path keeps the
        # activation dtype so its numerics stay what Phase 2 measured.
        #
        # `state_dtype` defaults to this engine's `state_dtype`, so a server
        # configured once gets narrow state everywhere without every call site
        # repeating it; passing it explicitly is for the gates that compare a
        # narrow pool against an fp32 one.
        return Cache.allocate(self.config, max_slots, max_len, self.device, self.dtype,
                              conv_dtype=torch.float32 if self.use_kernels else None,
                              state_dtype=state_dtype or self.state_dtype)

    # --- forward -------------------------------------------------------------

    @torch.no_grad()
    def hidden_states(
        self,
        input_ids: torch.Tensor,
        cache: Cache | None = None,
        last_only: bool = True,
        seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`input_ids` `[B, T]` -> post-final-norm hidden states `[B, T_out, hidden]`.

        Separate from `forward` because the LM head is the expensive part to
        materialise: the vocab is 248,320 wide, so logits cost ~1 MB per token in
        bf16 and 2 GB for a 2,048-token window in fp32. Perplexity applies the
        head in slices over *these*; generation only ever needs the last row.

        With a cache, rows may be at **different lengths** — the batch is a set
        of independent sequences, not a padded rectangle. `positions` is
        therefore per row, and so is rope.

        **Ragged batched prefill.** `seq_lens[b] <= T` says how many leading
        columns of row b are real; the rest are padding and may hold any valid
        token id. Every row still advances from its own `positions[b]`, now by
        its own `seq_lens[b]`. `None` means the rectangular case, every row
        owning all T columns.

        Padding is on the **right**, and that choice is load-bearing rather than
        conventional: a causal model never lets a real token at index t read
        index > t, so right-padding is invisible to the arithmetic. Left-padding
        would put pad tokens inside every real token's receptive field and would
        have to be masked in the conv, the scan and attention separately.
        """
        B, T = input_ids.shape
        slots = positions = attn_mask = slots_i32 = None
        kv_len = None

        if seq_lens is not None:
            if cache is None:
                raise ValueError("seq_lens describes a pooled batch; it needs a cache")
            if seq_lens.shape != (B,):
                raise ValueError(f"seq_lens must be [{B}], got {tuple(seq_lens.shape)}")

        if cache is not None:
            if cache.slots.numel() < B:
                raise ValueError(f"cache assigns {cache.slots.numel()} slots for B={B}")
            slots = cache.slots[:B]
            slots_i32 = cache.slots_i32[:B]
            positions = cache.lengths.index_select(0, slots)
            ends = positions + T if seq_lens is None else positions + seq_lens
            # One host sync, here, for the whole forward. `KVCache.write` trusts
            # it rather than repeating the check per attention layer.
            kv_len = int(ends.max().item())
            if not 0 < kv_len <= cache.max_len:
                raise ValueError(f"KV span {kv_len} outside (0, max_len={cache.max_len}]")
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
                      attn_mask=attn_mask, slots_i32=slots_i32, seq_lens=seq_lens)

        if cache is not None:
            cache.lengths.index_copy_(0, slots, ends)

        if last_only:
            if seq_lens is None:
                h = h[:, -1:]
            else:
                # Row b's last real token sits at seq_lens[b] - 1, not at T - 1.
                idx = (seq_lens - 1).clamp_(min=0)[:, None, None]
                h = h.gather(1, idx.expand(B, 1, h.shape[-1]))
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
        seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`input_ids` is `[B, T]` -> logits `[B, T_out, vocab]`."""
        return linear(self.hidden_states(input_ids, cache, last_only, seq_lens),
                      self.lm_head)

    # --- the graph-safe decode step ------------------------------------------

    @torch.no_grad()
    def decode_step(self, tokens: torch.Tensor, cache: Cache,
                    kv_len: int | None = None) -> torch.Tensor:
        """One token per row -> logits `[B, vocab]`. **No host sync, static shapes.**

        `hidden_states` is the general path and cannot be captured: it reads
        `positions.max().item()` to size the KV slice and takes a Python `bool`
        of a device tensor to decide whether a mask is needed. Both are host
        syncs, and a host sync during capture is a capture failure.

        Two things are traded away to remove them:

        * **`kv_len` is a caller-supplied constant**, defaulting to
          `cache.max_len`. It cannot be read from the cache here: a shape that
          depends on device state is a host sync, and a host sync during capture
          is a capture failure. Masked keys contribute `exp(-inf) = 0`, so any
          `kv_len` at or above every live row's length gives the same answer —
          the cost of overshooting is pure bandwidth. Measured at `max_len=512`
          with 8-token prompts, the default reads **64x** the KV it needs, which
          is what `GraphedDecoder`'s `kv_buckets` exists to cut. A scheduler
          tracks lengths on the host anyway, so picking the bucket is free
          there; it is only inside the graph that it would cost a sync.
        * **The mask is always built**, never skipped by a data-dependent branch.
        """
        B = tokens.shape[0]
        if cache.slots.numel() < B:
            raise ValueError(f"cache assigns {cache.slots.numel()} slots for B={B}")
        slots, slots_i32 = cache.slots[:B], cache.slots_i32[:B]
        if kv_len is None:
            kv_len = cache.max_len
        elif not 0 < kv_len <= cache.max_len:
            raise ValueError(f"kv_len {kv_len} outside (0, max_len={cache.max_len}]")

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
        return linear(h, self.lm_head)[:, 0]

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
        """Prefill every prompt as one ragged batch, then decode them together.

        Prompts of different lengths are right-padded to the longest and the
        real lengths handed down as `seq_lens`; pad steps are an exact identity
        on the recurrent state, so a short row's state is what it would have
        been prefilled alone. The padding is wasted work, and how much depends
        entirely on the spread of prompt lengths — a scheduler that wants a
        bound on that chunks instead (`braid/serve/scheduler.py`).

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

        # --- prefill, all prompts as one ragged batch -------------------------
        for slot in assign:
            cache.reset_slot(slot)
        batch = cache.select(assign)
        # Pad with 0 rather than leaving the buffer uninitialised: pad columns
        # are embedded and convolved like any other token, and a garbage id
        # would index outside the embedding table.
        ids = torch.tensor([p + [0] * (longest - len(p)) for p in prompts],
                           device=self.device, dtype=torch.long)
        lens = torch.tensor([len(p) for p in prompts], device=self.device)
        last_logits = self.forward(ids, batch, seq_lens=lens)[:, -1]

        # --- decode, all rows together ---------------------------------------
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
        """The **checkpoint's** footprint, i.e. what was loaded from the file.

        Unchanged by quantization on purpose: this is the "did the visual tower
        leak in" number. For what the engine actually reads per decode step, use
        `step_bytes`.
        """
        return self.ckpt_bytes

    def _projections(self) -> dict[str, list]:
        """Every weight a decode step sweeps, by group. Not `embed_tokens`,
        which is gathered per token rather than read."""
        out: dict[str, list] = {"mlp": [], "attn": [], "gdn": [], "head": []}
        for lyr in self.layers:
            m = lyr.mixer
            out["mlp"] += [lyr.mlp.gate_proj, lyr.mlp.up_proj, lyr.mlp.down_proj]
            if lyr.is_gdn:
                out["gdn"] += [m.in_proj_qkv, m.in_proj_z, m.in_proj_a,
                               m.in_proj_b, m.out_proj]
            else:
                out["attn"] += [m.q_proj, m.k_proj, m.v_proj, m.o_proj]
        out["head"] = [self.lm_head]
        return out

    def step_bytes(self) -> dict[str, int]:
        """Bytes of weight a single decode step reads, per group.

        **This is the number the head-to-head turns on.** Decode is weight-bound
        at the batches braid is built for, so the ratio of this total against the
        competitor's is the first-order predictor of the throughput ratio, and it
        is the only quantity FP8 moves.
        """
        def nbytes(w):
            return (w.nbytes if isinstance(w, FP8Weight)
                    else w.numel() * w.element_size())

        return {g: sum(nbytes(w) for w in ws) for g, ws in self._projections().items()}

    def quant_report(self) -> str:
        """What was actually quantized, and what it bought — one line per group.

        Stated rather than assumed because `maybe_quantize` **declines shapes**
        it cannot handle (`_scaled_mm` needs both dims a multiple of 16) and
        `gdn` deliberately leaves `in_proj_a` / `in_proj_b` alone. A run can ask
        for a group and get less of it than it asked for, and a benchmark that
        reported the request instead of the result would be publishing a label,
        not a measurement.
        """
        lines, tot_now, tot_bf16 = [], 0, 0
        for g, ws in self._projections().items():
            if not ws:
                continue
            q = sum(1 for w in ws if isinstance(w, FP8Weight))
            now = sum((w.nbytes if isinstance(w, FP8Weight)
                       else w.numel() * w.element_size()) for w in ws)
            bf16 = sum((w.data.numel() * 2 if isinstance(w, FP8Weight)
                        else w.numel() * w.element_size()) for w in ws)
            tot_now += now
            tot_bf16 += bf16
            asked = "on" if g in self.quant else "off"
            lines.append(f"  {g:<5} {q:>4}/{len(ws):<4} fp8  {now / 2 ** 30:>6.2f} GiB"
                         f"  (bf16 {bf16 / 2 ** 30:>6.2f})  [{asked}]")
        head = (f"decode-step weights {tot_now / 2 ** 30:.2f} GiB of "
                f"{tot_bf16 / 2 ** 30:.2f} bf16 ({tot_now / tot_bf16:.3f}x)")
        # Printed for the same reason the per-group counts are: the fused
        # quantizer falls back to an exact torch spelling if it will not build,
        # so a run can be nine-kernels-per-quantize while looking identical in
        # every other respect. A launch-count change that vanishes silently from
        # a benchmark is the failure mode this line exists to prevent.
        tail = f"  act quantizer: {fused_state()}" if self.quant else ""
        return head + "\n" + "\n".join(lines) + ("\n" + tail if tail else "")
