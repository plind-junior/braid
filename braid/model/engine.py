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

    # --- construction --------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, ckpt: Checkpoint, device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.bfloat16) -> "Engine":
        cfg = ckpt.config
        layers = [DecoderLayer(cfg, i, ckpt.layer(i)) for i in range(cfg.num_hidden_layers)]
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
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.bfloat16) -> "Engine":
        return cls.from_checkpoint(load_checkpoint(path, device=device), device, dtype)

    def allocate_cache(self, max_len: int, batch: int = 1) -> Cache:
        return Cache.allocate(self.config, batch, max_len, self.device, self.dtype)

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
        """
        B, T = input_ids.shape
        past = cache.seq_len if cache is not None else 0

        h = F.embedding(input_ids, self.embed_tokens)
        pos = torch.arange(past, past + T, device=self.device)[None].expand(B, T)
        cos, sin = self.rope(pos)

        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, cache=cache.layers[i] if cache is not None else None)

        if cache is not None:
            cache.seq_len += T

        if last_only:
            h = h[:, -1:]
        return rms_norm(h, self.final_norm, self.config.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Cache | None = None,
        last_only: bool = True,
    ) -> torch.Tensor:
        """`input_ids` is `[B, T]` -> logits `[B, T_out, vocab]`."""
        return F.linear(self.hidden_states(input_ids, cache, last_only), self.lm_head)

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
        """Greedy at `temperature == 0`, else top-p sampling. Returns `[B, n_new]`."""
        if input_ids.shape[0] != 1:
            raise NotImplementedError("B=1 only in Phase 2; the batch axis is Phase 3")
        cache = self.allocate_cache(max_len or input_ids.shape[1] + max_new_tokens + 1)
        gen = (torch.Generator(device=self.device).manual_seed(seed)
               if seed is not None else None)

        logits = self.forward(input_ids, cache)[:, -1]
        out: list[int] = []
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_p, gen)
            tok = int(nxt.item())
            out.append(tok)
            if eos_token_id is not None and tok == eos_token_id:
                break
            logits = self.forward(nxt[None], cache)[:, -1]
        return torch.tensor([out], device=self.device, dtype=torch.long)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float,
                gen: torch.Generator | None) -> torch.Tensor:
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
