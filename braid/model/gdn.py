"""The Gated DeltaNet layer — projections, causal conv, gates, scan, gated norm.

The recurrence itself is `braid.reference.gdn_ref.gdn_decode_vectorized`, the
same fp32 oracle the CUDA decode kernel is checked against. **Prefill runs that
one-token step in a loop.** That is slow and deliberate: it makes prefill and
decode the same arithmetic by construction rather than by test, so the classic
"generation drifts after the first token" bug cannot exist here. Phase 5's
ragged chunkwise scan replaces the loop; this phase makes no speed claim.

Gate arithmetic follows **HF, not the reference engine**:

    g    = A * softplus(a_raw + dt_bias)        with A = -exp(A_log), from the loader
    beta = sigmoid(b_raw)

The reference engine additionally clamps `A*dt` at -20 and `b_raw` at +-20
(`gdn.cu:89-96`, mirrored in `gdn_ref.gdn_gates`). Both clamps are no-ops on
real activations — `sigmoid(20)` is 1 to within 2e-9 — and HF is the
implementation the checkpoint was trained with, so parity beats the deviation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from braid.model.cache import RecurrentCache
from braid.model.config import ModelConfig
from braid.model.norm import rms_norm_gated
from braid.reference.gdn_ref import _l2norm, gdn_decode_vectorized


class GatedDeltaNet:
    """One `linear_attention` layer. Weights are `[out, in]`."""

    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor],
                 use_kernels: bool = False):
        self.cfg = cfg
        self.g = cfg.gdn
        self.use_kernels = use_kernels
        w = weights
        self.in_proj_qkv = w["linear_attn.in_proj_qkv"]
        self.in_proj_z = w["linear_attn.in_proj_z"]
        # `a` and `b` are two `[n_heads, hidden]` projections of the same x, and
        # the profile says they are expensive for their size: `[16, 2560] x
        # [2560, 32]` is two 16-wide tiles, i.e. **two CTAs on a 170-SM card**,
        # 15 us of device time apiece for 164 KiB of weights, 48 launches per
        # step across 24 layers. Concatenating them into one `[2560, 64]` GEMM
        # was tried and **rejected**: it is worth 0.35 ms/step at B=16 (2.7%,
        # barely over the 1.65% noise floor) and it moves bf16 prefill from
        # 8.3e-3 to 9.4e-3 against HF and flips a greedy token, because N=32 ->
        # N=64 changes cuBLAS's tile choice and so the reduction order. The
        # 0.35 ms is real and the way to collect it is a fused kernel that keeps
        # the arithmetic fixed, not a reshaped cuBLAS call.
        self.in_proj_a = w["linear_attn.in_proj_a"]
        self.in_proj_b = w["linear_attn.in_proj_b"]
        self.out_proj = w["linear_attn.out_proj"]
        self.conv1d = w["linear_attn.conv1d"]          # [C, K], fp/bf16
        self.conv_bias = w.get("linear_attn.conv1d_bias")
        self.A = w["linear_attn.A"]                    # fp32, already -exp(A_log)
        self.dt_bias = w["linear_attn.dt_bias"]        # fp32
        self.norm = w["linear_attn.norm"]              # fp32, gated -> NO 1+W

        if use_kernels:
            # The kernels are fp32 throughout. Keep the fp32 copies rather than
            # casting per step: a cast allocates, and allocating inside a
            # captured graph is a capture failure.
            self.conv1d_f32 = self.conv1d.float().contiguous()
            self.conv_bias_f32 = (self.conv_bias.float().contiguous()
                                  if self.conv_bias is not None
                                  else torch.empty(0, device=self.conv1d.device))
            from braid.kernels.loader import load_gdn

            self._mod = load_gdn()

    # --- the CUDA decode step ------------------------------------------------

    def _decode_kernels(self, x: torch.Tensor, cache: RecurrentCache,
                        slots_i32: torch.Tensor) -> torch.Tensor:
        """One token, all rows, through `conv1d_decode` + `gdn_decode`.

        The kernels read `slot_idx` on the device, so there is no gather and no
        scatter — the torch path's two 2 MiB-per-row-per-layer copies disappear.
        They also l2-normalise q and k internally, so raw q/k go in.

        Numerics differ from the torch path by design, and the difference is
        measured rather than assumed (`scripts/kernel_path_diag.py`): the conv
        runs in fp32 against the torch path's bf16, and the l2-norm follows it in
        fp32 rather than HF's bf16-then-widen. Both are *more* precise than the
        reference; neither is bit-identical to it.
        """
        g = self.g
        B = x.shape[0]
        mod = self._mod

        qkv = F.linear(x, self.in_proj_qkv)[:, 0].float().contiguous()   # [B, C]
        conv_out = torch.empty_like(qkv)
        mod.conv1d_decode(cache.conv, slots_i32, qkv,
                          self.conv1d_f32, self.conv_bias_f32, conv_out)

        key_dim = g.n_groups * g.state_size
        q, k, v = torch.split(conv_out, [key_dim, key_dim, g.inner_size], dim=-1)
        q = q.reshape(B, g.n_groups, g.state_size).contiguous()
        k = k.reshape(B, g.n_groups, g.state_size).contiguous()
        v = v.reshape(B, g.n_heads, g.head_dim).contiguous()

        a_raw = F.linear(x, self.in_proj_a)[:, 0]
        b_raw = F.linear(x, self.in_proj_b)[:, 0]
        beta = torch.sigmoid(b_raw).float().contiguous()
        alpha = torch.exp(self.A * F.softplus(a_raw.float() + self.dt_bias)).contiguous()

        y = torch.empty(B, g.n_heads, g.head_dim, device=x.device, dtype=torch.float32)
        mod.gdn_decode(cache.state, slots_i32, q, k, v, alpha, beta, y)
        return y[:, None]     # [B, 1, H, HD]

    # --- convolution ---------------------------------------------------------

    def _conv(self, x: torch.Tensor, cache: RecurrentCache | None,
              slots: torch.Tensor | None) -> torch.Tensor:
        """Depthwise causal conv + SiLU. `x` is `[B, C, T]`.

        SiLU is applied to **all** of Q, K and V, not just V.
        """
        B, C, T = x.shape
        K = self.g.conv_kernel
        w = self.conv1d.unsqueeze(1)  # [C, 1, K]

        # The pool may be fp32 (the CUDA conv kernel requires it) while the
        # activations are bf16, so every crossing is cast explicitly. Prefill
        # runs this path even under `use_kernels`, since the kernels decode only.
        if cache is not None:
            # Splice each row's own window in front — for T > 1 exactly as for
            # T == 1. The cached window holds the last K **pre-conv** inputs, so
            # `joined` is the true history and `conv1d` needs no padding at all:
            # output index i covers `joined[i : i+K-1]`, so outputs 1..T are
            # precisely x's tokens 0..T-1 and `[-T:]` selects them.
            #
            # The T > 1 branch used to left-pad with **zeros** (`padding=K-1`)
            # and ignore the window. For a fresh sequence that is the same thing
            # — the window is zeros — which is why every prefill test passed.
            # For a chunk landing on a *non-empty* cache it silently convolved
            # the first K-1 tokens as if the sequence started there: measured at
            # 1.6e-1 relative on the logits, in fp32, with the greedy token
            # still agreeing (`scripts/item3_gap_diag.py`). Fluent, not fatal,
            # which is the failure mode this codebase keeps having to catch.
            window = cache.conv.index_select(0, slots).to(x.dtype)   # [B, C, K]
            joined = torch.cat([window, x], dim=-1)                  # [B, C, K+T]
            cache.conv.index_copy_(0, slots, joined[:, :, -K:].to(cache.conv.dtype))
            out = F.conv1d(joined, w, self.conv_bias, padding=0, groups=C)
            return F.silu(out[:, :, -T:])

        # No cache: the sequence starts here, so zeros ARE its history.
        out = F.conv1d(x, w, self.conv_bias, padding=K - 1, groups=C)
        return F.silu(out[:, :, :T])

    # --- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor, cache: RecurrentCache | None = None,
                slots: torch.Tensor | None = None,
                slots_i32: torch.Tensor | None = None) -> torch.Tensor:
        cfg, g = self.cfg, self.g
        B, T, _ = x.shape
        dtype = x.dtype
        if cache is not None and slots is None:
            raise ValueError("a pooled cache needs a slot assignment")
        if cache is not None and T > 1 and B != 1:
            # The conv and the scan are both general in B. What is not general
            # is running one T for every row: rows in a pooled batch sit at
            # different lengths, so a rectangular [B, T] prefill would advance
            # them all by T from wherever each happens to be.
            raise NotImplementedError(
                f"multi-token GDN at B={B}; ragged batched prefill is Phase 3 item 3")

        if self.use_kernels and cache is not None and T == 1:
            y = self._decode_kernels(x, cache, slots_i32)
            return self._readout(x, y, B, T, dtype)

        qkv = F.linear(x, self.in_proj_qkv).transpose(1, 2)   # [B, C, T]
        qkv = self._conv(qkv, cache, slots).transpose(1, 2)   # [B, T, C]

        # [Q | K | V], Q FIRST. Settled empirically by tests/test_hf_parity.py,
        # not read off the reference engine — the two readings of it disagree and
        # the wrong order is fluent, not fatal.
        key_dim = g.n_groups * g.state_size
        q, k, v = torch.split(qkv, [key_dim, key_dim, g.inner_size], dim=-1)
        q = q.reshape(B, T, g.n_groups, g.state_size)
        k = k.reshape(B, T, g.n_groups, g.state_size)
        v = v.reshape(B, T, g.n_heads, g.head_dim)

        a_raw = F.linear(x, self.in_proj_a)
        b_raw = F.linear(x, self.in_proj_b)

        # beta's sigmoid is taken in the ACTIVATION dtype and only then widened;
        # `g` is computed entirely in fp32. That asymmetry is HF's (`beta =
        # b.sigmoid()` vs `g = -A_log.float().exp() * softplus(a.float() + ...)`)
        # and it is not cosmetic: taking beta's sigmoid in fp32 instead moves the
        # whole layer output by 4.8e-3 relative — one bf16 epsilon, flat across T
        # and flat across tokens, measured in scripts/gdn_layer_diag.py. beta is
        # the delta-rule step size, so its rounding lands straight in the output.
        beta = torch.sigmoid(b_raw).float()                                  # [B,T,H]
        alpha = torch.exp(self.A * F.softplus(a_raw.float() + self.dt_bias))  # [B,T,H]

        # l2norm in the ACTIVATION dtype, then cast — HF's order. Normalising
        # after the cast changes the last bits and compounds over 24 layers.
        qn = _l2norm(q).float()
        kn = _l2norm(k).float()
        vf = v.float()

        # Gather each row's slab, run the recurrence, scatter it back. The two
        # copies are 2 MiB per row per layer each way — the whole reason the
        # Phase 1 CUDA kernel takes `slot_idx` and does the indirection on the
        # device instead. This torch path exists to be obviously correct.
        if cache is not None:
            state = cache.state.index_select(0, slots)
        else:
            state = torch.zeros(B, g.n_heads, g.state_size, g.head_dim,
                                device=x.device, dtype=torch.float32)

        y = torch.empty(B, T, g.n_heads, g.head_dim, device=x.device, dtype=torch.float32)
        for t in range(T):
            y[:, t] = gdn_decode_vectorized(
                state=state, q=qn[:, t], k=kn[:, t], v=vf[:, t],
                alpha=alpha[:, t], beta=beta[:, t], cfg=g, normalize=False,
            )
        if cache is not None:
            cache.state.index_copy_(0, slots, state)

        return self._readout(x, y, B, T, dtype)

    def _readout(self, x: torch.Tensor, y: torch.Tensor, B: int, T: int,
                 dtype: torch.dtype) -> torch.Tensor:
        """Gated norm then out_proj. Shared by both scan paths.

        HF casts the scan output back to the activation dtype BEFORE gating, so
        this cast is part of the reference and not a rounding convenience.
        """
        g = self.g
        core = y.to(dtype).reshape(-1, g.head_dim)
        z = F.linear(x, self.in_proj_z).reshape(-1, g.head_dim)
        core = rms_norm_gated(core, z, self.norm, self.cfg.rms_norm_eps)
        return F.linear(core.reshape(B, T, g.inner_size), self.out_proj)

    __call__ = forward
