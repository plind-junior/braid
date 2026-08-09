"""The Gated DeltaNet layer — projections, causal conv, gates, scan, gated norm.

The recurrence itself is `braid.reference.gdn_ref.gdn_decode_vectorized`, the
same fp32 oracle the CUDA decode kernel is checked against. **Prefill runs that
one-token step in a loop.** That is deliberate: it makes prefill and decode the
same arithmetic by construction rather than by test, so the classic "generation
drifts after the first token" bug cannot exist here.

The loop costs one iteration per **column**, not per token, and each iteration
is a handful of small kernels — so it is launch-bound, and a `[B, T]` batch
costs what a `[1, T]` batch costs. That is why ragged batched prefill
(`seq_lens`, below) is the lever and not a faster single-sequence scan: it
divides the same fixed loop across B rows. Cutting the iteration *count* — a
chunkwise scan that keeps the state resident across C tokens — is the next lever
and a separate one.

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
from braid.model.quant import FP8Weight, linear, maybe_quantize, project
from braid.reference.gdn_ref import _l2norm, gdn_decode_vectorized


class GatedDeltaNet:
    """One `linear_attention` layer. Weights are `[out, in]`."""

    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor],
                 use_kernels: bool = False, quant: bool = False):
        self.cfg = cfg
        self.g = cfg.gdn
        self.use_kernels = use_kernels
        w = weights
        self.in_proj_qkv = maybe_quantize(w["linear_attn.in_proj_qkv"], quant)
        self.in_proj_z = maybe_quantize(w["linear_attn.in_proj_z"], quant)
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
        #
        # **`a` and `b` are never quantized**, and that is deliberate rather
        # than an oversight. They are `[n_heads, hidden]` — 0.4% of the layer's
        # weight bytes — so there is nothing to save, and the GEMM is two CTAs,
        # so it is launch-bound and fp8 cannot make it faster. What it could do
        # is damage: `a_raw` goes through `exp(A * softplus(a + dt_bias))`, so
        # fp8's three mantissa bits would land inside an exponent and the error
        # would be amplified rather than averaged. Cost nil, risk real.
        self.in_proj_a = w["linear_attn.in_proj_a"]
        self.in_proj_b = w["linear_attn.in_proj_b"]
        self.out_proj = maybe_quantize(w["linear_attn.out_proj"], quant)
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

    @property
    def quantized(self) -> bool:
        return isinstance(self.in_proj_qkv, FP8Weight)

    # --- the CUDA decode step ------------------------------------------------

    def _decode_kernels(self, qkv_proj: torch.Tensor, a_raw: torch.Tensor,
                        b_raw: torch.Tensor, cache: RecurrentCache,
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
        B = qkv_proj.shape[0]
        mod = self._mod

        qkv = qkv_proj[:, 0].float().contiguous()                        # [B, C]
        conv_out = torch.empty_like(qkv)
        mod.conv1d_decode(cache.conv, slots_i32, qkv,
                          self.conv1d_f32, self.conv_bias_f32, conv_out)

        key_dim = g.n_groups * g.state_size
        q, k, v = torch.split(conv_out, [key_dim, key_dim, g.inner_size], dim=-1)
        q = q.reshape(B, g.n_groups, g.state_size).contiguous()
        k = k.reshape(B, g.n_groups, g.state_size).contiguous()
        v = v.reshape(B, g.n_heads, g.head_dim).contiguous()

        beta = torch.sigmoid(b_raw[:, 0]).float().contiguous()
        alpha = torch.exp(
            self.A * F.softplus(a_raw[:, 0].float() + self.dt_bias)).contiguous()

        y = torch.empty(B, g.n_heads, g.head_dim, device=qkv.device,
                        dtype=torch.float32)
        mod.gdn_decode(cache.state, slots_i32, q, k, v, alpha, beta, y)
        return y[:, None]     # [B, 1, H, HD]

    # --- convolution ---------------------------------------------------------

    def _conv(self, x: torch.Tensor, cache: RecurrentCache | None,
              slots: torch.Tensor | None,
              seq_lens: torch.Tensor | None = None) -> torch.Tensor:
        """Depthwise causal conv + SiLU. `x` is `[B, C, T]`.

        SiLU is applied to **all** of Q, K and V, not just V.

        With `seq_lens`, `x` is right-padded: row b owns columns
        `0 .. seq_lens[b]-1`. The conv *outputs* need no masking — a causal conv
        reads only backwards, so a real output never sees a pad column — but the
        **cached window must not**, or the next chunk splices pad tokens in as
        history and the sequence quietly convolves against tokens it never saw.
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
            if seq_lens is None:
                keep = joined[:, :, -K:]
            else:
                # Row b's real inputs are joined[K : K+seq_lens[b]], so its last
                # K entries of true history end there: joined[len : len+K].
                # `-K:` would take the pad columns instead. At seq_lens == T the
                # two expressions are the same slice, which is what makes the
                # rectangular path a special case rather than a separate one.
                idx = seq_lens[:, None] + torch.arange(K, device=x.device)[None]
                keep = joined.gather(2, idx[:, None].expand(B, C, K))
            cache.conv.index_copy_(0, slots, keep.to(cache.conv.dtype))
            out = F.conv1d(joined, w, self.conv_bias, padding=0, groups=C)
            return F.silu(out[:, :, -T:])

        # No cache: the sequence starts here, so zeros ARE its history.
        out = F.conv1d(x, w, self.conv_bias, padding=K - 1, groups=C)
        return F.silu(out[:, :, :T])

    # --- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor, cache: RecurrentCache | None = None,
                slots: torch.Tensor | None = None,
                slots_i32: torch.Tensor | None = None,
                seq_lens: torch.Tensor | None = None) -> torch.Tensor:
        """`x` is `[B, T, hidden]`.

        `seq_lens[b] <= T` is how many leading columns of row b are real; `None`
        means all T. Rows advance from their **own** `positions[b]` by their own
        `seq_lens[b]`, which is what makes a ragged batch legal here at all — a
        padded rectangle with no length vector would run pad tokens through the
        recurrence and leave every short row's state advanced past its prompt.
        """
        g = self.g
        B, T, _ = x.shape
        dtype = x.dtype
        if cache is not None and slots is None:
            raise ValueError("a pooled cache needs a slot assignment")
        if seq_lens is not None and cache is None:
            raise ValueError("seq_lens describes a pooled batch; it needs a cache")

        # Every projection in this block reads the same x, so they are issued
        # together: when the block is quantized that is one activation quantize
        # for the four of them instead of four. `in_proj_a` / `in_proj_b` stay
        # bf16 and `project` leaves them on the ordinary path.
        qkv_proj, z, a_raw, b_raw = project(
            x, self.in_proj_qkv, self.in_proj_z, self.in_proj_a, self.in_proj_b)

        if self.use_kernels and cache is not None and T == 1:
            y = self._decode_kernels(qkv_proj, a_raw, b_raw, cache, slots_i32)
            return self._readout(z, y, B, T, dtype)

        qkv = qkv_proj.transpose(1, 2)                                   # [B, C, T]
        qkv = self._conv(qkv, cache, slots, seq_lens).transpose(1, 2)    # [B, T, C]

        # [Q | K | V], Q FIRST. Settled empirically by tests/test_hf_parity.py,
        # not read off the reference engine — the two readings of it disagree and
        # the wrong order is fluent, not fatal.
        key_dim = g.n_groups * g.state_size
        q, k, v = torch.split(qkv, [key_dim, key_dim, g.inner_size], dim=-1)
        q = q.reshape(B, T, g.n_groups, g.state_size)
        k = k.reshape(B, T, g.n_groups, g.state_size)
        v = v.reshape(B, T, g.n_heads, g.head_dim)

        # beta's sigmoid is taken in the ACTIVATION dtype and only then widened;
        # `g` is computed entirely in fp32. That asymmetry is HF's (`beta =
        # b.sigmoid()` vs `g = -A_log.float().exp() * softplus(a.float() + ...)`)
        # and it is not cosmetic: taking beta's sigmoid in fp32 instead moves the
        # whole layer output by 4.8e-3 relative — one bf16 epsilon, flat across T
        # and flat across tokens, measured in scripts/gdn_layer_diag.py. beta is
        # the delta-rule step size, so its rounding lands straight in the output.
        beta = torch.sigmoid(b_raw).float()                                  # [B,T,H]
        alpha = torch.exp(self.A * F.softplus(a_raw.float() + self.dt_bias))  # [B,T,H]

        if seq_lens is not None:
            # A pad step must be an exact identity on the state, not merely a
            # small one. `alpha = 1, beta = 0` gives
            #     delta = (v - 1*kv) * 0 = 0        (v and kv are finite, so
            #                                        this is a true zero, not a
            #                                        0 * NaN)
            #     S'    = 1*S + k (x) 0 = S         bit-identically.
            # So a padded row's state is the same object it would have been had
            # the row not been in the batch at all. y is garbage on those steps
            # and is never read. Masking the *output* instead would be wrong:
            # the state is what carries across chunks.
            live = (torch.arange(T, device=x.device)[None]
                    < seq_lens[:, None])[:, :, None]                # [B, T, 1]
            alpha = torch.where(live, alpha, 1.0)
            beta = torch.where(live, beta, 0.0)

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

        return self._readout(z, y, B, T, dtype)

    def _readout(self, z: torch.Tensor, y: torch.Tensor, B: int, T: int,
                 dtype: torch.dtype) -> torch.Tensor:
        """Gated norm then out_proj. Shared by both scan paths.

        Takes the already-projected gate `z` rather than `x`, so that the single
        activation quantization in `forward` covers this projection too.

        HF casts the scan output back to the activation dtype BEFORE gating, so
        this cast is part of the reference and not a rounding convenience.
        """
        g = self.g
        core = y.to(dtype).reshape(-1, g.head_dim)
        core = rms_norm_gated(core, z.reshape(-1, g.head_dim), self.norm,
                              self.cfg.rms_norm_eps)
        return linear(core.reshape(B, T, g.inner_size), self.out_proj)

    __call__ = forward
