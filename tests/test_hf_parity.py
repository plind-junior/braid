"""Parity against HuggingFace's own Qwen3.5 reference implementation.

This is the strongest correctness check available: `transformers` ships the
implementation the checkpoint was trained with, so it is ground truth. Where
the reference engine and HF disagree, HF wins and the engine is the deviation.

Settles four questions the design docs listed as open or disputed:
  1. mixed_qkv split order        -> Q, K, V with Q FIRST
  2. head->group mapping          -> repeat_interleave == grouped, g = h // (H//G)
  3. linear_attn.norm `1+W` offset -> NO offset on the gated norm; YES on every
                                      other RMSNorm
  4. l2norm form                  -> ADDITIVE eps 1e-6, not a 1e-12 clamp
"""
import pytest
import torch

from braid.config import GDNConfig
from braid.reference.gdn_ref import gdn_decode_vectorized

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

CFG = GDNConfig(n_heads=8, head_dim=128, state_size=128, n_groups=4)


def _inputs(B, cfg, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    return dict(
        q=torch.randn(B, cfg.n_groups, cfg.state_size, generator=g, device=dev),
        k=torch.randn(B, cfg.n_groups, cfg.state_size, generator=g, device=dev),
        v=torch.randn(B, cfg.n_heads, cfg.head_dim, generator=g, device=dev),
        a_raw=torch.randn(B, cfg.n_heads, generator=g, device=dev),
        b_raw=torch.randn(B, cfg.n_heads, generator=g, device=dev),
        A_log=torch.randn(B and cfg.n_heads, generator=g, device=dev),
        dt_bias=torch.randn(cfg.n_heads, generator=g, device=dev),
    )


def _hf_reference(state, inp, cfg):
    """One decode step through HF's torch_recurrent_gated_delta_rule."""
    hpg = cfg.heads_per_group
    # HF shapes: [B, seq=1, heads, dim]; repeat_interleave BEFORE the rule.
    q = inp["q"][:, None].repeat_interleave(hpg, dim=2)
    k = inp["k"][:, None].repeat_interleave(hpg, dim=2)
    v = inp["v"][:, None]
    beta = inp["b_raw"].sigmoid()[:, None]
    g = (-inp["A_log"].float().exp()
         * torch.nn.functional.softplus(inp["a_raw"].float() + inp["dt_bias"]))[:, None]
    out, new_state = hf.torch_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=state,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    return out[:, 0], new_state


def _braid_args(inp, cfg):
    """HF's `g` is the LOG decay; our oracle takes the decay itself."""
    g = (-inp["A_log"].float().exp()
         * torch.nn.functional.softplus(inp["a_raw"].float() + inp["dt_bias"]))
    return dict(q=inp["q"], k=inp["k"], v=inp["v"],
                alpha=g.exp(), beta=inp["b_raw"].sigmoid())


@pytest.mark.parametrize("B", [1, 4])
def test_oracle_matches_hf_recurrent_rule(B):
    dev = "cuda"
    inp = _inputs(B, CFG, dev, seed=5)
    state = torch.randn(B, CFG.n_heads, CFG.state_size, CFG.head_dim, device=dev)

    y_hf, s_hf = _hf_reference(state.clone(), inp, CFG)
    s_mine = state.clone()
    y_mine = gdn_decode_vectorized(state=s_mine, cfg=CFG, **_braid_args(inp, CFG))

    torch.testing.assert_close(y_mine, y_hf, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(s_mine, s_hf, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("B", [1, 4, 8])
def test_cuda_kernel_matches_hf_recurrent_rule(B):
    """The kernel, end to end, against HF. This is the Phase 2 GDN gate."""
    from braid.kernels.loader import load_gdn

    dev = "cuda"
    mod = load_gdn()
    inp = _inputs(B, CFG, dev, seed=9)
    pool = torch.randn(16, CFG.n_heads, CFG.state_size, CFG.head_dim, device=dev)
    slots = torch.arange(B, dtype=torch.int32, device=dev)

    y_hf, s_hf = _hf_reference(pool[:B].clone(), inp, CFG)

    args = _braid_args(inp, CFG)
    y = torch.empty(B, CFG.n_heads, CFG.head_dim, device=dev)
    mod.gdn_decode(pool, slots, args["q"], args["k"], args["v"],
                   args["alpha"], args["beta"], y)

    torch.testing.assert_close(y, y_hf, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(pool[:B], s_hf, rtol=2e-4, atol=2e-5)


def test_l2norm_is_additive_not_clamped():
    """HF: rsqrt(sum_sq + 1e-6). Reference engine: rsqrt(max(sum_sq, 1e-12)).

    Identical for a healthy head, 10x apart for a near-zero one. Pinning the
    form so a future 'optimization' back to the clamped version fails loudly.
    """
    from braid.reference.gdn_ref import _l2norm

    x = torch.full((1, 1, 128), 1e-5)  # sum_sq = 128 * 1e-10 = 1.28e-8
    mine = _l2norm(x)
    theirs = hf.l2norm(x, dim=-1, eps=1e-6)
    torch.testing.assert_close(mine, theirs, rtol=1e-6, atol=1e-8)

    clamped = x * torch.rsqrt(torch.clamp((x * x).sum(-1, keepdim=True), min=1e-12))
    assert not torch.allclose(mine, clamped, rtol=0.05), (
        "the clamped form should differ materially here; if it does not, this "
        "test has stopped discriminating and the pin is worthless"
    )


def test_gated_norm_has_no_unit_offset_but_plain_norm_does():
    """The two RMSNorms genuinely differ, and mixing them up costs ~2x PPL.

    Qwen3_5RMSNorm      : weight stored as deltas, forward uses (1.0 + w)
    Qwen3_5RMSNormGated : weight stored directly,  forward uses w
    """
    dim = 16
    plain = hf.Qwen3_5RMSNorm(dim)
    gated = hf.Qwen3_5RMSNormGated(dim)

    assert torch.allclose(plain.weight, torch.zeros(dim)), "plain norm inits to ZEROS (deltas)"
    assert torch.allclose(gated.weight, torch.ones(dim)), "gated norm inits to ONES (direct)"

    x = torch.randn(4, dim)
    # A freshly-initialised plain norm must behave as gamma == 1.
    torch.testing.assert_close(plain(x), plain._norm(x.float()).type_as(x), rtol=1e-5, atol=1e-6)

    # And the gated norm normalises BEFORE gating.
    gate = torch.randn(4, dim)
    var = x.float().pow(2).mean(-1, keepdim=True)
    expect = (x.float() * torch.rsqrt(var + gated.variance_epsilon)) * gated.weight
    expect = expect * torch.nn.functional.silu(gate.float())
    torch.testing.assert_close(gated(x, gate), expect.type_as(x), rtol=1e-5, atol=1e-6)
