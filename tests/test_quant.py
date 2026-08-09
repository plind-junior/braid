"""FP8 W8A8 on the projections, by group (`mlp`, `attn`, `gdn`, `head`).

The interesting thing about this path is that its gate cannot be a tolerance.
e4m3 has 3 mantissa bits, so a single GEMM lands ~3.7e-2 from bf16 — roughly
10x braid's 5e-3 parity gate, and no correct implementation does better. A
tolerance tight enough to catch a bug would fail the working code, and one loose
enough to pass would catch nothing.

So the gates here are structural (does the quantized weight actually round-trip,
does the scale sit where `_scaled_mm` expects it, does the shape guard fire) and
the *numeric* gate lives in `braid/bench/perplexity.py`, where +0.50% on the
whole corpus is a claim worth making. That split is deliberate.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from braid.model.quant import (
    FP8,
    FP8_MAX,
    GROUPS,
    FP8Weight,
    fp8_linear,
    linear,
    maybe_quantize,
    parse_groups,
    project,
    quantize_act,
    quantize_weight,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _w(N=512, K=256, scale=0.02):
    g = torch.Generator(device="cuda").manual_seed(7)
    return torch.randn(N, K, generator=g, device="cuda", dtype=torch.bfloat16) * scale


# --- the weight itself --------------------------------------------------------

def test_quantized_weight_is_half_the_bytes():
    w = _w()
    q = quantize_weight(w)
    assert q.data.dtype is FP8
    assert q.nbytes < w.numel() * 2 / 1.9, "fp8 should be ~half of bf16"


def test_dequantizing_recovers_the_weight_to_fp8_precision():
    """3 mantissa bits is ~6% per element; the tensor-relative error is ~3.6e-2."""
    w = _w()
    q = quantize_weight(w)
    back = q.data.float() * q.scale
    rel = ((back - w.float()).norm() / w.float().norm()).item()
    assert rel < 6e-2, f"round-trip {rel:.3e} is worse than e4m3 allows"
    assert rel > 1e-3, f"round-trip {rel:.3e} is too good — is it really fp8?"


def test_the_scale_is_a_device_tensor_not_a_python_float():
    """A Python float would be re-boxed per call, and an allocation inside a
    captured graph is a capture failure."""
    q = quantize_weight(_w())
    assert isinstance(q.scale, torch.Tensor) and q.scale.is_cuda
    assert q.scale.shape == (1, 1) and q.scale.dtype is torch.float32


def test_the_two_scale_paths_agree():
    """`_weight_scale` avoids an fp32 image of the tensor; it must still be the
    same number `_act_scale` computes the obvious way."""
    from braid.model.quant import _act_scale, _weight_scale
    for w in (_w(512, 256), -_w(512, 256).abs(), _w(64, 32) * 100):
        torch.testing.assert_close(_weight_scale(w), _act_scale(w), rtol=0, atol=0)


def test_a_weight_too_large_for_one_fp32_temporary_is_banded():
    """The lm_head is `[248320, 4096]` — an fp32 image of it is 3.8 GiB and OOMs
    a card that is already holding the model, which is precisely the situation
    where quantizing it is worth doing. Banding must not change the result."""
    from braid.model import quant as q

    w = _w(2048, 256)
    full = quantize_weight(w)
    old = q._CAST_CHUNK_BYTES
    q._CAST_CHUNK_BYTES = 256 * 4 * 3        # 3 rows at a time
    try:
        banded = quantize_weight(w)
    finally:
        q._CAST_CHUNK_BYTES = old
    assert torch.equal(banded.data, full.data)
    torch.testing.assert_close(banded.scale, full.scale, rtol=0, atol=0)


def test_an_all_zero_weight_does_not_produce_nans():
    q = quantize_weight(torch.zeros(64, 32, device="cuda", dtype=torch.bfloat16))
    assert torch.isfinite(q.data.float()).all() and (q.scale > 0).all()


@pytest.mark.parametrize("shape", [(30, 256), (512, 250)])
def test_shapes_scaled_mm_cannot_take_are_refused(shape):
    w = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiples of 16"):
        quantize_weight(w)
    # ...and `maybe_quantize` declines rather than raising, because it sweeps a
    # whole model whose projection widths differ.
    assert maybe_quantize(w, True) is w


def test_maybe_quantize_is_a_no_op_when_disabled():
    w = _w()
    assert maybe_quantize(w, False) is w
    assert isinstance(maybe_quantize(w, True), FP8Weight)


# --- the GEMM -----------------------------------------------------------------

def test_fp8_linear_agrees_with_bf16_to_fp8_precision():
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    w = _w()
    got = fp8_linear(x, quantize_weight(w))
    ref = F.linear(x, w)
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 8e-2, f"rel {rel:.3e} — worse than e4m3 quantization explains"
    cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.99, f"cosine {cos:.6f} — the product is wrong, not just coarse"


def test_leading_dims_survive():
    x = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    out = fp8_linear(x, quantize_weight(_w()))
    assert out.shape == (2, 3, 512) and out.dtype is torch.bfloat16


def test_linear_dispatches_on_weight_type():
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    w = _w()
    torch.testing.assert_close(linear(x, w), F.linear(x, w), rtol=0, atol=0)
    assert linear(x, quantize_weight(w)).shape == (4, 512)


def test_activation_quantization_uses_the_full_fp8_range():
    """A scale that leaves headroom is throwing away mantissa bits for nothing."""
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    xq, sx = quantize_act(x)
    assert xq.dtype is FP8 and sx.shape == (1, 1)
    peak = xq.float().abs().max().item()
    assert peak > FP8_MAX * 0.5, f"peak {peak} — the scale wastes range"


# --- end to end ---------------------------------------------------------------

def test_quantized_mlp_tracks_the_bf16_one():
    from types import SimpleNamespace

    from braid.model.mlp import MLP

    # A stub, not a real ModelConfig: MLP reads exactly one field, and building
    # a whole config here would couple this test to rope validation it does not
    # exercise. `test_full_forward` covers the real config.
    cfg = SimpleNamespace(hidden_act="silu")
    g = torch.Generator(device="cuda").manual_seed(3)
    r = lambda N, K: torch.randn(N, K, generator=g, device="cuda",
                                 dtype=torch.bfloat16) * 0.05
    w = {"mlp.gate_proj": r(512, 256), "mlp.up_proj": r(512, 256),
         "mlp.down_proj": r(256, 512)}

    x = torch.randn(2, 1, 256, device="cuda", dtype=torch.bfloat16)
    ref = MLP(cfg, w, quant=False)(x)
    got = MLP(cfg, w, quant=True)(x)

    assert got.shape == ref.shape and got.dtype is ref.dtype
    cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.99, f"cosine {cos:.6f}"


def test_quantized_mlp_reports_itself():
    """`Engine` has to be able to state the split it actually applied — a
    projection silently declined for its shape is the failure mode."""
    from braid.model.quant import maybe_quantize
    assert isinstance(maybe_quantize(_w(512, 256), True), FP8Weight)
    assert not isinstance(maybe_quantize(_w(512, 250), True), FP8Weight)


# --- group selection ----------------------------------------------------------

def test_group_names_round_trip():
    assert parse_groups("") == frozenset()
    assert parse_groups(None) == frozenset()
    assert parse_groups("mlp") == {"mlp"}
    assert parse_groups("mlp, attn") == {"mlp", "attn"}
    assert parse_groups(["gdn", "head"]) == {"gdn", "head"}
    assert parse_groups("all") == frozenset(GROUPS)


def test_an_unknown_group_raises_rather_than_quietly_doing_nothing():
    """A typo that parsed to the empty set would produce a run labelled
    quantized that is not, and the label is what reaches the table."""
    with pytest.raises(ValueError, match="unknown quant group"):
        parse_groups("mpl")
    with pytest.raises(ValueError, match="unknown quant group"):
        parse_groups("mlp,attention")


# --- one activation shared across several projections -------------------------

def test_project_matches_linear_per_weight():
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    a, b = _w(512, 256), _w(128, 256, scale=0.03)
    qa, qb = quantize_weight(a), quantize_weight(b)

    got_a, got_b = project(x, qa, qb)
    torch.testing.assert_close(got_a, fp8_linear(x, qa), rtol=0, atol=0)
    torch.testing.assert_close(got_b, fp8_linear(x, qb), rtol=0, atol=0)


def test_project_leaves_unquantized_weights_on_the_bf16_path():
    """A mixed set is the normal case — `gdn` quantizes qkv/z/out and never
    touches in_proj_a / in_proj_b — and the bf16 ones must be untouched."""
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    a, b = _w(512, 256), _w(128, 256)
    got_a, got_b = project(x, quantize_weight(a), b)
    assert isinstance(got_a, torch.Tensor)
    torch.testing.assert_close(got_b, F.linear(x, b), rtol=0, atol=0)


def test_project_with_no_fp8_weights_is_exactly_f_linear():
    x = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    a, b = _w(512, 256), _w(128, 256)
    got_a, got_b = project(x, a, b)
    torch.testing.assert_close(got_a, F.linear(x, a), rtol=0, atol=0)
    torch.testing.assert_close(got_b, F.linear(x, b), rtol=0, atol=0)


def test_project_preserves_leading_dims():
    x = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    out, = project(x, quantize_weight(_w(512, 256)))
    assert out.shape == (2, 3, 512) and out.dtype is torch.bfloat16


# --- the real checkpoint: which weights actually became fp8 -------------------

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
needs_model = pytest.mark.skipif(not MODEL_DIR.exists(),
                                 reason=f"no checkpoint at {MODEL_DIR}")


def _engine(quant):
    from braid.model.engine import Engine
    from braid.model.loader import load_checkpoint

    ck = load_checkpoint(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16, quant=quant)
    del ck
    gc.collect()
    torch.cuda.empty_cache()
    return eng


@needs_model
@pytest.mark.parametrize("group", GROUPS)
def test_a_group_quantizes_itself_and_nothing_else(group):
    """Asking for one group must not quietly quantize a neighbour. `attn` and
    `gdn` in particular sit on alternating layers of the same stack."""
    eng = _engine(group)
    try:
        got = {g: [isinstance(w, FP8Weight) for w in ws]
               for g, ws in eng._projections().items()}
        assert all(got[group]) or group == "gdn", f"{group} was not quantized"
        if group == "gdn":
            # a and b are excluded by design; everything else in the group goes.
            assert sum(got["gdn"]) == len(got["gdn"]) * 3 // 5
        for other in set(GROUPS) - {group}:
            assert not any(got[other]), f"asking for {group} also quantized {other}"
    finally:
        del eng
        gc.collect()
        torch.cuda.empty_cache()


@needs_model
def test_the_gdn_gates_stay_bf16_even_under_all():
    """`in_proj_a` feeds `exp(A * softplus(...))`, so fp8's three mantissa bits
    would land inside an exponent. It is 0.4% of the layer's bytes, so there is
    nothing to win and something to lose."""
    eng = _engine("all")
    try:
        gdn = [lyr.mixer for lyr in eng.layers if lyr.is_gdn]
        assert gdn, "no linear_attention layers in this checkpoint"
        for m in gdn:
            assert not isinstance(m.in_proj_a, FP8Weight)
            assert not isinstance(m.in_proj_b, FP8Weight)
            assert isinstance(m.in_proj_qkv, FP8Weight)
            assert isinstance(m.in_proj_z, FP8Weight)
            assert isinstance(m.out_proj, FP8Weight)
    finally:
        del eng
        gc.collect()
        torch.cuda.empty_cache()


@needs_model
def test_embed_tokens_is_never_quantized():
    """It is gathered per token, not swept, so quantizing it buys no bandwidth
    and costs accuracy on every embedding lookup."""
    eng = _engine("all")
    try:
        assert isinstance(eng.embed_tokens, torch.Tensor)
        assert eng.embed_tokens.dtype is torch.bfloat16
    finally:
        del eng
        gc.collect()
        torch.cuda.empty_cache()


@needs_model
def test_each_group_removes_its_own_bytes_from_the_decode_step():
    """The claim the head-to-head turns on: decode is weight-bound, so what FP8
    is worth is exactly the bytes it removes from `step_bytes`."""
    base = _engine("")
    want = base.step_bytes()
    del base
    gc.collect()
    torch.cuda.empty_cache()

    for group in GROUPS:
        eng = _engine(group)
        try:
            got = eng.step_bytes()
            assert got[group] < want[group] * 0.75, (
                f"{group}: {got[group]} vs bf16 {want[group]} — barely moved")
            for other in set(GROUPS) - {group}:
                assert got[other] == want[other], f"{group} moved {other}'s bytes"
        finally:
            del eng
            gc.collect()
            torch.cuda.empty_cache()


@needs_model
def test_quant_report_states_the_result_not_the_request():
    eng = _engine("gdn")
    try:
        rep = eng.quant_report()
        assert "[on]" in rep and "[off]" in rep
        # 3 of the 5 gdn projections are quantizable; the report must show that
        # rather than claiming the group.
        line = next(x for x in rep.splitlines() if x.strip().startswith("gdn"))
        n_q, n = line.split()[1].split("/")
        assert int(n_q) == int(n) * 3 // 5, line
    finally:
        del eng
        gc.collect()
        torch.cuda.empty_cache()


# --- the fused activation quantizer -------------------------------------------
#
# `quantize_act` is nine kernels in torch and two in CUDA, and it runs ~129 times
# per decode step — measured at ~1.18 ms of a 9.75 ms step at B=1 on the 9B.
# Because it feeds every fp8 GEMM, the gate is **bit-identity against the torch
# spelling**, not a tolerance. That is a reasonable thing to demand here for one
# specific reason: the scale is an amax, and max is associative *and exact* in
# floating point, so the reduction order the grid happens to use cannot change
# it. A sum would not have this property and would not get this gate.

def _quant_arms(x):
    """(fused, torch) results for the same input, with the flag left restored."""
    from braid.model import quant as Q

    Q.warm_fused(True)
    if Q.fused_state() != "active":
        pytest.skip(f"fused quantizer not available: {Q.fused_state()}")
    fused = Q.quantize_act(x)
    Q.warm_fused(False)
    try:
        ref = Q.quantize_act(x)
    finally:
        Q.warm_fused(True)
    return fused, ref


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float32, torch.float16],
                         ids=["bf16", "fp32", "fp16"])
@pytest.mark.parametrize("shape", [(1, 4096), (8, 4096), (37, 12288), (128, 12288)],
                         ids=["m1", "m8", "m37", "m128"])
def test_the_fused_quantizer_is_bit_identical_to_the_torch_spelling(dt, shape):
    g = torch.Generator(device="cuda").manual_seed(17)
    x = (torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32) * 3.5).to(dt)
    (q, s), (qr, sr) = _quant_arms(x)

    assert torch.equal(s, sr), f"scale differs: {s.item():.9e} vs {sr.item():.9e}"
    # Compare the fp8 payloads as raw bytes; fp8 has no `==` worth trusting.
    assert torch.equal(q.view(torch.uint8), qr.view(torch.uint8)), (
        f"{dt} {shape}: "
        f"{int((q.view(torch.uint8) != qr.view(torch.uint8)).sum())} of {q.numel()} "
        f"fp8 bytes differ")


def test_the_fused_quantizer_handles_the_degenerate_scales():
    """An all-zero activation must not divide by zero, and a huge one must
    saturate rather than produce a NaN. Both are reachable: a masked row can be
    exactly zero, and the `clamp(min=EPS)` in the torch spelling exists for it."""
    for label, x in (
        ("zeros", torch.zeros(4, 4096, device="cuda", dtype=torch.bfloat16)),
        ("huge", torch.full((4, 4096), 60000.0, device="cuda", dtype=torch.bfloat16)),
        # 1e-10, not 1e-8: the clamp band is 0 < amax < 448*EPS = 4.48e-10, and
        # 1e-8 sits above it — that case would exercise nothing degenerate. At
        # 1e-10 the scale clamps to EPS with a nonzero payload, which is the
        # branch this case exists to cover.
        ("tiny", torch.full((4, 4096), 1e-10, device="cuda", dtype=torch.bfloat16)),
    ):
        (q, s), (qr, sr) = _quant_arms(x)
        assert torch.isfinite(s).all(), f"{label}: scale not finite"
        assert torch.equal(s, sr), f"{label}: scale differs from torch"
        assert torch.equal(q.view(torch.uint8), qr.view(torch.uint8)), (
            f"{label}: payload differs from torch")


def test_a_non_contiguous_activation_falls_back_rather_than_corrupting():
    """The kernel indexes linearly and so requires contiguity; the wrapper checks
    it and falls back. The failure this guards is silent: a strided view read as
    if it were dense returns the wrong elements at full speed.

    So the reference is the **contiguous copy**, which does take the fused path.
    Same values, same layout, therefore the same scale and the same payload —
    comparing the view against the torch spelling instead would only assert that
    the fallback equals itself.
    """
    from braid.model import quant as Q

    Q.warm_fused(True)
    if Q.fused_state() != "active":
        pytest.skip(Q.fused_state())
    g = torch.Generator(device="cuda").manual_seed(5)
    base = torch.randn(64, 4096, generator=g, device="cuda", dtype=torch.bfloat16)
    view = base.t()
    assert not view.is_contiguous()

    q, s = Q.quantize_act(view)                 # falls back, strided
    qc, sc = Q.quantize_act(view.contiguous())  # fused, dense
    assert torch.equal(s, sc), "the fallback found a different amax than the kernel"
    assert torch.equal(q.view(torch.uint8), qc.view(torch.uint8)), (
        "a strided activation quantized to different bytes than its own "
        "contiguous copy")


def test_a_nan_activation_survives_quantization():
    """A NaN must reach the output, not be clamped away.

    `fmaxf(NaN, x)` returns x, so the obvious spelling of a max-reduction
    silently swallows NaN while `torch.amax` propagates it. If the kernel did
    that, a diverged activation would quantize to a full tensor of plausible
    -448..448 fp8 and the run would keep going looking healthy — the failure
    would be invisible exactly when it matters most. `quant_act.cu` uses a
    comparison-based max for this reason, and this is the test that says so.
    """
    from braid.model import quant as Q

    Q.warm_fused(True)
    if Q.fused_state() != "active":
        pytest.skip(Q.fused_state())
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(4, 4096, generator=g, device="cuda", dtype=torch.bfloat16)
    x[2, 100] = float("nan")

    (q, s), (qr, sr) = _quant_arms(x)
    assert torch.isnan(s).all(), (
        f"a NaN activation produced a finite scale {s.item():.3e} — the "
        f"reduction swallowed it")
    assert torch.isnan(sr).all(), "the torch reference should propagate NaN too"
    assert torch.equal(q.view(torch.uint8), qr.view(torch.uint8)), (
        "fused and torch disagree on how a NaN activation quantizes")
