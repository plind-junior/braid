"""The T=1 GQA attention path, against the SDPA call it replaces.

Phase 3 item 3. `grouped_decode_attention` exists because at `head_dim = 256`
every fused SDPA backend declines and the math fallback replicates K and V to
the full head count — 3.1 ms per step at B=16 to feed 1.3 ms of matmul.

The gate that actually protects anything here is **not** the numeric one. Both
forms are a softmax over the same dot products, so a rewrite that got the
*grouping* backwards would still produce finite, plausible, fluent output — it
would just have every head attending the wrong kv head. `test_head_h_reads_kv_head_h_over_groups`
is the one that fails in that case; the tolerance tests would not.

No checkpoint needed: this pins the kernel-level contract, not the model.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from braid.model.attention import grouped_decode_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

B, H, KVH, D, L = 3, 16, 4, 256, 96
G = H // KVH
SCALE = D ** -0.5


def _inputs(dtype: torch.dtype, seed: int = 3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda", dtype=dtype)
    return r(B, H, 1, D), r(B, KVH, L, D), r(B, KVH, L, D)


def _mask(dtype: torch.dtype) -> torch.Tensor:
    """`[B, 1, 1, L]` additive mask, one live prefix length per row."""
    lengths = torch.tensor([L, L // 2, 7], device="cuda")
    key = torch.arange(L, device="cuda")
    blocked = key[None, :] > lengths[:, None]
    return torch.zeros(B, L, device="cuda", dtype=dtype).masked_fill_(
        blocked, torch.finfo(dtype).min)[:, None, None]


def _sdpa(q, k, v, mask):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False,
                                          scale=SCALE, enable_gqa=True)


# --- against the path it replaces --------------------------------------------

@pytest.mark.parametrize("masked", [False, True])
def test_fp32_matches_sdpa_to_machine_precision(masked):
    q, k, v = _inputs(torch.float32)
    mask = _mask(torch.float32) if masked else None
    got = grouped_decode_attention(q, k, v, mask, SCALE, G)
    want = _sdpa(q, k, v, mask)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("masked", [False, True])
def test_bf16_is_within_the_bf16_noise_floor_of_fp32(masked):
    """The reference is **fp32**, deliberately, and not SDPA's bf16 arm.

    SDPA-bf16 looks like the obvious baseline and is not a valid one: at
    `head_dim = 256` the math backend carries extra internal precision, which
    is most of why it is slow. Its key-scale kernel moves 268 MB for a tensor
    that is 134 MB in bf16, and it lands 2x closer to fp32 than a bf16
    computation can. Comparing against it measures that promotion, not us —
    and would fail a correct kernel. Three independent causes for the gap were
    tested and refuted before this reference was changed (`scripts/decode_attn_diag.py`):
    matmul shape (grouped M=4 is numerically identical to an expanded M=1),
    `allow_bf16_reduced_precision_reduction`, and scale placement (the scale
    is `2**-4`, exact). What remains is the bf16 rounding of the softmax
    output, which every fused attention kernel also pays.

    So the gate is Phase 2's rel L2 <= 5e-3 against fp32, plus a floor ratio:
    rounding the fp32 answer to bf16 is the best *any* bf16-output kernel can
    do, and this path must stay within 3x of it. That catches an arithmetic
    regression, which an absolute tolerance on flat random scores would not.
    """
    q, k, v = _inputs(torch.bfloat16)
    mask = _mask(torch.bfloat16) if masked else None

    got = grouped_decode_attention(q, k, v, mask, SCALE, G).float()
    # Same bf16 inputs, widened: input rounding is common to both arms and
    # cancels, so what is left is the arithmetic.
    ref = grouped_decode_attention(q.float(), k.float(), v.float(),
                                   None if mask is None else mask.float(), SCALE, G)
    floor = (ref.bfloat16().float() - ref).norm() / ref.norm()

    rel = (got - ref).norm() / ref.norm()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    assert rel <= 5e-3, f"rel L2 {rel:.3e} vs fp32"
    assert cos >= 0.99999, f"cosine {cos:.9f} vs fp32"
    assert rel <= 3 * floor, (
        f"rel L2 {rel:.3e} is {rel / floor:.2f}x the irreducible bf16 output "
        f"rounding ({floor:.3e}); arithmetic regressed")


# --- the gate that a tolerance test cannot give you ---------------------------

def test_head_h_reads_kv_head_h_over_groups():
    """Blocked grouping, not interleaved. Swapping the two is fluent garbage.

    Each kv head is given a one-hot V, so the output of head `h` names the kv
    head it actually attended. `q.reshape(B, G, KVH, D)` — the transposed
    reading — sends head 1 to kv head 1 instead of kv head 0, and every
    numeric test above still passes because the values are equally plausible.
    """
    q = torch.ones(1, H, 1, D, device="cuda", dtype=torch.float32)
    k = torch.zeros(1, KVH, L, D, device="cuda", dtype=torch.float32)
    v = torch.zeros(1, KVH, L, D, device="cuda", dtype=torch.float32)
    for kvh in range(KVH):
        v[0, kvh, :, kvh] = 1.0          # kv head j marks channel j

    o = grouped_decode_attention(q, k, v, None, SCALE, G)[0, :, 0]   # [H, D]
    read = o[:, :KVH].argmax(-1).tolist()
    assert read == [h // G for h in range(H)], (
        f"head->kv map is {read}, want blocked {[h // G for h in range(H)]}")


def test_it_refuses_a_multi_token_query():
    q, k, v = _inputs(torch.float32)
    with pytest.raises(ValueError, match="T=1 path"):
        grouped_decode_attention(q.expand(B, H, 1, D).repeat(1, 1, 2, 1),
                                 k, v, None, SCALE, G)


def test_a_fully_masked_row_is_not_silently_nan():
    """`kv_len` is pinned to `max_len`, so rows are mostly masked in practice."""
    q, k, v = _inputs(torch.float32)
    mask = torch.zeros(B, 1, 1, L, device="cuda", dtype=torch.float32)
    mask[:, :, :, 1:] = torch.finfo(torch.float32).min      # one live key
    o = grouped_decode_attention(q, k, v, mask, SCALE, G)
    assert torch.isfinite(o).all()
    torch.testing.assert_close(
        o, v[:, :, 0].repeat_interleave(G, dim=1).reshape(B, H, 1, D),
        rtol=1e-5, atol=1e-5)
