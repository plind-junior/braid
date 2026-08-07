import pytest
import torch

from braid.config import GDNConfig
from braid.kernels.loader import load_gdn
from braid.reference.gdn_ref import gdn_decode_vectorized

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _mk(B, cfg, dev="cuda", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    return dict(
        q=torch.randn(B, cfg.n_groups, cfg.state_size, generator=g, device=dev),
        k=torch.randn(B, cfg.n_groups, cfg.state_size, generator=g, device=dev),
        v=torch.randn(B, cfg.n_heads, cfg.head_dim, generator=g, device=dev),
        alpha=torch.rand(B, cfg.n_heads, generator=g, device=dev) * 0.5 + 0.5,
        beta=torch.rand(B, cfg.n_heads, generator=g, device=dev),
    )


@pytest.mark.parametrize("B", [1, 2, 8, 32])
def test_kernel_matches_oracle(B):
    cfg = GDNConfig(n_heads=8, head_dim=128, state_size=128, n_groups=4)
    mod = load_gdn()
    pool = torch.randn(64, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
    slots = torch.arange(B, dtype=torch.int32, device="cuda")

    inp = _mk(B, cfg)
    ref_state = pool[:B].clone()
    y_ref = gdn_decode_vectorized(state=ref_state, cfg=cfg, **inp)

    y = torch.empty(B, cfg.n_heads, cfg.head_dim, device="cuda")
    mod.gdn_decode(pool, slots, inp["q"], inp["k"], inp["v"], inp["alpha"], inp["beta"], y)

    torch.testing.assert_close(y, y_ref, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(pool[:B], ref_state, rtol=2e-5, atol=2e-6)


def test_nonidentity_slots_are_honoured():
    """Row b must touch pool[slot_idx[b]] and nothing else."""
    cfg = GDNConfig(n_heads=4, head_dim=128, state_size=128, n_groups=2)
    mod = load_gdn()
    pool = torch.randn(16, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
    untouched = pool.clone()
    slots = torch.tensor([7, 3], dtype=torch.int32, device="cuda")

    inp = _mk(2, cfg)
    ref = pool[slots.long()].clone()
    y_ref = gdn_decode_vectorized(state=ref, cfg=cfg, **inp)

    y = torch.empty(2, cfg.n_heads, cfg.head_dim, device="cuda")
    mod.gdn_decode(pool, slots, inp["q"], inp["k"], inp["v"], inp["alpha"], inp["beta"], y)

    torch.testing.assert_close(y, y_ref, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(pool[7], ref[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(pool[3], ref[1], rtol=2e-5, atol=2e-6)
    for s in (0, 1, 2, 4, 5, 6, 8):
        torch.testing.assert_close(pool[s], untouched[s])


def test_rows_do_not_leak_into_each_other():
    """Row 1's output must not depend on row 0's inputs.

    The single property the whole batched design exists to preserve, checked
    against the kernel rather than the oracle. A shared-memory buffer sized
    per-block but indexed per-grid would pass every test above and fail this.
    """
    cfg = GDNConfig(n_heads=8, head_dim=128, state_size=128, n_groups=4)
    mod = load_gdn()

    pool_a = torch.randn(8, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
    pool_b = pool_a.clone()
    slots2 = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    slots1 = torch.tensor([1], dtype=torch.int32, device="cuda")

    two = _mk(2, cfg, seed=11)
    y2 = torch.empty(2, cfg.n_heads, cfg.head_dim, device="cuda")
    mod.gdn_decode(pool_a, slots2, two["q"], two["k"], two["v"], two["alpha"], two["beta"], y2)

    solo = {kk: vv[1:2].contiguous() for kk, vv in two.items()}
    y1 = torch.empty(1, cfg.n_heads, cfg.head_dim, device="cuda")
    mod.gdn_decode(pool_b, slots1, solo["q"], solo["k"], solo["v"], solo["alpha"], solo["beta"], y1)

    torch.testing.assert_close(y2[1:2], y1, rtol=0, atol=0)
    torch.testing.assert_close(pool_a[1], pool_b[1], rtol=0, atol=0)


def test_repeated_slot_is_rejected():
    """Two rows aliasing one state slab is silent cross-sequence corruption.

    The reference engine has exactly this hazard: engine_sampling_stop.cpp:256-262
    falls back to `slot = req_id % cap` when the pool is exhausted, putting two live
    sequences on the same 63.8 MiB slab with no fault and no log. Refuse it.
    """
    cfg = GDNConfig(n_heads=4, head_dim=128, state_size=128, n_groups=2)
    mod = load_gdn()
    pool = torch.randn(8, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
    slots = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    inp = _mk(2, cfg)
    y = torch.empty(2, cfg.n_heads, cfg.head_dim, device="cuda")
    with pytest.raises(RuntimeError, match="duplicate|distinct"):
        mod.gdn_decode(pool, slots, inp["q"], inp["k"], inp["v"], inp["alpha"], inp["beta"], y)


def test_out_of_range_slot_is_rejected():
    cfg = GDNConfig(n_heads=4, head_dim=128, state_size=128, n_groups=2)
    mod = load_gdn()
    pool = torch.randn(8, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
    slots = torch.tensor([0, 99], dtype=torch.int32, device="cuda")
    inp = _mk(2, cfg)
    y = torch.empty(2, cfg.n_heads, cfg.head_dim, device="cuda")
    with pytest.raises(RuntimeError, match="range|bounds"):
        mod.gdn_decode(pool, slots, inp["q"], inp["k"], inp["v"], inp["alpha"], inp["beta"], y)
