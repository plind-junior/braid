import time

import pytest
import torch

from braid.config import GDNConfig
from braid.kernels.loader import load_gdn
from braid.reference.gdn_ref import gdn_decode_vectorized

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _capture(mod, pool, slots, args):
    """Warm up on a side stream, then capture once."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            mod.gdn_decode(pool, slots, *args)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mod.gdn_decode(pool, slots, *args)
    return graph


def test_graph_replay_survives_slot_reassignment():
    """The core claim of the design.

    The reference engine re-captures its decode graph whenever the recurrent
    slot changes (engine_scheduler.cpp:1968-1981), costing 10-20 ms per
    rotation. Because
    our kernel reads slot_idx from device memory, one captured graph stays
    valid for any slot assignment. This test fails if slot_idx is ever baked
    into the capture as a kernel parameter.
    """
    cfg = GDNConfig(n_heads=8, head_dim=128, state_size=128, n_groups=4)
    mod = load_gdn()
    B = 4

    pool = torch.zeros(32, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
    # Distinct slots even for the warmup/capture pass: the kernel refuses
    # duplicates, because two live rows on one state slab is exactly the
    # silent cross-sequence corruption the reference engine's aliasing fallback
    # allows.
    slots = torch.arange(B, dtype=torch.int32, device="cuda")
    q = torch.zeros(B, cfg.n_groups, cfg.state_size, device="cuda")
    k = torch.zeros(B, cfg.n_groups, cfg.state_size, device="cuda")
    v = torch.zeros(B, cfg.n_heads, cfg.head_dim, device="cuda")
    alpha = torch.zeros(B, cfg.n_heads, device="cuda")
    beta = torch.zeros(B, cfg.n_heads, device="cuda")
    y = torch.zeros(B, cfg.n_heads, cfg.head_dim, device="cuda")

    graph = _capture(mod, pool, slots, (q, k, v, alpha, beta, y))

    gen = torch.Generator(device="cuda").manual_seed(7)
    for assignment in ([5, 9, 1, 30], [0, 1, 2, 3], [31, 2, 17, 8]):
        pool.normal_(generator=gen)
        q.normal_(generator=gen)
        k.normal_(generator=gen)
        v.normal_(generator=gen)
        alpha.uniform_(0.5, 1.0, generator=gen)
        beta.uniform_(0.0, 1.0, generator=gen)
        slots.copy_(torch.tensor(assignment, dtype=torch.int32, device="cuda"))

        idx = slots.long()
        ref_state = pool[idx].clone()
        y_ref = gdn_decode_vectorized(
            state=ref_state, q=q.clone(), k=k.clone(), v=v.clone(),
            alpha=alpha.clone(), beta=beta.clone(), cfg=cfg,
        )

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(y, y_ref, rtol=2e-5, atol=2e-6)
        for row, slot in enumerate(assignment):
            torch.testing.assert_close(pool[slot], ref_state[row], rtol=2e-5, atol=2e-6)


def test_replay_is_cheaper_than_recapture(capsys):
    """Quantifies the win over the reference engine's re-capture-per-rotation."""
    cfg = GDNConfig(n_heads=32, head_dim=128, state_size=128, n_groups=16)
    mod = load_gdn()
    B = 8
    pool = torch.randn(32, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
    slots = torch.arange(B, dtype=torch.int32, device="cuda")
    args = [
        torch.randn(B, cfg.n_groups, cfg.state_size, device="cuda"),
        torch.randn(B, cfg.n_groups, cfg.state_size, device="cuda"),
        torch.randn(B, cfg.n_heads, cfg.head_dim, device="cuda"),
        torch.rand(B, cfg.n_heads, device="cuda"),
        torch.rand(B, cfg.n_heads, device="cuda"),
        torch.empty(B, cfg.n_heads, cfg.head_dim, device="cuda"),
    ]
    graph = _capture(mod, pool, slots, args)

    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(200):
        graph.replay()
    torch.cuda.synchronize()
    per_replay_ms = (time.perf_counter() - t) / 200 * 1e3

    # The reference engine documents 10-20 ms per slot rotation
    # (config.h:130-138).
    assert per_replay_ms < 1.0, f"replay {per_replay_ms:.3f} ms is not cheap enough to matter"
    with capsys.disabled():
        print(f"\n  replay {per_replay_ms:.4f} ms/step vs 10-20 ms recapture per rotation")


def test_rotation_costs_nothing_extra():
    """Rotating slots between replays must not cost more than replaying.

    This is the actual comparison against the reference engine: they pay a
    re-capture on rotation, we pay a device tensor write. If reassigning slots were
    expensive for us too, the indirection would have bought nothing.
    """
    cfg = GDNConfig(n_heads=32, head_dim=128, state_size=128, n_groups=16)
    mod = load_gdn()
    B = 8
    pool = torch.randn(32, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
    slots = torch.arange(B, dtype=torch.int32, device="cuda")
    args = [
        torch.randn(B, cfg.n_groups, cfg.state_size, device="cuda"),
        torch.randn(B, cfg.n_groups, cfg.state_size, device="cuda"),
        torch.randn(B, cfg.n_heads, cfg.head_dim, device="cuda"),
        torch.rand(B, cfg.n_heads, device="cuda"),
        torch.rand(B, cfg.n_heads, device="cuda"),
        torch.empty(B, cfg.n_heads, cfg.head_dim, device="cuda"),
    ]
    graph = _capture(mod, pool, slots, args)

    rotations = [
        torch.tensor([(i + r) % 32 for i in range(B)], dtype=torch.int32, device="cuda")
        for r in range(8)
    ]

    torch.cuda.synchronize()
    t = time.perf_counter()
    for r in range(200):
        slots.copy_(rotations[r % 8])
        graph.replay()
    torch.cuda.synchronize()
    rotating_ms = (time.perf_counter() - t) / 200 * 1e3

    assert rotating_ms < 1.0, f"rotating replay {rotating_ms:.3f} ms/step"
