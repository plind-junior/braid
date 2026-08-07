"""Measure how the batched GDN scan scales with concurrency.

This is the Phase 1 evidence. Two things must be true, and the second matters
more than the first:

1. Aggregate rows/second grows with batch size. The reference engine's
   equivalent is flat by construction -- its scan takes one `float* h_state`
   and launches
   <<<n_heads>>>, so the scheduler clamps hybrid decode to one sequence per
   step (engine_scheduler.cpp:1475).

2. The kernel's achieved DRAM bandwidth is close to a raw copy of the same
   bytes. Condition 1 alone only says we beat a bad baseline; condition 2 says
   whether the scan is at its ceiling, and it sets the per-sequence linear
   term that caps the concurrency curve around B=14-18.

HOT vs COLD, and why both are reported
--------------------------------------
A naive harness allocates a pool of exactly `batch` state slabs. At 2 MiB per
row that is 16 MiB at B=8 and 64 MiB at B=32 -- entirely inside the 5090's
96 MB L2. The scan then reports 4000+ GB/s, well above the 1514 GB/s HBM
ceiling this box actually has, because it never touches DRAM.

Production does not look like that. One sequence carries 30 GDN layers of
state, so between two visits to layer L the engine streams 30 x B x 2 MiB of
*other* layers' state through the cache -- 480 MiB at B=8. Layer L's slab is
long gone by the time the next token arrives.

COLD reproduces that by giving each of the `inner` captured launches its own
slot assignment over a pool sized past L2, so nothing is resident on reuse.
COLD is the number the throughput model should use. HOT is reported next to
it because the gap between them IS the L2 effect, quantified.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from braid.bench.noise_floor import HostHealthSampler, NoiseReport, measure_graphed
from braid.config import GDNConfig
from braid.kernels.loader import load_gdn

# Pool budget for the cold measurement. Must comfortably exceed the 96 MB L2;
# 2 GiB also leaves room for the copy baseline's two buffers on a 32 GB card.
_COLD_POOL_BYTES = 2 << 30


@dataclass(frozen=True)
class ScalingPoint:
    batch: int
    report: NoiseReport
    state_bytes: int

    @property
    def rows_per_s(self) -> float:
        return self.batch / self.report.median_s

    @property
    def achieved_gbs(self) -> float:
        """State traffic only: read once, written once, per row."""
        return self.state_bytes / self.report.median_s / 1e9


def _row_bytes(cfg: GDNConfig) -> int:
    return cfg.n_heads * cfg.state_size * cfg.head_dim * 4


def _state_bytes(cfg: GDNConfig, batch: int) -> int:
    return 2 * _row_bytes(cfg) * batch  # read + write


def _slot_plan(cfg: GDNConfig, batch: int, inner: int, cold: bool):
    """Pool size and per-launch slot assignments.

    Cold: every launch works on a different stripe of a pool sized past L2, so
    a slab is never resident when it is revisited. Hot: one small pool reused
    by every launch.
    """
    if not cold:
        return batch, [torch.arange(batch, dtype=torch.int32, device="cuda")] * inner

    max_rows = max(batch, _COLD_POOL_BYTES // _row_bytes(cfg))
    stripes = max(1, min(inner, max_rows // batch))
    n_slots = stripes * batch
    slots = [
        torch.arange(i * batch, (i + 1) * batch, dtype=torch.int32, device="cuda")
        for i in range(stripes)
    ]
    return n_slots, [slots[i % stripes] for i in range(inner)]


def raw_copy_bandwidth(cfg: GDNConfig, batch: int, reps: int = 30, inner: int = 64,
                       cold: bool = True) -> float:
    """Ceiling for the scan: a plain copy of exactly the same state bytes.

    Measured the same graphed, same-residency way as the scan itself. It has
    to be: a sync-per-rep copy of 4 MiB reports ~440 GB/s (host-bound) and a
    sync-per-rep copy of 64 MiB reports ~3600 GB/s (L2-resident, above the HBM
    ceiling). Neither is a bandwidth, and a scan measured one way against a
    ceiling measured another is an arbitrary ratio.
    """
    n_slots, _ = _slot_plan(cfg, batch, inner, cold)
    stripes = max(1, n_slots // batch)
    src = torch.randn(n_slots, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
    dst = torch.empty_like(src)

    counter = {"i": 0}

    def step() -> None:
        # CONTIGUOUS slice copy, not advanced indexing. `dst[idx] = src[idx]`
        # with an int64 index tensor is a gather+scatter and is SLOWER than
        # the scan kernel -- it reported the scan at 118-175% "of copy",
        # which is a broken baseline, not superluminal bandwidth.
        i = counter["i"] % stripes
        counter["i"] += 1
        dst[i * batch:(i + 1) * batch].copy_(src[i * batch:(i + 1) * batch])

    rep = measure_graphed(step, inner=inner, reps=reps)
    return _state_bytes(cfg, batch) / rep.median_s / 1e9


def hbm_stream_bandwidth(reps: int = 30, inner: int = 16) -> float:
    """Measured achievable HBM streaming rate — the true roofline.

    The per-batch copy baseline is size-dependent and therefore residency-
    dependent. This is a single size-independent ceiling: a 1 GB contiguous
    clone, the same measurement `tests/test_env.py` gates the box on.
    """
    n = int(1e9 // 4)
    src = torch.randn(n, device="cuda")
    dst = torch.empty_like(src)
    rep = measure_graphed(lambda: dst.copy_(src), inner=inner, reps=reps)
    return 2 * n * 4 / rep.median_s / 1e9


def scan_scaling(cfg: GDNConfig, batches: list[int], reps: int = 30, inner: int = 64,
                 cold: bool = True) -> dict[int, ScalingPoint]:
    mod = load_gdn()
    out: dict[int, ScalingPoint] = {}
    for b in batches:
        n_slots, plan = _slot_plan(cfg, b, inner, cold)
        pool = torch.randn(n_slots, cfg.n_heads, cfg.state_size, cfg.head_dim, device="cuda")
        q = torch.randn(b, cfg.n_groups, cfg.state_size, device="cuda")
        k = torch.randn(b, cfg.n_groups, cfg.state_size, device="cuda")
        v = torch.randn(b, cfg.n_heads, cfg.head_dim, device="cuda")
        alpha = torch.rand(b, cfg.n_heads, device="cuda") * 0.5 + 0.5
        beta = torch.rand(b, cfg.n_heads, device="cuda")
        y = torch.empty(b, cfg.n_heads, cfg.head_dim, device="cuda")

        counter = {"i": 0}

        def step() -> None:
            s = plan[counter["i"] % len(plan)]
            counter["i"] += 1
            mod.gdn_decode(pool, s, q, k, v, alpha, beta, y)

        out[b] = ScalingPoint(b, measure_graphed(step, inner=inner, reps=reps),
                              _state_bytes(cfg, b))
        del pool
        torch.cuda.empty_cache()
    return out


def _table(cfg: GDNConfig, batches: list[int], cold: bool, hbm: float) -> None:
    points = scan_scaling(cfg, batches, cold=cold)
    copies = {b: raw_copy_bandwidth(cfg, b, cold=cold) for b in batches}
    base = points[1].rows_per_s
    label = "COLD (pool past L2, production-realistic)" if cold else "HOT (L2-resident)"
    print(f"\n=== {label} ===")
    print(f"{'batch':>6} {'us/step':>9} {'rows/s':>12} {'vs b=1':>8} "
          f"{'GB/s':>8} {'copy':>8} {'%HBM':>7} {'noise':>7}")
    for b in batches:
        p = points[b]
        print(f"{b:>6} {p.report.median_s * 1e6:>9.2f} {p.rows_per_s:>12.0f} "
              f"{p.rows_per_s / base:>7.2f}x {p.achieved_gbs:>8.0f} {copies[b]:>8.0f} "
              f"{p.achieved_gbs / hbm * 100:>6.0f}% {p.report.cv_pct:>6.1f}%")


def main() -> None:
    cfg = GDNConfig.qwen36_35b_a3b()
    batches = [1, 2, 4, 8, 16, 32]

    with HostHealthSampler() as health:
        hbm = hbm_stream_bandwidth()
        print(f"measured HBM streaming ceiling: {hbm:.0f} GB/s")
        _table(cfg, batches, cold=True, hbm=hbm)
        _table(cfg, batches, cold=False, hbm=hbm)

    print()
    print(f"per-row state per GDN layer : {_row_bytes(cfg) / 2**20:.0f} MiB")
    print(f"per-sequence, all {cfg.n_gdn_layers} layers: "
          f"{cfg.recurrent_bytes_per_seq / 2**20:.1f} MiB")
    print(f"c=8  recurrent footprint    : {cfg.recurrent_bytes_per_seq * 8 / 2**20:.0f} MiB")
    print(f"c=16 recurrent footprint    : {cfg.recurrent_bytes_per_seq * 16 / 2**20:.0f} MiB")
    print("L2 = 96 MB; a decode step streams "
          f"{cfg.recurrent_bytes_per_seq * 8 * 2 / 2**20:.0f} MiB of state at B=8 "
          "(read+write), so layer L is never resident on revisit")
    print("host:", health.report())


if __name__ == "__main__":
    main()
