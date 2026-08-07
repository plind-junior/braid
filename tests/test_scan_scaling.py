import pytest
import torch

from braid.bench.scan_scaling import hbm_stream_bandwidth, scan_scaling
from braid.config import GDNConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

BATCHES = [1, 2, 4, 8, 16, 32]


@pytest.fixture(scope="module")
def points():
    """COLD is the production-realistic residency and the one we gate on.

    A pool sized to `batch` alone keeps the whole working set inside the 96 MB
    L2 and reports 4300 GB/s — nearly 3x this card's HBM. Real decode streams
    30 layers of state between two visits to the same layer, so nothing is
    resident on reuse.
    """
    cfg = GDNConfig.qwen36_35b_a3b()
    return cfg, scan_scaling(cfg, batches=BATCHES, reps=30, cold=True)


def test_aggregate_throughput_grows_with_batch(points):
    """The whole thesis in one assertion.

    The reference engine's hybrid decode is FLAT in N because its scan is
    single-sequence. Ours must not be. At batch 8 the scan should deliver well over 2x the rows/sec
    of batch 1, since a batch-1 recurrent scan leaves the GPU almost idle --
    the grid is (n_heads) = 32 blocks on a 170-SM card, and B=1 measures 40%
    of HBM against 104% at B=8.
    """
    _, reports = points
    rps = {b: p.rows_per_s for b, p in reports.items()}
    assert rps[8] > 2.0 * rps[1], (
        f"batch 8 gives {rps[8] / rps[1]:.2f}x batch 1; the scan is not actually batching"
    )


def test_throughput_saturates_rather_than_collapsing(points):
    """Past saturation the curve must flatten, not fall off.

    Measured: aggregate rows/s peaks at B=8 and eases ~9% by B=32 as the scan
    becomes pure DRAM streaming. That is expected and fine — but a real
    collapse (an L2 thrash, a spill, an occupancy cliff) would look different
    and must fail. Note this assertion deliberately does NOT require rps to
    keep rising: it does not, and a test asserting so would be asserting a
    wish rather than the hardware.
    """
    _, reports = points
    rps = {b: p.rows_per_s for b, p in reports.items()}
    peak = max(rps.values())
    assert rps[32] > 0.80 * peak, (
        f"B=32 at {rps[32]:.0f} rows/s is {rps[32] / peak * 100:.0f}% of peak {peak:.0f} — "
        "that is a collapse, not saturation"
    )


def test_scan_reaches_the_bandwidth_roofline(points):
    """The condition that matters more than raw scaling.

    Beating a batch-1 baseline only says the baseline was bad. This says the
    kernel is at its ceiling. The ceiling is the measured HBM streaming rate
    (~1528 GB/s), not a per-batch copy: a copy of `batch` rows is itself
    residency-dependent and at small sizes is slower than the kernel, which is
    how an earlier version of this test reported the scan at "175% of copy".

    Once the scan is at the roofline it cannot be made faster — only smaller.
    That is what makes FP16 h_state (halving the bytes) the highest-value open
    question for the concurrency curve.
    """
    _, reports = points
    hbm = hbm_stream_bandwidth()
    for b in (4, 8, 16, 32):
        achieved = reports[b].achieved_gbs
        assert achieved > 0.80 * hbm, (
            f"B={b}: scan {achieved:.0f} GB/s vs HBM {hbm:.0f} GB/s "
            f"({achieved / hbm * 100:.0f}%) — more than 20% off the roofline"
        )


def test_l2_cliff_is_characterised_not_assumed(points):
    """One sequence's state may be L2-resident; two are not.

    Per-layer h_state is 2 MiB and the whole per-sequence footprint is 63.75
    MiB against a 96 MB L2. If per-row cost jumped discontinuously at B=2 the
    whole linear model would be wrong. This does not assert there is no cliff
    — it asserts the cliff is not catastrophic, and the recorded curve is the
    real deliverable.
    """
    _, reports = points
    per_row_1 = reports[1].report.median_s
    per_row_2 = reports[2].report.median_s / 2
    assert per_row_2 < 1.5 * per_row_1, (
        f"per-row cost jumped {per_row_2 / per_row_1:.2f}x from B=1 to B=2; "
        "the L2 cliff is worse than the throughput model assumes"
    )
