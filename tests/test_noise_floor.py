import pytest
import torch

from braid.bench.noise_floor import measure_noise_floor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_report_fields_and_ordering():
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    rep = measure_noise_floor(lambda: a @ a, reps=20, warmup=3)
    assert rep.reps == 20
    assert rep.p10_s <= rep.median_s <= rep.p90_s
    assert rep.cv_pct >= 0.0


def test_noise_floor_is_reported_not_assumed():
    """A stable workload on a healthy box should be under 15% spread.

    This is the gate the design doc requires before any speedup claim.
    """
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    rep = measure_noise_floor(lambda: a @ a, reps=30, warmup=5)
    assert rep.cv_pct < 15.0, f"noise floor {rep.cv_pct:.1f}% is too wide to measure against"
