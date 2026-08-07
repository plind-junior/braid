import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_device_is_blackwell_consumer():
    assert torch.cuda.get_device_capability(0) == (12, 0)


def test_torch_built_with_sm120():
    assert "sm_120" in torch.cuda.get_arch_list()


def test_achievable_bandwidth_above_floor():
    """Guards against landing on a throttled or contended box.

    Every later measurement is invalid if this fails. The architecture doc
    pins measured achievable bandwidth at 1508 GB/s (84% of the 1792
    datasheet figure); 1200 is a floor, not a target.
    """
    n = int(1e9 // 2)  # 1 GB of fp16
    x = torch.empty(n, device="cuda", dtype=torch.float16)
    for _ in range(3):
        x.clone()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        x.clone()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 10
    gbs = 2 * n * 2 / dt / 1e9  # read + write
    print(f"\nachievable bandwidth: {gbs:.0f} GB/s")
    assert gbs > 1200, f"only {gbs:.0f} GB/s; expected >1200 on a healthy 5090"
