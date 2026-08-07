"""Measure the variance floor of a GPU workload.

The design doc requires the significance threshold to be measured rather than
chosen. A 2% gate against a 10% noise floor is a coin flip.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class NoiseReport:
    median_s: float
    cv_pct: float
    p10_s: float
    p90_s: float
    reps: int

    def __str__(self) -> str:
        return (
            f"median {self.median_s * 1e3:.3f} ms | "
            f"p10-p90 {self.p10_s * 1e3:.3f}-{self.p90_s * 1e3:.3f} ms | "
            f"noise floor {self.cv_pct:.2f}% over {self.reps} reps"
        )


def measure_noise_floor(fn: Callable[[], object], reps: int = 30, warmup: int = 5) -> NoiseReport:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)

    samples.sort()
    median = samples[len(samples) // 2]
    p10 = samples[max(0, int(0.10 * len(samples)))]
    p90 = samples[min(len(samples) - 1, int(0.90 * len(samples)))]
    cv = (p90 - p10) / median * 100.0 if median > 0 else 0.0
    return NoiseReport(median_s=median, cv_pct=cv, p10_s=p10, p90_s=p90, reps=reps)


def measure_graphed(fn: Callable[[], object], inner: int = 64, reps: int = 30,
                    warmup: int = 10) -> NoiseReport:
    """Time a kernel by replaying a graph of `inner` back-to-back launches.

    `measure_noise_floor` syncs on every rep, so for a kernel in the tens of
    microseconds it measures the host round-trip, not the kernel. The tell is
    unmistakable and we hit it: per-step time barely moved from B=1 to B=8,
    the "bandwidth" of a plain copy came out above the HBM ceiling, and the
    host-health sampler reported the GPU *idle* at 360 MHz during its own
    benchmark. That produces a flattering scaling curve made entirely of
    amortized launch overhead.

    Capturing `inner` launches into one graph removes per-launch CPU cost
    entirely, which is also what braid's decode step will actually do. The
    reference engine's own rule: always A/B with graphs ON, because a
    graphs-OFF kernel-time sum
    runs ~1.8x the real step and yields a systematically wrong lever list
    (known-issues.md:97).

    Returns seconds per single call.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1e3 / inner)

    samples.sort()
    median = samples[len(samples) // 2]
    p10 = samples[max(0, int(0.10 * len(samples)))]
    p90 = samples[min(len(samples) - 1, int(0.90 * len(samples)))]
    cv = (p90 - p10) / median * 100.0 if median > 0 else 0.0
    return NoiseReport(median_s=median, cv_pct=cv, p10_s=p10, p90_s=p90, reps=reps)


@dataclass(frozen=True)
class HealthReport:
    sm_mhz: float
    mem_mhz: float
    power_w: float
    samples: int
    depressed: bool
    reason: str

    def __str__(self) -> str:
        verdict = f"DEPRESSED ({self.reason})" if self.depressed else "healthy"
        return (
            f"{self.sm_mhz:.0f} MHz SM / {self.mem_mhz:.0f} MHz mem / "
            f"{self.power_w:.0f} W peak over {self.samples} samples — {verdict}"
        )


class HostHealthSampler:
    """Background 1 Hz clock/power sampler, per the reference engine's
    measurement contract.

    Sampled *concurrently with the timed work* — an idle sample reads ~180 MHz
    and 17 W and says nothing about the run. The first two samples are dropped
    because the card is still ramping. Depressed if median mem < 13801 MHz, or
    median SM < 2000 MHz, or peak power < 400 W.

    They sample at 1 Hz, but their bench runs for tens of seconds. Ours run for
    ~1 s, so 1 Hz minus a 2-sample ramp yields nothing at all. 0.1 s keeps the
    same 2-sample ramp discipline while actually producing a verdict; callers
    must still give the sampler >1 s of busy time to measure.
    """

    _QUERY = "clocks.sm,clocks.mem,power.draw"

    def __init__(self, interval_s: float = 0.1) -> None:
        self._interval = interval_s
        self._samples: list[tuple[float, float, float]] = []
        self._stop = None
        self._thread = None

    def _run(self) -> None:
        import subprocess

        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self._QUERY}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=5,
                ).stdout.strip().split("\n")[0]
                self._samples.append(tuple(float(x) for x in out.split(",")))
            except Exception:  # a failed sample must not kill the measurement
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> "HostHealthSampler":
        import threading

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def report(self) -> HealthReport:
        import statistics

        usable = self._samples[2:]  # drop the ramp
        if not usable:
            return HealthReport(0, 0, 0, 0, True, "no samples")
        sm = statistics.median(s[0] for s in usable)
        mem = statistics.median(s[1] for s in usable)
        pwr = max(s[2] for s in usable)
        reasons = []
        if mem < 13801:
            reasons.append(f"mem {mem:.0f} < 13801 MHz")
        if sm < 2000:
            reasons.append(f"sm {sm:.0f} < 2000 MHz")
        if pwr < 400:
            reasons.append(f"power {pwr:.0f} < 400 W")
        return HealthReport(sm, mem, pwr, len(usable), bool(reasons), "; ".join(reasons))


def main() -> None:
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    x = torch.empty(int(1e9 // 2), device="cuda", dtype=torch.float16)

    # rep counts chosen so each workload stays busy >1 s, which is both the
    # reference engine's stated minimum and what the health sampler needs to
    # return a verdict.
    with HostHealthSampler() as health:
        mm = measure_noise_floor(lambda: a @ a, reps=250, warmup=10)
        clone = measure_noise_floor(lambda: x.clone(), reps=800, warmup=10)

    print("fp16 8192 matmul :", mm)
    print("1GB fp16 clone   :", clone)
    gbs = 2 * x.numel() * 2 / clone.median_s / 1e9
    print(f"clone bandwidth  : {gbs:.0f} GB/s")
    print("host             :", health.report())


if __name__ == "__main__":
    main()
