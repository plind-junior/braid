"""Serving benchmark: aggregate tok/s, per-stream ITL, TTFT. Phase 4 item 2.

Drives `Scheduler` **in process**. Not over HTTP, deliberately: the roadmap's
own fairness notes record a threaded Python client inflating c=64 TTFT by 19x,
and a loopback socket plus the GIL would be measuring the client. The HTTP layer
is tested for correctness in `tests/test_server.py`; it is not what a throughput
claim should be taken through.

**Aggregate and ITL are not two results.** Aggregate tok/s is exactly
`concurrency / mean_ITL`, so quoting both as independent wins is how a 6x error
gets made. They are reported together because they answer different questions —
how much work the box does, and what one user feels — not because they are
independent evidence.

Timing here is wall-clock on the host, not CUDA events, and that is correct for
this metric: TTFT and ITL are what a client experiences, including the sampler's
sync and the scheduler's Python. The kernel-level contract (events around a
whole run, never a sync per step) governs `decode_speed.py`, which measures the
step. This measures the service.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from braid.bench.noise_floor import HostHealthSampler
from braid.model.engine import Engine
from braid.model.loader import load_checkpoint
from braid.serve.scheduler import Request, Scheduler

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))


@dataclass
class Point:
    concurrency: int
    n_requests: int
    tokens: int
    wall_s: float
    aggregate_tok_s: float
    ttft_ms_p50: float
    ttft_ms_p90: float
    itl_ms_p50: float
    itl_ms_p90: float
    itl_ms_p99: float
    peak_vram_gb: float
    # Where the wall clock actually went. Prefill runs the GDN recurrence in a
    # Python loop over T, so this split is the difference between "the engine is
    # slow" and "the engine decodes fast and prefills slowly".
    prefill_s: float = 0.0
    decode_s: float = 0.0
    prefill_tokens: int = 0
    health: str = ""
    # `aggregate = concurrency / mean_itl` up to prefill and ragged tails. It is
    # printed so a reader can check the two numbers against each other rather
    # than take them as independent.
    itl_implied_tok_s: float = 0.0


@dataclass
class Report:
    points: list[Point] = field(default_factory=list)
    env: dict = field(default_factory=dict)


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def _env() -> dict:
    """Every SHA/version field a published number has to carry."""
    def run(*argv):
        # Argument list, never `shell=True`: these are fixed commands, but a
        # benchmark that shells out is a benchmark someone will later
        # parameterise.
        try:
            return subprocess.check_output(argv, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {
        # The remote box gets an rsync of the tree, not the .git dir, so the
        # commit is passed in rather than discovered. A published number without
        # its commit is a number nobody can recheck.
        "commit": os.environ.get("BRAID_COMMIT")
                  or run("git", "-C", str(Path(__file__).resolve().parents[2]),
                         "rev-parse", "--short", "HEAD"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": run("nvidia-smi", "--query-gpu=driver_version",
                      "--format=csv,noheader"),
        "gpu": torch.cuda.get_device_name(0),
    }


def _prompts(n: int, length: int, seed: int) -> list[list[int]]:
    """Fresh, non-repeating prompts.

    The roadmap makes this a fairness condition: the reference engine has prefix
    caching worth 3.5x on multi-turn TTFT and braid has none, so repeated
    prompts would flatter them and penalise us. Distinct random ids also defeat
    any accidental caching on either side.
    """
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 100_000, (length,), generator=g).tolist()
            for _ in range(n)]


def measure(engine: Engine, concurrency: int, n_requests: int, prompt_len: int,
            max_new_tokens: int, max_len: int, graphed: bool, seed: int,
            prefill_chunk: int = 256, prefill_budget: int | None = None) -> Point:
    sched = Scheduler(engine, capacity=concurrency, max_len=max_len,
                      graphed=graphed, prefill_chunk=prefill_chunk,
                      prefill_budget=prefill_budget)

    # Warmup: clocks need ~1 s to ramp, and the first replay of every bucket
    # allocates. Discarded.
    warm = [Request(prompt=p, max_new_tokens=8)
            for p in _prompts(concurrency, prompt_len, seed + 999)]
    sched.run(warm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sched.reset_stats()

    reqs = [Request(prompt=p, max_new_tokens=max_new_tokens)
            for p in _prompts(n_requests, prompt_len, seed)]
    submitted: dict[str, float] = {}
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    itls: list[float] = []
    total = 0

    with HostHealthSampler() as health:
        t0 = time.perf_counter()
        for r in reqs:
            sched.submit(r)
            submitted[r.id] = t0
        while not sched.idle:
            for up in sched.step():
                now = time.perf_counter()
                if not up.tokens:
                    continue
                total += len(up.tokens)
                if up.id not in first:
                    first[up.id] = now
                else:
                    # One tick can emit one token per stream, so a gap is an
                    # inter-token latency for that stream.
                    itls.append((now - last[up.id]) * 1e3)
                last[up.id] = now
        wall = time.perf_counter() - t0
    report = str(health.report())

    ttfts = [(first[i] - submitted[i]) * 1e3 for i in first]
    mean_itl = statistics.mean(itls) if itls else float("nan")
    return Point(
        concurrency=concurrency, n_requests=n_requests, tokens=total,
        wall_s=wall, aggregate_tok_s=total / wall,
        ttft_ms_p50=_pct(ttfts, 0.50), ttft_ms_p90=_pct(ttfts, 0.90),
        itl_ms_p50=_pct(itls, 0.50), itl_ms_p90=_pct(itls, 0.90),
        itl_ms_p99=_pct(itls, 0.99),
        peak_vram_gb=torch.cuda.max_memory_reserved() / 1e9,
        prefill_s=sched.prefill_s, decode_s=sched.decode_s,
        prefill_tokens=sched.prefill_tokens,
        health=report,
        itl_implied_tok_s=concurrency / (mean_itl / 1e3) if itls else float("nan"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--requests-per-stream", type=int, default=2,
                   help="requests submitted per concurrent slot")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--quant-mlp", action="store_true")
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--prefill-chunk", type=int, default=256,
                   help="per-row prompt tokens per tick; bounds the scan loop's T")
    p.add_argument("--prefill-budget", type=int, default=None,
                   help="total prompt tokens per tick across rows; bounds B*T. "
                        "Setting it to the prompt length forces one row per "
                        "forward, which is the pre-batching baseline.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    ck = load_checkpoint(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16,
                                 use_kernels=True, quant_mlp=args.quant_mlp)

    rep = Report(env=_env())
    for c in args.concurrency:
        rep.points.append(measure(
            eng, concurrency=c, n_requests=c * args.requests_per_stream,
            prompt_len=args.prompt_len, max_new_tokens=args.max_new_tokens,
            max_len=args.max_len, graphed=not args.no_graphs, seed=args.seed + c,
            prefill_chunk=args.prefill_chunk, prefill_budget=args.prefill_budget))

    if args.json:
        print(json.dumps({"points": [asdict(x) for x in rep.points],
                          "env": rep.env}))
        return

    print(f"\nenv: {rep.env}")
    print(f"prompt {args.prompt_len} tok, {args.max_new_tokens} new, "
          f"{args.requests_per_stream} requests per slot, "
          f"graphs {'off' if args.no_graphs else 'on'}, "
          f"MLP {'fp8' if args.quant_mlp else 'bf16'}, "
          f"prefill chunk {args.prefill_chunk} budget "
          f"{args.prefill_budget or 'capacity*chunk'}\n")
    print(f"{'c':>3}{'tok/s':>9}{'TTFT p50':>10}{'p90':>8}"
          f"{'ITL p50':>9}{'p90':>8}{'p99':>8}{'VRAM GB':>9}{'c/ITL':>8}"
          f"{'prefill%':>9}{'pf tok/s':>10}")
    for x in rep.points:
        share = x.prefill_s / x.wall_s * 100 if x.wall_s else 0.0
        pf = x.prefill_tokens / x.prefill_s if x.prefill_s else float("nan")
        print(f"{x.concurrency:>3}{x.aggregate_tok_s:>9.1f}{x.ttft_ms_p50:>10.1f}"
              f"{x.ttft_ms_p90:>8.1f}{x.itl_ms_p50:>9.2f}{x.itl_ms_p90:>8.2f}"
              f"{x.itl_ms_p99:>8.2f}{x.peak_vram_gb:>9.2f}"
              f"{x.itl_implied_tok_s:>8.1f}{share:>8.0f}%{pf:>10.0f}")
    print("\nhost health per point:")
    for x in rep.points:
        print(f"  c={x.concurrency:<3} {x.health}")
    print("\n`c/ITL` is aggregate implied by the mean ITL. It is printed because "
          "aggregate\nand ITL are the same measurement seen twice, not two results.")


if __name__ == "__main__":
    main()
