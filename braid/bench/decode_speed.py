"""Decode throughput: graphs on vs off, per batch. ROADMAP Phase 3 item 2's
last secondary gate (`graphs_on / graphs_off >= 1.3`).

**This is a speed claim, so the measurement contract applies.** Timing is taken
with CUDA events around a whole run of steps — never a sync per step, which for
work in the tens of microseconds measures the host round-trip instead of the
GPU. Clocks are sampled *concurrently*, and each timed run is kept busy >1 s so
the sampler returns a verdict rather than a shrug.

**One trap this bench must not fall into.** The CUDA kernels validate their
`slot_idx` range with a device->host copy on every call, and skip it while a
graph is capturing (`cudaStreamIsCapturing`). That is 48 host syncs per eager
decode step and zero per replay. A ratio taken between eager-with-kernels and
graphed-with-kernels would therefore be measuring *validation*, not graphs. So
three arms are reported, not two:

    eager, torch scan      no validation syncs, but gathers/scatters the state
    eager, CUDA kernels    no gather/scatter, but 48 validation syncs per step
    graphed, CUDA kernels  neither

The honest `graphs_on / graphs_off` is the third against the *better* of the
first two — taking the worse one would flatter the graph.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from braid.bench.noise_floor import HostHealthSampler
from braid.model.engine import Engine
from braid.model.graph import GraphedDecoder
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
PROMPT_LEN = 8


@dataclass
class Arm:
    name: str
    batch: int
    ms_per_step: float
    tok_per_s: float
    steps: int


def _seed(engine: Engine, cache, batch: int, prompt_len: int = PROMPT_LEN) -> None:
    g = torch.Generator(device="cuda").manual_seed(11)
    for slot in range(cache.max_slots):
        cache.reset_slot(slot)
    for row in range(batch):
        ids = torch.randint(0, 1000, (1, prompt_len), generator=g, device="cuda")
        engine.forward(ids, cache.select([row]))


def _time(fn, steps: int, reset) -> float:
    """Seconds per step. Events around the WHOLE run; no per-step sync."""
    # Warmup long enough to ramp the clocks (>1 s), discarded.
    reset()
    for _ in range(min(steps, 64)):
        fn()
    torch.cuda.synchronize()

    reset()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1e3 / steps


def measure(batch: int, steps: int, max_len: int, ckpt,
            quant_mlp: bool = False) -> list[Arm]:
    arms: list[Arm] = []
    tokens = torch.full((batch, 1), 42, dtype=torch.long, device="cuda")
    slots = torch.arange(batch, device="cuda")

    for name, use_kernels in (("eager-torch", False), ("eager-kernels", True)):
        eng = Engine.from_checkpoint(ckpt, device="cuda", dtype=torch.bfloat16,
                                     use_kernels=use_kernels, quant_mlp=quant_mlp)
        cache = eng.allocate_cache(max_len, max_slots=batch)
        view = cache.select(list(range(batch)))
        _seed(eng, cache, batch)
        snap = cache.snapshot()
        s = _time(lambda: eng.decode_step(tokens, view), steps,
                  lambda: cache.restore(snap))
        arms.append(Arm(name, batch, s * 1e3, batch / s, steps))
        del eng, cache, view, snap

    eng = Engine.from_checkpoint(ckpt, device="cuda", dtype=torch.bfloat16,
                                 use_kernels=True, quant_mlp=quant_mlp)
    cache = eng.allocate_cache(max_len, max_slots=batch)
    _seed(eng, cache, batch)
    snap = cache.snapshot()
    dec = GraphedDecoder(eng, cache, buckets=(batch,))
    cache.restore(snap)
    s = _time(lambda: dec.step(tokens, slots), steps, lambda: cache.restore(snap))
    arms.append(Arm("graphed-kernels", batch, s * 1e3, batch / s, steps))

    # Same graphs, but selecting the KV bucket from the live length the way a
    # scheduler would -- it tracks sequence lengths on the host already, so the
    # choice costs nothing. Lengths grow across the run (prompt + step index),
    # so this crosses bucket boundaries exactly as serving does; it is not a
    # fixed short kv_len chosen to flatter the number.
    n = [0]

    def kv_step():
        n[0] += 1
        return dec.step(tokens, slots, live_len=PROMPT_LEN + n[0])

    def kv_reset():
        n[0] = 0
        cache.restore(snap)

    s = _time(kv_step, steps, kv_reset)
    arms.append(Arm("graphed-kvbucket", batch, s * 1e3, batch / s, steps))
    del eng, cache, dec
    torch.cuda.empty_cache()
    return arms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quant-mlp", action="store_true",
                   help="FP8 W8A8 on the MLP projections")
    args = p.parse_args()

    ckpt = load_checkpoint(MODEL_DIR, device="cuda")
    out: list[Arm] = []
    with HostHealthSampler() as health:
        for b in args.batches:
            out.extend(measure(b, args.steps, args.max_len, ckpt,
                               quant_mlp=args.quant_mlp))
    report = health.report()

    if args.json:
        print(json.dumps({"arms": [asdict(a) for a in out],
                          "health": str(report)}))
        return

    print(f"\nhost health: {report}\n")
    print(f"{'batch':>5} {'arm':<18} {'ms/step':>9} {'tok/s':>10}")
    for a in out:
        print(f"{a.batch:>5} {a.name:<18} {a.ms_per_step:>9.3f} {a.tok_per_s:>10.1f}")

    print(f"\n{'batch':>5} {'graphs_on/off':>14}  (vs the better eager arm)")
    for b in args.batches:
        rows = {a.name: a for a in out if a.batch == b}
        best_eager = min(rows["eager-torch"].ms_per_step,
                         rows["eager-kernels"].ms_per_step)
        ratio = best_eager / rows["graphed-kernels"].ms_per_step
        verdict = "PASS" if ratio >= 1.3 else "below 1.3"
        print(f"{b:>5} {ratio:>14.2f}  {verdict}")


if __name__ == "__main__":
    main()
