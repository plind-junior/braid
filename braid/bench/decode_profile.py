"""Where the decode step actually goes. ROADMAP Phase 3 item 3's first task.

`docs/runbooks/decode-speed.md` records a gap it deliberately refused to
explain: at B=16 the step is 16.73 ms, the weight sweep is 5.58 ms and the
state plus KV traffic is ~1.3 ms. **Roughly 8 ms is unaccounted for**, and no
lever should be chosen from a gap that has never been profiled.

This is the attribution half of that profile. It is two tools, not one, because
they answer different questions and only one of them reports real time:

  `--mode attribute`   torch.profiler over graph replays. CUPTI kernel activity
                       is real device time, so the per-kernel sum is checkable
                       against the wall-clock step: if they disagree the
                       attribution is wrong and nothing below it is worth
                       reading. Reported as a check, not assumed.

  `--mode spin`        bare replays with no profiler attached, for `ncu` to
                       profile. ncu serialises and replays kernels, so its
                       wall-clock is not real time -- metrics come from here,
                       timing does not.

Both run with **graphs ON**, per the measurement contract: a graphs-off kernel
sum runs ~1.8x the real step and produces a systematically wrong lever list,
because the launch-latency classes it flags overlap away under the real graph.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from braid.bench.noise_floor import HostHealthSampler
from braid.model.engine import Engine
from braid.model.graph import GraphedDecoder
from braid.model.loader import load_checkpoint

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

# Coarse buckets. The point is to say which *class* of work owns the step, not
# to name kernels -- kernel names are cuBLAS build details and change under us.
CLASSES: list[tuple[str, re.Pattern]] = [
    ("gemm", re.compile(r"cutlass|gemm|gemv|s16816|sm\d+_xmma|nvjet|splitk|tensor_op")),
    ("gdn-kernel", re.compile(r"gdn_|delta_rule|causal_conv|conv1d")),
    ("attn-expand", re.compile(r"repeat|expand|CatArrayBatched|index_select|gather|scatter")),
    ("softmax", re.compile(r"softmax|masked_fill")),
    ("norm", re.compile(r"rms|norm|layer_norm|vectorized_layer")),
    ("elementwise", re.compile(r"elementwise|vectorized_elementwise|unrolled_elementwise|"
                               r"silu|sigmoid|mul|add|copy|fill|CUDAFunctor")),
    ("reduce", re.compile(r"reduce|Reduce|sum|mean|max")),
]


def classify(name: str) -> str:
    for label, pat in CLASSES:
        if pat.search(name):
            return label
    return "other"


def _seed(engine: Engine, cache, batch: int, prompt_len: int = 8) -> None:
    g = torch.Generator(device="cuda").manual_seed(11)
    for slot in range(cache.max_slots):
        cache.reset_slot(slot)
    for row in range(batch):
        ids = torch.randint(0, 1000, (1, prompt_len), generator=g, device="cuda")
        engine.forward(ids, cache.select([row]))


def _build(batch: int, max_len: int, ckpt):
    eng = Engine.from_checkpoint(ckpt, device="cuda", dtype=torch.bfloat16,
                                 use_kernels=True)
    cache = eng.allocate_cache(max_len, max_slots=batch)
    _seed(eng, cache, batch)
    snap = cache.snapshot()
    dec = GraphedDecoder(eng, cache, buckets=(batch,))
    cache.restore(snap)
    tokens = torch.full((batch, 1), 42, dtype=torch.long, device="cuda")
    slots = torch.arange(batch, device="cuda")
    return eng, cache, snap, dec, tokens, slots


def _wall_ms_per_step(step, steps: int, reset) -> float:
    """CUDA events around the WHOLE run. Never a sync per step."""
    reset()
    for _ in range(min(steps, 64)):      # warmup >1 s, discarded
        step()
    torch.cuda.synchronize()
    reset()
    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    a.record()
    for _ in range(steps):
        step()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / steps


def attribute(batch: int, steps: int, max_len: int, ckpt) -> dict:
    eng, cache, snap, dec, tokens, slots = _build(batch, max_len, ckpt)
    step = lambda: dec.step(tokens, slots)
    reset = lambda: cache.restore(snap)

    with HostHealthSampler() as health:
        wall = _wall_ms_per_step(step, steps, reset)
    health_report = str(health.report())

    reset()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()

    per_kernel: dict[str, list] = defaultdict(lambda: [0.0, 0])
    for ev in prof.key_averages():
        us = float(ev.self_device_time_total)
        if us <= 0:
            continue
        per_kernel[ev.key][0] += us
        per_kernel[ev.key][1] += int(ev.count)

    kernels = [{"name": k, "ms_per_step": v[0] / 1e3 / steps,
                "calls_per_step": v[1] / steps, "cls": classify(k)}
               for k, v in per_kernel.items()]
    kernels.sort(key=lambda k: -k["ms_per_step"])

    by_class: dict[str, list] = defaultdict(lambda: [0.0, 0.0])
    for k in kernels:
        by_class[k["cls"]][0] += k["ms_per_step"]
        by_class[k["cls"]][1] += k["calls_per_step"]

    total = sum(k["ms_per_step"] for k in kernels)
    del eng, cache, dec
    torch.cuda.empty_cache()
    return {
        "batch": batch, "steps": steps, "max_len": max_len,
        "wall_ms_per_step": wall, "kernel_ms_per_step": total,
        "coverage_pct": total / wall * 100 if wall else 0.0,
        "kernels_per_step": sum(k["calls_per_step"] for k in kernels),
        "by_class": {c: {"ms_per_step": v[0], "calls_per_step": v[1]}
                     for c, v in sorted(by_class.items(), key=lambda x: -x[1][0])},
        "top": kernels[:30],
        "health": health_report,
    }


def locate(batch: int, steps: int, max_len: int, ckpt) -> None:
    """Which aten op, at which shapes, launched the kernels `attribute` ranks.

    Runs **eager**, deliberately. Graphs-off timing is wrong by ~1.8x and must
    not be used for a lever list -- but graphs-off runs the *same kernels on the
    same shapes*, and only eager keeps the CPU-side op correlation that says
    `aten::mul` on `[16, 32, 128, 128]` rather than `MulFunctor<float>`. Time
    comes from `--mode attribute`; identity comes from here.
    """
    eng, cache, snap, _, tokens, slots = _build(batch, max_len, ckpt)
    view = cache.select(slots.tolist())
    for _ in range(8):
        eng.decode_step(tokens, view)
    torch.cuda.synchronize()
    cache.restore(snap)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(steps):
            eng.decode_step(tokens, view)
        torch.cuda.synchronize()

    rows = [(float(e.self_device_time_total) / 1e3 / steps, int(e.count) / steps,
             e.key, str(e.input_shapes)[:76])
            for e in prof.key_averages(group_by_input_shape=True)
            if float(e.self_device_time_total) > 0]
    rows.sort(key=lambda r: -r[0])
    total = sum(r[0] for r in rows)
    print(f"\n=== B={batch} EAGER op attribution "
          f"(identity only -- timing here is NOT the graphed step)")
    print(f"    device total {total:.3f} ms/step over {sum(r[1] for r in rows):.0f} ops")
    print(f"\n    {'ms/step':>8}{'calls':>7}  {'op':<34}shapes")
    for ms, n, key, shapes in rows[:26]:
        print(f"    {ms:>8.3f}{n:>7.0f}  {key[:32]:<34}{shapes}")


def spin(batch: int, steps: int, max_len: int, ckpt) -> None:
    """Replays and nothing else, so `ncu` profiles the step and not the setup."""
    _, cache, snap, dec, tokens, slots = _build(batch, max_len, ckpt)
    for _ in range(8):                   # ncu --launch-skip covers these
        dec.step(tokens, slots)
    torch.cuda.synchronize()
    cache.restore(snap)
    for _ in range(steps):
        dec.step(tokens, slots)
    torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("attribute", "locate", "spin"),
                   default="attribute")
    p.add_argument("--batches", type=int, nargs="+", default=[16])
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    ckpt = load_checkpoint(MODEL_DIR, device="cuda")

    if args.mode == "spin":
        spin(args.batches[0], args.steps, args.max_len, ckpt)
        return

    if args.mode == "locate":
        for b in args.batches:
            locate(b, args.steps, args.max_len, ckpt)
        return

    out = [attribute(b, args.steps, args.max_len, ckpt) for b in args.batches]
    if args.json:
        print(json.dumps(out))
        return

    for r in out:
        print(f"\n=== B={r['batch']}  wall {r['wall_ms_per_step']:.3f} ms/step  "
              f"| kernels {r['kernel_ms_per_step']:.3f} ms "
              f"({r['coverage_pct']:.1f}% covered, "
              f"{r['kernels_per_step']:.0f} launches/step)")
        print(f"    host health: {r['health']}")
        print(f"\n    {'class':<14}{'ms/step':>9}{'% wall':>8}{'calls':>8}")
        for c, v in r["by_class"].items():
            print(f"    {c:<14}{v['ms_per_step']:>9.3f}"
                  f"{v['ms_per_step'] / r['wall_ms_per_step'] * 100:>8.1f}"
                  f"{v['calls_per_step']:>8.0f}")
        print(f"\n    {'ms/step':>8}{'calls':>7}  kernel")
        for k in r["top"][:14]:
            print(f"    {k['ms_per_step']:>8.3f}{k['calls_per_step']:>7.0f}  {k['name']}")


if __name__ == "__main__":
    main()
