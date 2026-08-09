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
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from braid.bench.noise_floor import HostHealthSampler
from braid.model.engine import Engine
from braid.model.graph import GraphedDecoder
from braid.model.loader import load_checkpoint
from braid.model.quant import GROUPS, parse_groups

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

# How much KV each row sits on before the timed steps begin. **This is a
# comparison-critical parameter, not a detail.** Decode attention reads the whole
# live KV every step, so a run seeded with 8 tokens is measuring a cheaper step
# than one seeded with 128 — and `llama-batched-bench -npp 128 -ntg 128`, which
# is braid's published denominator, decodes at KV 128..256. Timing braid at 8
# against that is a shape mismatch that flatters braid. The default stays 8 for
# continuity with the graphs-on/off gate this module was written for; the
# head-to-head passes `--prompt-len 128`.
PROMPT_LEN = 8


@dataclass
class Arm:
    name: str
    batch: int
    ms_per_step: float
    tok_per_s: float
    steps: int
    peak_gib: float = 0.0


def _seed(engine: Engine, cache, batch: int, prompt_len: int = PROMPT_LEN) -> None:
    """Put `prompt_len` tokens of KV under every row before timing starts.

    One ragged batched forward rather than `batch` sequential ones. The rows are
    a true rectangle here (same length, all fresh), so this is the `seq_lens=None`
    path; it is seeding, not the measurement, and is untimed either way.
    """
    g = torch.Generator(device="cuda").manual_seed(11)
    for slot in range(cache.max_slots):
        cache.reset_slot(slot)
    ids = torch.randint(0, 1000, (batch, prompt_len), generator=g, device="cuda")
    engine.forward(ids, cache.select(list(range(batch))))


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


STATE_DTYPES = {"fp32": torch.float32, "fp16": torch.float16,
                "bf16": torch.bfloat16}

# Bench-process levers, applied per engine build in `_engine`. Module state
# rather than parameters because `measure` builds engines at three sites and
# threading two booleans through all of them obscures the actual measurement.
_TUNE = {"fused_quant": True, "raw_gates": True}


def _engine(use_kernels: bool, quant, state_dtype=torch.float32) -> Engine:
    """A fresh engine, with the source checkpoint dropped immediately.

    **Loading once outside the loop and reusing the checkpoint is what this
    avoids, and it was not free.** `from_checkpoint` aliases the checkpoint's
    tensors when it leaves a weight alone, but *quantizing* allocates a new fp8
    copy while the bf16 original stays reachable through the checkpoint. At
    Qwen3.5-9B that is 16.7 GiB of bf16 held beside the engine that no longer
    needs it, and it OOMs the B=64 arm on a 31 GiB card — the quantized arm,
    the one whose whole purpose is to need less memory. Reloading costs ~2 s
    from page cache.
    """
    ck = load_checkpoint(MODEL_DIR, device="cuda")
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16,
                                 use_kernels=use_kernels, quant=quant,
                                 state_dtype=state_dtype)
    del ck
    torch.cuda.empty_cache()
    if not _TUNE["fused_quant"]:
        from braid.model.quant import warm_fused
        warm_fused(False)
    if not _TUNE["raw_gates"]:
        eng.set_raw_gates(False)
    return eng


def measure(batch: int, steps: int, max_len: int, quant=None,
            prompt_len: int = PROMPT_LEN,
            state_dtype: torch.dtype = torch.float32) -> list[Arm]:
    arms: list[Arm] = []
    tokens = torch.full((batch, 1), 42, dtype=torch.long, device="cuda")
    slots = torch.arange(batch, device="cuda")

    for name, use_kernels in (("eager-torch", False), ("eager-kernels", True)):
        eng = _engine(use_kernels, quant, state_dtype)
        cache = eng.allocate_cache(max_len, max_slots=batch)
        view = cache.select(list(range(batch)))
        _seed(eng, cache, batch, prompt_len)
        snap = cache.snapshot()
        s = _time(lambda: eng.decode_step(tokens, view), steps,
                  lambda: cache.restore(snap))
        arms.append(Arm(name, batch, s * 1e3, batch / s, steps))
        del eng, cache, view, snap

    eng = _engine(True, quant, state_dtype)
    # Peak is measured from **here**, not from process start, and the difference
    # is not cosmetic. Building a quantized engine holds the bf16 originals and
    # the fp8 copies at the same moment, so the construction transient is larger
    # for fp8 than for bf16 — 24.6 GiB against 16.9 at B=1 on Qwen3.5-9B. Left
    # in, the published VRAM column would say fp8 costs *more* memory to serve,
    # which is the opposite of true: `reset_peak_memory_stats` rebases the peak
    # to what is resident now, so what follows measures weights + caches + graph
    # pool + activations, i.e. what serving actually needs. The construction
    # transient is a real load-time limit and is reported separately, not folded
    # into this number.
    torch.cuda.reset_peak_memory_stats()
    cache = eng.allocate_cache(max_len, max_slots=batch)
    _seed(eng, cache, batch, prompt_len)
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
        return dec.step(tokens, slots, live_len=prompt_len + n[0])

    def kv_reset():
        n[0] = 0
        cache.restore(snap)

    s = _time(kv_step, steps, kv_reset)
    arms.append(Arm("graphed-kvbucket", batch, s * 1e3, batch / s, steps))

    # Own-peak VRAM, which the roadmap's bench harness is required to publish and
    # which is the field that says whether a batch is near the card. Read after
    # the graph pool exists, so it includes it. Attributed to every arm at this
    # batch: the graphed arm is the one a server would run, and the eager arms
    # differ only in the cache they hold.
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    for a in arms:
        a.peak_gib = peak
    del eng, cache, dec
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return arms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quant", default="",
                   help=f"FP8 W8A8 groups, comma separated: {','.join(GROUPS)}, "
                        f"or 'all'")
    p.add_argument("--quant-mlp", action="store_true",
                   help="shorthand for --quant mlp")
    p.add_argument("--state-dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="storage type for the recurrent state pool; the scan is "
                        "fp32 either way")
    p.add_argument("--prompt-len", type=int, default=PROMPT_LEN,
                   help="KV under each row before timing; match the competitor's "
                        "shape when producing a head-to-head number")
    # Launch-count levers, off-switchable for attribution runs. Both are
    # asserted bit-identical to their fallbacks, so an arm differs only in
    # launch count — never in what it computes.
    p.add_argument("--no-fused-quant", action="store_true",
                   help="torch-spelling activation quantization")
    p.add_argument("--no-raw-gates", action="store_true",
                   help="alpha/beta computed in torch, not in-kernel")
    args = p.parse_args()

    # Applied in `_engine`, NOT here: every arm rebuilds its engine, and
    # `from_checkpoint` re-enables the fused path as part of construction, so a
    # one-shot disable at startup would silently evaporate on the second arm —
    # the flag would print "off" and measure "on".
    _TUNE["fused_quant"] = not args.no_fused_quant
    _TUNE["raw_gates"] = not args.no_raw_gates

    quant = "mlp" if args.quant_mlp else args.quant
    # Built once up front purely to state what the run actually quantized --
    # `maybe_quantize` declines shapes it cannot handle, so the request and the
    # result are not the same thing (`Engine.quant_report`).
    state_dtype = STATE_DTYPES[args.state_dtype]
    probe = _engine(True, quant, state_dtype)
    report_q, step_bytes = probe.quant_report(), probe.step_bytes()
    del probe
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    out: list[Arm] = []
    with HostHealthSampler() as health:
        for b in args.batches:
            try:
                out.extend(measure(b, args.steps, args.max_len, quant=quant,
                                   prompt_len=args.prompt_len,
                                   state_dtype=state_dtype))
            except torch.OutOfMemoryError as e:
                # **A batch that does not fit is a result, not a crash.** The
                # sweep spans arms with different footprints -- an fp32 state
                # pool stops at B=64 on this card where an fp16 one reaches 128
                # -- and killing the whole run at the first arm to run out means
                # no arm gets measured at any batch. Report the hole and go on;
                # `h2h_summarize.py` renders a missing point as "-", which is
                # the honest way to publish "this configuration does not fit".
                print(f"OOM at batch {b}: {str(e).splitlines()[0]}", file=sys.stderr)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    report = health.report()

    if args.json:
        print(json.dumps({"arms": [asdict(a) for a in out],
                          "health": str(report),
                          "quant": sorted(parse_groups(quant)),
                          "state_dtype": args.state_dtype,
                          "step_bytes": step_bytes}))
        return

    print(f"\nhost health: {report}")
    print(report_q)
    print(f"seeded with {args.prompt_len} tokens of KV per row; "
          f"{args.steps} timed steps -> KV {args.prompt_len}.."
          f"{args.prompt_len + args.steps}\n")
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f"{'batch':>5} {'arm':<18} {'ms/step':>9} {'tok/s':>10} {'peak GiB':>9}")
    for a in out:
        print(f"{a.batch:>5} {a.name:<18} {a.ms_per_step:>9.3f} {a.tok_per_s:>10.1f} "
              f"{a.peak_gib:>9.2f}")
    print(f"  (card total {total:.2f} GiB)")

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
