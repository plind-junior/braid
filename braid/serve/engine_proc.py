"""The engine half of the split server. Phase 3 of the c=64 deficit work.

Run as `python -m braid.serve.engine_proc --uds <path> ...`, normally spawned by
`braid.serve.aserver.serve_split`. This process owns the GPU and nothing else:
it holds the checkpoint, the scheduler, the CUDA graphs and the recurrent slot
pool, and it speaks `braid.serve.proto` over a unix socket. No HTTP, no
sockets to clients, no per-request Python beyond building a `Request`.

**Why it exists.** Phase 2 removed 64 handler threads and won 9.3%, which
confirmed the mechanism, and left `step()` at 25.06 ms per tick against the
~19 ms the in-process arm implies. One event loop is still one interpreter
shared with the scheduler: every SSE frame, every `json.loads` of a request
body, every `Queue.put_nowait` is Python that can only run while the GPU thread
does not hold the GIL. Splitting the process is the only move left that removes
the sharing rather than reducing it, and it is what vLLM does.

**Threads here: two.** The scheduler thread inside `EngineService` (unchanged,
still the only thing that touches CUDA) and this module's main thread, which
reads frames and pushes them into the same `_inbox` an HTTP handler used to.
Both write to the socket, so writes take a lock; the GPU thread takes it ~40
times a second and the reader ~50, each for a few hundred bytes to a UDS, so
the contention is nothing like the 64-way contention this whole exercise is
about.

**Backpressure is deliberate.** `sendall` from the GPU thread blocks if the
front end has stopped draining. That is the correct behaviour: a front end that
cannot keep up should slow the engine down, not have ticks silently dropped
with streams left hanging.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

from braid.serve import proto


def _connect(path: str, timeout: float):
    import socket

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(path)
    s.settimeout(None)          # blocking from here on; see proto.FrameReader
    return s


def main() -> None:
    p = argparse.ArgumentParser(description="braid engine process")
    p.add_argument("--uds", required=True,
                   help="unix socket the front end is listening on")
    p.add_argument("--model-dir", type=Path,
                   default=Path(os.environ.get("BRAID_MODEL_DIR",
                                               "/root/models/Qwen3.5-4B")))
    p.add_argument("--capacity", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--quant", default="")
    p.add_argument("--state-dtype", default="fp32",
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--stats-ms", type=float, default=20.0,
                   help="cadence of the pushed stats frame. /health on the "
                        "front end is served from the last one, so this is "
                        "also the staleness bound on the per-tick split the "
                        "benchmark harness reads. 20 ms against a ~25 ms tick "
                        "costs ~50 json.dumps a second in this process and "
                        "keeps the window edges honest.")
    args = p.parse_args()

    sock = _connect(args.uds, timeout=30.0)

    # Imported here, after the socket is up, so a bad --uds fails in a second
    # instead of after a two-minute checkpoint load.
    import torch

    from braid.bench.decode_speed import STATE_DTYPES
    from braid.model.engine import Engine
    from braid.model.loader import load_checkpoint
    from braid.serve.scheduler import Request, Update
    from braid.serve.server import EngineService, _apply_switch_interval

    _apply_switch_interval()
    ck = load_checkpoint(args.model_dir, device="cuda", dtype=torch.bfloat16)
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16,
                                 use_kernels=True, quant=args.quant,
                                 state_dtype=STATE_DTYPES[args.state_dtype])
    del ck
    torch.cuda.empty_cache()

    wlock = threading.Lock()
    dead = threading.Event()

    def send(b: bytes) -> None:
        with wlock:
            sock.sendall(b)

    def sink(ups) -> None:
        # Runs on the GPU thread. One frame for the whole tick.
        try:
            send(proto.encode_tick(ups))
        except OSError:
            # Front end vanished. Do not let this kill the scheduler thread
            # mid-step; the main loop below notices and shuts down in order.
            dead.set()

    service = EngineService(eng, capacity=args.capacity, max_len=args.max_len,
                            graphed=not args.no_graphs, sink=sink)
    send(proto.json_frame(proto.READY, {"pid": os.getpid(),
                                        "capacity": args.capacity,
                                        "max_len": args.max_len}))
    print(f"braid engine process ready (pid {os.getpid()}, "
          f"capacity {args.capacity}, max_len {args.max_len})", flush=True)

    fr = proto.FrameReader(sock)
    period = args.stats_ms / 1e3
    nxt = 0.0
    try:
        while not dead.is_set():
            now = time.perf_counter()
            if now >= nxt:
                # Counters are written by the GPU thread and read here without
                # a lock, exactly as /health read them in the threaded server:
                # each field is one attribute read, so the worst case is a
                # snapshot whose fields are microseconds apart. Locking them
                # would add the contention they exist to measure.
                send(proto.json_frame(proto.STATS, service.stats))
                nxt = now + period
            got = fr.next(timeout=period)
            if got is None:
                continue
            kind, payload = got
            if kind == proto.SUBMIT:
                o = json.loads(payload)
                try:
                    req = Request(prompt=o["prompt"], id=o["id"],
                                  max_new_tokens=o["max_new_tokens"],
                                  temperature=o["temperature"],
                                  top_p=o["top_p"], seed=o["seed"],
                                  eos_token_id=o["eos_token_id"])
                except (ValueError, TypeError, KeyError) as e:
                    # The front end validates too, so this is the belt to its
                    # braces. Answer with a finished update rather than an
                    # error frame: the stream is already open and the only
                    # thing that must not happen is that it hangs.
                    print(f"braid engine: rejecting {o.get('id')}: {e}",
                          flush=True)
                    send(proto.encode_tick(
                        [Update(id=o.get("id", ""), finished=True,
                                reason="rejected")]))
                    continue
                service.submit_raw(req)
            elif kind == proto.CANCEL:
                service.close(payload.decode())
            elif kind == proto.SHUTDOWN:
                break
    except ConnectionError as e:
        print(f"braid engine: {e}", flush=True)
    finally:
        service.shutdown()
        try:
            sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
