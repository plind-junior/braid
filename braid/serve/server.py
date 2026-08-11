"""SSE streaming server. ROADMAP Phase 4 item 1.

Stdlib only — `http.server` and `json`. The box is provisioned with torch and
pytest, not a web framework, and a serving loop whose bottleneck is a 10 ms GPU
step does not need one.

**The scheduler is single-threaded and owns the GPU.** Every CUDA call happens
on `_loop`; request handlers only ever touch thread-safe queues. That is not
tidiness — a captured CUDA graph replayed from two threads at once would
interleave writes into the same static buffers, and the failure would look like
one stream receiving another's tokens.

**Disconnect is the path that leaks.** A client that closes mid-generation must
release its recurrent slot *and* its KV extent, or the pool fills up and the
server wedges with no error anywhere. The roadmap flags this because the
reference engine shipped exactly that leak. Here the write to a dead socket
raises, the handler's `finally` calls `close()`, and `close()` reaches the
scheduler's `cancel()` — which is the only thing that returns a slot. The
`finally` is load-bearing; so is the fact that `cancel` is idempotent, because a
client that disconnects on the last token races with normal completion.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from braid.model.engine import Engine
from braid.model.quant import GROUPS
from braid.serve.scheduler import Request, Scheduler

_DONE = object()


def _fast_frame() -> bool:
    """`BRAID_FAST_FRAME=0` is the rollback seam for the hand-built SSE frame.

    Default ON. The frame it builds is BYTE-IDENTICAL to the `json.dumps` one:
    the payload is `{"id": <str>, "token": <int>}` in that insertion order,
    json's default separators are `', '` and `': '`, and request ids are `r0`,
    `r1`, ... from a counter, so no escaping is reachable. Only the no-tokenizer
    path takes it; a request that wants decoded text still goes through json.
    """
    return os.environ.get("BRAID_FAST_FRAME", "1") != "0"


def _apply_switch_interval() -> float | None:
    """Optionally widen CPython's GIL switch interval (`BRAID_SWITCH_INTERVAL`).

    Phase 0 (2026-08-11) measured `Scheduler.step()` taking 27.5 ms per tick
    under the HTTP server against an 11.9 ms in-process ITL for the same
    scheduler and the same engine, with the loop idle 0.7% of the window and
    the queue fan-out costing 0.29 ms. The inflation is INSIDE step(), which
    is where the scheduler's own Python runs — batch assembly, the `.tolist()`
    readback, the per-row sampling loop — between CUDA calls that release the
    GIL. 64 handler threads become runnable on every tick and contend for it.

    CPython hands the GIL over every `sys.getswitchinterval()` seconds (5 ms by
    default). Widening it lets the scheduler thread finish a burst of Python
    before it is preempted, at the cost of stream-latency jitter. This is a
    PROBE as much as a fix: if throughput does not move with it, the mechanism
    above is wrong and the fix is structural (one thread, not 65) rather than
    a tuning knob.
    """
    raw = os.environ.get("BRAID_SWITCH_INTERVAL")
    if not raw:
        return None
    v = float(raw)
    sys.setswitchinterval(v)
    return v


@dataclass
class _Stream:
    q: queue.Queue
    req: Request


class EngineService:
    """Owns the scheduler and the one thread allowed to touch the GPU."""

    def __init__(self, engine: Engine, capacity: int = 8, max_len: int = 2048,
                 graphed: bool = True, idle_sleep: float = 0.001,
                 sink=None):
        self.scheduler = Scheduler(engine, capacity=capacity, max_len=max_len,
                                   graphed=graphed)
        self.idle_sleep = idle_sleep
        # SINK MODE (braid/serve/aserver.py). When set, the GPU thread hands a
        # whole tick's updates to `sink` in ONE call instead of doing the
        # per-stream `Queue.put` fan-out itself. The fan-out is not expensive
        # in its own right — phase 0 measured it at 0.29 ms/tick — but each put
        # WAKES an OS thread, so the threaded server makes 64 threads runnable
        # every tick and they then serialize against the scheduler's own Python
        # inside `step()`, which is where phase 0 found the 45% inflation.
        # A sink that marshals once per tick onto an event loop turns 64
        # cross-thread wakeups into 1.
        self._sink = sink
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # TICK ACCOUNTING (diagnostic). The v3 recon measured braid 31% below
        # its own in-process `serve_bench` at the identical c=64 shape, and the
        # ITL arithmetic says the decode kernels are not where it went: the
        # published B=64 sweep is 12.23 ms/step against a served ITL of 19.55.
        # These three counters split one loop iteration into the part that is
        # the GPU (`step_s`) and the part that is this thread doing Python
        # (`fanout_s` — one lock acquire and one `Queue.put` per live stream,
        # each of which WAKES a blocked handler thread, 64 of them at c=64,
        # after which this thread has to win the GIL back to launch the next
        # step). `busy_s` is the whole non-idle iteration, so
        # busy - step - fanout is everything else, including time simply not
        # scheduled. Written only by the GPU thread; read from `/health` by a
        # handler thread, where a torn or stale read is a diagnostic that is one
        # tick out of date and nothing worse — so they are deliberately not
        # locked, because locking them would add the very contention they exist
        # to measure.
        self._ticks = 0
        self._step_s = 0.0
        self._fanout_s = 0.0
        self._busy_s = 0.0
        # IDLE ACCOUNTING. The first cut of the counters above measured only
        # non-idle iterations, which made the one quantity that mattered
        # invisible: a loop with nothing to run does `_stop.wait(idle_sleep)`
        # and `continue`s WITHOUT incrementing a tick. Phase 0 (2026-08-11)
        # duly reported `other_ms = -0.000` — no starvation inside a tick —
        # while the HTTP arm still ran 36.6% under the in-process one, because
        # the missing time was never inside a tick at all. `idle_s` is the
        # scheduler having no work: request turnaround, the client's ramp, and
        # anything else that keeps the batch from being full.
        self._idle_ticks = 0
        self._idle_s = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="braid-scheduler")
        self._thread.start()

    # --- client side ----------------------------------------------------------

    def submit(self, req: Request) -> queue.Queue:
        """Returns a queue yielding token ids, then `None` at end of stream."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._streams[req.id] = _Stream(q=q, req=req)
        self._inbox.put(("submit", req))
        return q

    def submit_raw(self, req: Request) -> None:
        """Submit with no delivery queue — sink mode owns delivery itself."""
        self._inbox.put(("submit", req))

    def close(self, req_id: str) -> None:
        """Idempotent. Safe to call from a handler's `finally` on any exit."""
        with self._lock:
            self._streams.pop(req_id, None)
        self._inbox.put(("cancel", req_id))

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    @property
    def stats(self) -> dict:
        s = self.scheduler
        n = max(self._ticks, 1)
        return {"capacity": s.capacity, "free_slots": s.free_slots,
                "live": len(s.live_ids), "max_decode_batch": s.max_decode_batch,
                # Per-tick millisecond split. `other_ms` is the residual —
                # busy minus the two accounted parts — and it is the number
                # this instrumentation exists for: if it grows with
                # concurrency while `step_ms` stays flat, the GPU thread is
                # being starved rather than the kernels being slow.
                "ticks": self._ticks,
                # Raw cumulative sums as well as the averages: a caller that
                # samples /health before and after a measured wave can diff
                # these and get the window's split exactly, instead of one
                # polluted by the warmup wave and by server-start idle.
                "step_s": round(self._step_s, 6),
                "fanout_s": round(self._fanout_s, 6),
                "busy_s": round(self._busy_s, 6),
                "idle_ticks": self._idle_ticks,
                "idle_s": round(self._idle_s, 6),
                "step_ms": round(self._step_s / n * 1e3, 3),
                "fanout_ms": round(self._fanout_s / n * 1e3, 3),
                "busy_ms": round(self._busy_s / n * 1e3, 3),
                "other_ms": round(
                    (self._busy_s - self._step_s - self._fanout_s) / n * 1e3, 3)}

    # --- the one GPU thread ---------------------------------------------------

    def _loop(self) -> None:
        sched = self.scheduler
        while not self._stop.is_set():
            drained = False
            while True:
                try:
                    kind, payload = self._inbox.get_nowait()
                except queue.Empty:
                    break
                drained = True
                if kind == "submit":
                    sched.submit(payload)
                else:
                    sched.cancel(payload)

            if sched.idle:
                t_idle = time.perf_counter()
                if not drained:
                    self._stop.wait(self.idle_sleep)
                self._idle_s += time.perf_counter() - t_idle
                self._idle_ticks += 1
                continue

            t_busy = time.perf_counter()
            ups = sched.step()
            t_step = time.perf_counter()
            if self._sink is not None:
                # One handoff for the whole tick. The sink owns delivery and
                # its own stream bookkeeping; `_streams` stays empty here.
                self._sink(ups)
                now = time.perf_counter()
                self._step_s += t_step - t_busy
                self._fanout_s += now - t_step
                self._busy_s += now - t_busy
                self._ticks += 1
                continue
            for up in ups:
                with self._lock:
                    stream = self._streams.get(up.id)
                if stream is None:
                    # Cancelled between the step and here; its slot is already
                    # on its way back through the inbox.
                    continue
                for tok in up.tokens:
                    stream.q.put(tok)
                if up.finished:
                    stream.q.put(None)
                    with self._lock:
                        self._streams.pop(up.id, None)
            now = time.perf_counter()
            self._step_s += t_step - t_busy
            self._fanout_s += now - t_step
            self._busy_s += now - t_busy
            self._ticks += 1


def make_handler(service: EngineService, tokenizer=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):        # noqa: D102 - quiet by default
            pass

        def _json(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):               # noqa: N802
            if self.path.rstrip("/") in ("/health", ""):
                self._json(200, {"ok": True, **service.stats})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):              # noqa: N802
            if self.path.rstrip("/") != "/generate":
                self._json(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": f"bad body: {e}"})
                return

            tokens = body.get("prompt_tokens")
            if tokens is None:
                text = body.get("prompt")
                if text is None or tokenizer is None:
                    self._json(400, {"error": "send prompt_tokens, or prompt with "
                                              "a tokenizer configured"})
                    return
                tokens = tokenizer.encode(text)

            try:
                req = Request(
                    prompt=list(tokens),
                    max_new_tokens=int(body.get("max_new_tokens", 32)),
                    temperature=float(body.get("temperature", 0.0)),
                    top_p=float(body.get("top_p", 1.0)),
                    seed=body.get("seed"),
                    eos_token_id=body.get("eos_token_id"),
                )
            except (ValueError, TypeError) as e:
                self._json(400, {"error": str(e)})
                return

            q = service.submit(req)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            # Built once per request, not once per token: at c=64 this loop
            # runs ~3,300 times a second across 64 threads, and every
            # microsecond of Python in it is a microsecond the scheduler thread
            # is not holding the GIL.
            fast = tokenizer is None and _fast_frame()
            head = f'data: {{"id": "{req.id}", "token": '.encode()
            try:
                while True:
                    tok = q.get()
                    if tok is None:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return
                    if fast:
                        self.wfile.write(head + b"%d}\n\n" % tok)
                    else:
                        payload = {"id": req.id, "token": tok}
                        if tokenizer is not None:
                            payload["text"] = tokenizer.decode([tok])
                        self.wfile.write(
                            f"data: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The client went away mid-generation. `finally` releases.
                return
            finally:
                service.close(req.id)

    return Handler


class _Server(ThreadingHTTPServer):
    # socketserver's default listen backlog is 5. A server whose whole purpose
    # is c=128 concurrency meets 128 near-simultaneous connects on its first
    # busy second, and a full accept queue does not degrade politely: the
    # kernel resets or silently drops SYNs, the client sees ConnectionReset or
    # a 1-2-4s TCP retransmission ladder, and the measurement reads "braid
    # collapsed at c=128" when nothing past the listen(2) call ever ran.
    # Measured before this line existed: 358 resets and ~10 tok/s at c=128;
    # zero errors after. llama-server's httplib has the same default and the
    # same failure shape, which is worth knowing when benchmarking against it.
    request_queue_size = 512


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8000,
          capacity: int = 8, max_len: int = 2048, graphed: bool = True,
          tokenizer=None) -> tuple[ThreadingHTTPServer, EngineService]:
    si = _apply_switch_interval()
    if si is not None:
        print(f"braid: sys.setswitchinterval({si}) "
              f"(BRAID_SWITCH_INTERVAL); fast_frame={_fast_frame()}")
    service = EngineService(engine, capacity=capacity, max_len=max_len,
                            graphed=graphed)
    httpd = _Server((host, port), make_handler(service, tokenizer))
    return httpd, service


def main() -> None:
    import argparse
    import os
    from pathlib import Path

    import torch

    from braid.model.loader import load_checkpoint

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path,
                   default=Path(os.environ.get("BRAID_MODEL_DIR",
                                               "/root/models/Qwen3.5-4B")))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--capacity", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--quant", default="",
                   help=f"FP8 W8A8 groups: {','.join(GROUPS)}, or 'all'")
    p.add_argument("--quant-mlp", action="store_true",
                   help="shorthand for --quant mlp")
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--state-dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="storage type for the recurrent state pool; the scan is "
                        "fp32 either way. The shipping config is fp16.")
    p.add_argument("--http", default="threaded",
                   choices=["threaded", "asyncio", "split"],
                   help="serving front end. 'threaded' is one OS thread per "
                        "connection (the original); 'asyncio' is one event-loop "
                        "thread with a single scheduler->loop handoff per tick "
                        "(phase 2, +9.3%%); 'split' keeps that front end and "
                        "moves the engine into its own process, so serving "
                        "Python and engine Python stop sharing a GIL (phase 3). "
                        "See braid/serve/aserver.py and braid/serve/proto.py "
                        "for what phases 0-2 measured before each was written.")
    args = p.parse_args()

    quant = "mlp" if args.quant_mlp else args.quant

    if args.http == "split":
        # Dispatched before the checkpoint load: in split mode THIS process
        # must never own a GPU context. `torch` is still imported — this
        # module imports Engine at module scope — but importing it allocates
        # nothing on the device, so the process does not appear in
        # `nvidia-smi --query-compute-apps`. The phase-3 harness records that
        # listing as a receipt rather than trusting this comment.
        from braid.serve.aserver import serve_split

        engine_argv = ["--model-dir", str(args.model_dir),
                       "--quant", quant, "--state-dtype", args.state_dtype]
        if args.no_graphs:
            engine_argv.append("--no-graphs")
        serve_split(args.host, args.port, engine_argv, capacity=args.capacity,
                    max_len=args.max_len)
        return

    from braid.bench.decode_speed import STATE_DTYPES

    ck = load_checkpoint(args.model_dir, device="cuda", dtype=torch.bfloat16)
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16,
                                 use_kernels=True, quant=quant,
                                 state_dtype=STATE_DTYPES[args.state_dtype])
    del ck
    torch.cuda.empty_cache()
    if args.http == "asyncio":
        from braid.serve.aserver import serve_async
        _apply_switch_interval()
        serve_async(eng, args.host, args.port, capacity=args.capacity,
                    max_len=args.max_len, graphed=not args.no_graphs)
        return

    httpd, service = serve(eng, args.host, args.port, capacity=args.capacity,
                           max_len=args.max_len, graphed=not args.no_graphs)
    print(f"braid serving on http://{args.host}:{args.port} "
          f"(capacity {args.capacity}, max_len {args.max_len})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        service.shutdown()


if __name__ == "__main__":
    main()
