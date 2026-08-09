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
import queue
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from braid.model.engine import Engine
from braid.model.quant import GROUPS
from braid.serve.scheduler import Request, Scheduler

_DONE = object()


@dataclass
class _Stream:
    q: queue.Queue
    req: Request


class EngineService:
    """Owns the scheduler and the one thread allowed to touch the GPU."""

    def __init__(self, engine: Engine, capacity: int = 8, max_len: int = 2048,
                 graphed: bool = True, idle_sleep: float = 0.001):
        self.scheduler = Scheduler(engine, capacity=capacity, max_len=max_len,
                                   graphed=graphed)
        self.idle_sleep = idle_sleep
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
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
        return {"capacity": s.capacity, "free_slots": s.free_slots,
                "live": len(s.live_ids), "max_decode_batch": s.max_decode_batch}

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
                if not drained:
                    self._stop.wait(self.idle_sleep)
                continue

            for up in sched.step():
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
            try:
                while True:
                    tok = q.get()
                    if tok is None:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return
                    payload = {"id": req.id, "token": tok}
                    if tokenizer is not None:
                        payload["text"] = tokenizer.decode([tok])
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The client went away mid-generation. `finally` releases.
                return
            finally:
                service.close(req.id)

    return Handler


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8000,
          capacity: int = 8, max_len: int = 2048, graphed: bool = True,
          tokenizer=None) -> tuple[ThreadingHTTPServer, EngineService]:
    service = EngineService(engine, capacity=capacity, max_len=max_len,
                            graphed=graphed)
    httpd = ThreadingHTTPServer((host, port), make_handler(service, tokenizer))
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
    args = p.parse_args()

    quant = "mlp" if args.quant_mlp else args.quant
    ck = load_checkpoint(args.model_dir, device="cuda", dtype=torch.bfloat16)
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16,
                                 use_kernels=True, quant=quant)
    del ck
    torch.cuda.empty_cache()
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
