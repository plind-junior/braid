"""Single-threaded asyncio serving path. Phase 2 of the c=64 deficit work.

**Why this exists, measured rather than assumed.** The threaded server
(`braid/serve/server.py`) runs one OS thread per connection. Phase 0
(2026-08-11, `docs/runbooks/vllm-recon.md`) put braid's engine AHEAD of
vLLM-FP8 in-process — 3,322 vs 3,164 tok/s at c=64 — and found the entire
deficit in the HTTP layer: the same `Scheduler.step()` takes 27.5 ms per tick
under the server against an 11.9 ms in-process ITL. Four candidate causes were
measured and eliminated:

  * arrival pattern — the scheduler is idle 0.7% of the window
  * the queue fan-out — 0.29 ms per tick
  * GIL handoff cadence — widening `sys.setswitchinterval` to 20 ms and 100 ms
    is a null (phase 1, +3.3% / +4.1% against a frame-only +3.9%)
  * core starvation — the box has 64 cores

What survives is that only one thread runs Python at a time, and 64 handler
threads each doing microseconds per token serialize against the scheduler's own
Python inside `step()`. So the fix is structural: one thread, and **one
cross-thread handoff per tick instead of 64**. The GPU thread hands a whole
tick's updates to `EngineService`'s sink; the loop fans out to per-stream
`asyncio.Queue`s with no OS-thread wakeups at all.

**The invariants the threaded server earned are kept.** The scheduler still owns
the GPU on its own thread and nothing here touches CUDA. A client that
disconnects mid-generation still releases its slot *and* its KV extent — the
write raises, `finally` cancels, and cancel is idempotent, because a client
that drops on the last token races with normal completion. The listen backlog
is still large: `asyncio.start_server(backlog=...)` gets the same 512 the
threaded server needed after 358 resets at c=128.

This module deliberately imports nothing that needs torch at module scope, so
its HTTP and delivery logic can be tested on a machine with no GPU.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from braid.serve import proto

MAX_HEADER_BYTES = 64 * 1024
MAX_BODY_BYTES = 64 * 1024 * 1024


class Bridge:
    """Marshals scheduler ticks onto the event loop and fans out to streams.

    `on_tick` runs on the GPU thread and must do as little as possible: one
    `call_soon_threadsafe` per tick, carrying the whole batch. Everything else
    happens on the loop, where `Queue.put_nowait` wakes no OS thread.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._streams: dict[str, asyncio.Queue] = {}

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def register(self, req_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._streams[req_id] = q
        return q

    def unregister(self, req_id: str) -> None:
        self._streams.pop(req_id, None)

    # --- GPU thread ---------------------------------------------------------

    def on_tick(self, ups: list) -> None:
        """Called from the scheduler thread, once per tick. Never blocks."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._deliver, ups)
        except RuntimeError:
            # Loop already closed (shutdown race). Dropping is correct: there
            # is nobody left to deliver to.
            pass

    # --- event loop ---------------------------------------------------------

    def _deliver(self, ups: list) -> None:
        for up in ups:
            q = self._streams.get(up.id)
            if q is None:
                # Cancelled between the step and here; its slot is already on
                # its way back through the scheduler inbox.
                continue
            for tok in up.tokens:
                q.put_nowait(tok)
            if up.finished:
                q.put_nowait(None)
                self._streams.pop(up.id, None)

    def close_all(self) -> None:
        """End every open stream. Only the split server uses this, and only
        when the engine process has gone: a handler parked on `q.get()` waits
        forever otherwise, which hangs the client AND wedges shutdown, because
        `Server.wait_closed()` waits for its handlers. Ending the stream turns
        an engine crash into short responses the client can see rather than a
        server that answers /health while producing nothing."""
        for q in list(self._streams.values()):
            q.put_nowait(None)
        self._streams.clear()

    def deliver_decoded(self, items: list[tuple[str, list, bool]]) -> None:
        """Same fan-out for the split server, whose ticks arrive off a socket
        as plain tuples rather than as `Update`s (the front-end process has no
        torch and so cannot import the scheduler)."""
        for rid, toks, fin in items:
            q = self._streams.get(rid)
            if q is None:
                continue
            for tok in toks:
                q.put_nowait(tok)
            if fin:
                q.put_nowait(None)
                self._streams.pop(rid, None)


async def read_request(reader: asyncio.StreamReader
                       ) -> tuple[str, str, dict[str, str], bytes] | None:
    """Minimal HTTP/1.1 request read. `None` on a clean EOF or a bad frame.

    Deliberately not a general HTTP implementation: it accepts the request
    line, the headers it needs, and a Content-Length body. Chunked request
    bodies are not accepted, because no client of this server sends one and
    silently mis-framing a body is worse than refusing it.
    """
    try:
        line = await reader.readline()
    except (ConnectionResetError, asyncio.IncompleteReadError):
        return None
    if not line:
        return None
    parts = line.decode("latin-1").split()
    if len(parts) < 2:
        return None
    method, path = parts[0], parts[1]
    headers: dict[str, str] = {}
    total = len(line)
    while True:
        h = await reader.readline()
        total += len(h)
        if total > MAX_HEADER_BYTES:
            return None
        if h in (b"\r\n", b"\n", b""):
            break
        k, _, v = h.decode("latin-1").partition(":")
        headers[k.strip().lower()] = v.strip()
    n = int(headers.get("content-length", "0") or 0)
    if n < 0 or n > MAX_BODY_BYTES:
        return None
    body = await reader.readexactly(n) if n else b""
    return method, path, headers, body


def _response(code: int, ctype: str, body: bytes, extra: str = "") -> bytes:
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found",
              413: "Payload Too Large"}.get(code, "OK")
    return (f"HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n{extra}\r\n"
            ).encode() + body


def make_handler(service: Any, bridge: Bridge, tokenizer: Any = None,
                 request_factory: Callable[..., Any] | None = None):
    """Build the connection handler. `request_factory` is injectable so the
    protocol can be exercised without importing the CUDA-backed scheduler."""

    def _make_request(**kw):
        if request_factory is not None:
            return request_factory(**kw)
        from braid.serve.scheduler import Request
        return Request(**kw)

    async def handle(reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter) -> None:
        req_id = None
        try:
            got = await read_request(reader)
            if got is None:
                return
            method, path, _headers, body = got

            if method == "GET" and path.startswith("/health"):
                payload = json.dumps({"ok": True, **service.stats}).encode()
                writer.write(_response(200, "application/json", payload))
                await writer.drain()
                return
            if method != "POST" or not path.startswith("/generate"):
                writer.write(_response(404, "application/json",
                                       b'{"error": "not found"}'))
                await writer.drain()
                return

            try:
                obj = json.loads(body or b"{}")
            except (ValueError, json.JSONDecodeError) as e:
                writer.write(_response(400, "application/json",
                                       json.dumps({"error": f"bad body: {e}"}
                                                  ).encode()))
                await writer.drain()
                return

            tokens = obj.get("prompt_tokens")
            if tokens is None:
                text = obj.get("prompt")
                if text is None or tokenizer is None:
                    writer.write(_response(
                        400, "application/json",
                        b'{"error": "send prompt_tokens, or prompt with a '
                        b'tokenizer configured"}'))
                    await writer.drain()
                    return
                tokens = tokenizer.encode(text)

            try:
                req = _make_request(
                    prompt=list(tokens),
                    max_new_tokens=int(obj.get("max_new_tokens", 32)),
                    temperature=float(obj.get("temperature", 0.0)),
                    top_p=float(obj.get("top_p", 1.0)),
                    seed=obj.get("seed"),
                    eos_token_id=obj.get("eos_token_id"),
                )
            except (ValueError, TypeError) as e:
                writer.write(_response(400, "application/json",
                                       json.dumps({"error": str(e)}).encode()))
                await writer.drain()
                return

            req_id = req.id
            q = bridge.register(req_id)
            service.submit_raw(req)

            writer.write(b"HTTP/1.1 200 OK\r\n"
                         b"Content-Type: text/event-stream\r\n"
                         b"Cache-Control: no-cache\r\n"
                         b"Connection: close\r\n\r\n")
            await writer.drain()

            # Same byte-identical frame the threaded path uses: request ids are
            # `r0`, `r1`, ... from a counter and the payload is
            # {"id": <str>, "token": <int>}, so no escaping is reachable.
            head = f'data: {{"id": "{req_id}", "token": '.encode()
            while True:
                tok = await q.get()
                if tok is None:
                    writer.write(b"data: [DONE]\n\n")
                    await writer.drain()
                    return
                if tokenizer is None:
                    writer.write(head + b"%d}\n\n" % tok)
                else:
                    payload = {"id": req_id, "token": tok,
                               "text": tokenizer.decode([tok])}
                    writer.write(f"data: {json.dumps(payload)}\n\n".encode())
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError,
                asyncio.IncompleteReadError):
            # The client went away mid-generation. `finally` releases.
            return
        finally:
            # Load-bearing, exactly as in the threaded server: this is the only
            # thing that returns a slot, and it must run on every exit path.
            if req_id is not None:
                bridge.unregister(req_id)
                service.close(req_id)
            try:
                writer.close()
            except OSError:
                pass

    return handle


async def _serve_forever(engine, host: str, port: int, capacity: int,
                         max_len: int, graphed: bool, tokenizer=None):
    from braid.serve.server import EngineService

    bridge = Bridge()
    bridge.bind(asyncio.get_running_loop())
    service = EngineService(engine, capacity=capacity, max_len=max_len,
                            graphed=graphed, sink=bridge.on_tick)
    handler = make_handler(service, bridge, tokenizer)
    # backlog 512 for the same reason the threaded server needed it: at c=128
    # a default backlog of 5 produced 358 resets and ~10 tok/s, and the row
    # read like a capacity collapse when nothing past listen(2) had run.
    server = await asyncio.start_server(handler, host, port, backlog=512)
    print(f"braid serving on http://{host}:{port} "
          f"(asyncio, capacity {capacity}, max_len {max_len})", flush=True)
    try:
        async with server:
            await server.serve_forever()
    finally:
        service.shutdown()


def serve_async(engine, host: str = "127.0.0.1", port: int = 8000,
                capacity: int = 8, max_len: int = 2048, graphed: bool = True,
                tokenizer=None) -> None:
    """Blocking entrypoint, mirroring `server.serve` + `serve_forever`."""
    asyncio.run(_serve_forever(engine, host, port, capacity, max_len,
                               graphed, tokenizer))


# --- split server: the engine in its own process (phase 3) -------------------
#
# Everything below runs in a process that never imports the model. It keeps the
# HTTP machinery above verbatim — same handler, same Bridge, same byte-identical
# SSE frame — and swaps `EngineService` for a stub that puts the same three
# calls (`submit_raw`, `close`, `stats`) on a socket instead of a queue. That
# substitution is the whole of phase 3: if the split arm wins, the win is
# attributable to the process boundary and nothing else, because no other line
# on the request path changed.

_rids = itertools.count(1)


@dataclass
class RemoteRequest:
    """Front-end mirror of `scheduler.Request`, minus everything torch-shaped.

    Validation is duplicated on purpose. The engine constructs the real
    `Request` and would reject the same inputs, but a round trip to find out
    turns a 400 into a hung stream, and the front end already owed the client
    an answer before it owed the engine a submit.
    """

    prompt: list
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    eos_token_id: int | None = None
    id: str = field(default_factory=lambda: f"r{next(_rids)}")

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError(f"request {self.id} has an empty prompt")
        if self.max_new_tokens < 1:
            raise ValueError(f"request {self.id}: max_new_tokens must be >= 1")


class RemoteService:
    """`EngineService`'s surface, backed by a socket to the engine process.

    `stats` is served from the last pushed frame rather than fetched, so
    `/health` never waits on the GPU process. The cost is a staleness bound
    equal to the engine's `--stats-ms` (20 ms by default). The benchmark
    harness brackets a ~2 s wave with two samples, so both edges carry the same
    bound and the wave's per-tick split is out by at most one stats period at
    each end. tok/s comes from the client's own wall clock and is untouched.
    """

    def __init__(self, writer: asyncio.StreamWriter, extra: dict | None = None):
        self._w = writer
        self._extra = extra or {}
        self._stats: dict = {}

    @property
    def stats(self) -> dict:
        return {**self._extra, **self._stats}

    def set_stats(self, d: dict) -> None:
        self._stats = d

    def submit_raw(self, req: RemoteRequest) -> None:
        self._w.write(proto.json_frame(proto.SUBMIT, {
            "id": req.id, "prompt": req.prompt,
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature, "top_p": req.top_p,
            "seed": req.seed, "eos_token_id": req.eos_token_id}))

    def close(self, req_id: str) -> None:
        # Load-bearing exactly as in-process: this is what returns the slot and
        # the KV extent. It is also the only frame a disconnect produces, so it
        # must never be able to raise into the handler's `finally`.
        try:
            self._w.write(proto.frame(proto.CANCEL, req_id.encode()))
        except (OSError, RuntimeError):
            pass


async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    hdr = await reader.readexactly(4)
    n = int.from_bytes(hdr, "big")
    if n > proto.MAX_FRAME:
        raise ConnectionError(f"frame of {n} bytes: stream lost")
    p = await reader.readexactly(n)
    return p[0], p[1:]


async def _watch_child(proc: subprocess.Popen) -> int:
    while proc.poll() is None:
        await asyncio.sleep(0.5)
    return proc.returncode


async def _race_child(aw, proc: subprocess.Popen, timeout: float, what: str):
    """Await `aw`, but fail fast if the engine process dies first.

    Without this a checkpoint that OOMs or a bad `--quant` shows up as the
    front end hanging on accept for fifteen minutes, and the benchmark reports
    "never healthy" for a reason that was printed on stderr in the first second.
    """
    task = asyncio.ensure_future(aw)
    watch = asyncio.ensure_future(_watch_child(proc))
    try:
        done, _ = await asyncio.wait({task, watch}, timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        if watch in done:
            raise RuntimeError(f"engine process exited with code "
                               f"{watch.result()} before it could {what}")
        raise RuntimeError(f"engine process did not {what} in {timeout:.0f}s")
    finally:
        for t in (task, watch):
            if not t.done():
                t.cancel()


async def _pump(reader: asyncio.StreamReader, bridge: Bridge,
                service: RemoteService) -> None:
    """The only reader of the engine socket. One frame per tick, not per token.

    This is the phase-2 handoff moved across a process boundary: the engine
    packs a whole tick, this coroutine unpacks it and fans out to per-stream
    queues, and no OS thread is woken anywhere in the path.
    """
    while True:
        try:
            kind, payload = await _read_frame(reader)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            print("braid: engine process went away", file=sys.stderr,
                  flush=True)
            bridge.close_all()
            return
        if kind == proto.TICK:
            bridge.deliver_decoded(proto.decode_tick(payload))
        elif kind == proto.STATS:
            service.set_stats(json.loads(payload))


def _set_once(fut: asyncio.Future, value) -> None:
    if not fut.done():
        fut.set_result(value)


def _pdeathsig() -> None:
    """`PR_SET_PDEATHSIG` in the child, between fork and exec.

    The last line of defence against an orphaned engine. If the front end is
    SIGKILLed — or the benchmark harness kills it and something eats the
    SIGTERM — the child would otherwise keep the checkpoint and the whole slot
    pool resident, and the next arm of an A/B finds the GPU busy and either
    OOMs or, worse, runs slower and gets published. Best effort by design: on
    a kernel or libc where this is unavailable the process-group kill in the
    harness is still there.
    """
    try:
        import ctypes
        import signal as _sig

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, _sig.SIGTERM)
    except Exception:  # noqa: BLE001 - never let this break the spawn
        pass


async def _split_serve(host: str, port: int, engine_argv: list[str],
                       capacity: int, max_len: int, tokenizer,
                       engine_cmd: list[str] | None, ready_timeout: float
                       ) -> None:
    loop = asyncio.get_running_loop()
    tmp = tempfile.mkdtemp(prefix="braid-engine-")
    upath = os.path.join(tmp, "s")
    lsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    proc: subprocess.Popen | None = None
    writer: asyncio.StreamWriter | None = None
    try:
        lsock.bind(upath)
        lsock.listen(1)
        lsock.setblocking(False)
        cmd = list(engine_cmd or [sys.executable, "-B", "-m",
                                  "braid.serve.engine_proc"])
        cmd += ["--uds", upath, "--capacity", str(capacity),
                "--max-len", str(max_len), *engine_argv]
        # stdout/stderr are inherited: the engine's load progress and any
        # traceback land in the same log as the front end's, which is what a
        # benchmark harness tails when an arm fails to come up.
        proc = subprocess.Popen(cmd, preexec_fn=_pdeathsig)  # noqa: PLW1509
        conn, _ = await _race_child(loop.sock_accept(lsock), proc,
                                    ready_timeout, "connect")
        reader, writer = await asyncio.open_unix_connection(sock=conn)
        kind, payload = await _race_child(_read_frame(reader), proc,
                                          ready_timeout, "load the model")
        if kind != proto.READY:
            raise RuntimeError(f"engine sent frame {kind:#04x} before READY")
        info = json.loads(payload or b"{}")

        bridge = Bridge()
        bridge.bind(loop)
        service = RemoteService(writer, extra={"front_pid": os.getpid(),
                                               "engine_pid": info.get("pid")})
        pump = asyncio.ensure_future(_pump(reader, bridge, service))
        handler = make_handler(service, bridge, tokenizer,
                               request_factory=RemoteRequest)
        server = await asyncio.start_server(handler, host, port, backlog=512)
        print(f"braid serving on http://{host}:{port} (split; front end pid "
              f"{os.getpid()}, engine pid {info.get('pid')}, capacity "
              f"{capacity}, max_len {max_len})", flush=True)
        forever = asyncio.ensure_future(server.serve_forever())
        # SIGTERM has to reach the `finally` below, because that is what stops
        # the engine process. Python's default disposition for SIGTERM ends the
        # interpreter without unwinding, which would leave a CUDA process
        # holding the card after `kill <front end pid>` — exactly what a
        # benchmark harness does between arms.
        term: asyncio.Future = loop.create_future()
        handled = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _set_once, term, sig)
                handled.append(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        try:
            # Whichever ends first ends the server: if the engine dies there is
            # nothing left to serve, and a front end that keeps answering
            # /health after that would report a healthy arm producing nothing.
            await asyncio.wait({pump, forever, term},
                               return_when=asyncio.FIRST_COMPLETED)
            if term.done():
                print(f"braid: signal {term.result()}, shutting down",
                      flush=True)
        finally:
            for sig in handled:
                loop.remove_signal_handler(sig)
            for t in (pump, forever, term):
                if not t.done():
                    t.cancel()
            bridge.close_all()
            server.close()
            # Bounded: `wait_closed` waits for every live handler, and a client
            # that has stopped reading can keep one parked in `drain()` past
            # any deadline worth honouring during shutdown.
            try:
                await asyncio.wait_for(server.wait_closed(), timeout=10.0)
            except (TimeoutError, asyncio.TimeoutError):
                print("braid: handlers still live at shutdown; exiting anyway",
                      file=sys.stderr, flush=True)
    finally:
        if writer is not None:
            try:
                writer.write(proto.frame(proto.SHUTDOWN))
                writer.close()
            except (OSError, RuntimeError):
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        lsock.close()
        for path, rm in ((upath, os.unlink), (tmp, os.rmdir)):
            try:
                rm(path)
            except OSError:
                pass


def serve_split(host: str = "127.0.0.1", port: int = 8000,
                engine_argv: list[str] | None = None, capacity: int = 8,
                max_len: int = 2048, tokenizer=None,
                engine_cmd: list[str] | None = None,
                ready_timeout: float = 900.0) -> None:
    """Blocking entrypoint. Spawns the engine process and serves HTTP.

    `engine_cmd` overrides the child command so the protocol can be exercised
    against a stub engine on a machine with no GPU.
    """
    asyncio.run(_split_serve(host, port, list(engine_argv or []), capacity,
                             max_len, tokenizer, engine_cmd, ready_timeout))
