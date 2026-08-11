"""Wire format between the HTTP front end and the engine process (phase 3).

**Why there is a wire at all.** Phases 0-2 (2026-08-11, `docs/runbooks/vllm-recon.md`)
localised braid's entire deficit against vLLM-FP8 to the serving layer: the
engine in-process does 3,322 tok/s at c=64 against vLLM's 3,164, and the same
engine behind braid's HTTP server did 2,283. Phase 2 replaced 64 handler
threads with one event loop and recovered 9.3%, leaving `step()` at 25.06 ms
per tick against the ~19 ms the in-process arm implies. What is left is that
serving Python and engine Python still share one GIL, because they are two
threads in one process. Two threads is better than 65 and it is still one
interpreter. So: two processes, which is what vLLM does — its own logs on this
box read `(EngineCore pid=2128)`, a different pid from its API server.

**Design constraints this format is answering.**

* The GPU thread encodes a tick and must not be made to think about it. One
  `sendall` per tick of a few hundred bytes, no pickle, no allocation storm.
* The front end must stay importable without torch, so nothing here may reach
  into the model. Updates arrive as plain tuples.
* A frame must be self-delimiting over a stream socket. Length-prefixed, and
  the reader is buffered, because a UDS will happily split a 700-byte write.

**Frame:** ``<u32 BE length><u8 kind><payload>``, where `length` counts the kind
byte and the payload. Control traffic (submit, stats) is JSON — it is per
request or 50 Hz, not per token, and legibility in a packet dump is worth more
than the microseconds. The tick path is hand-packed binary.

Ids stay strings on the wire rather than becoming integer handles. Handles were
the first design; they were dropped because they buy ~20 us per tick against a
25 ms tick (0.08%) and cost the front end the ability to reuse `Bridge` and
`make_handler` unchanged. Optimising an unmeasured 0.08% by making the two
serving paths diverge is the wrong trade.
"""
from __future__ import annotations

import json
import select
import socket
import struct

# front end -> engine
SUBMIT = 0x01
CANCEL = 0x02
SHUTDOWN = 0x03
# engine -> front end
READY = 0x11
TICK = 0x12
STATS = 0x13

_HDR = struct.Struct(">I")
_U16 = struct.Struct(">H")
_UP = struct.Struct(">BH")          # finished flag, token count

MAX_FRAME = 16 * 1024 * 1024


def frame(kind: int, payload: bytes = b"") -> bytes:
    return _HDR.pack(len(payload) + 1) + bytes((kind,)) + payload


def json_frame(kind: int, obj) -> bytes:
    return frame(kind, json.dumps(obj).encode())


def encode_tick(ups) -> bytes:
    """`list[Update]` -> one frame. Runs on the GPU thread; keep it flat."""
    parts = [_U16.pack(len(ups))]
    for up in ups:
        rid = up.id.encode()
        toks = up.tokens
        parts.append(bytes((len(rid),)))
        parts.append(rid)
        parts.append(_UP.pack(1 if up.finished else 0, len(toks)))
        if toks:
            parts.append(struct.pack(f">{len(toks)}i", *toks))
    return frame(TICK, b"".join(parts))


def decode_tick(buf: bytes) -> list[tuple[str, list[int], bool]]:
    """One frame payload -> `[(req_id, tokens, finished), ...]`."""
    out: list[tuple[str, list[int], bool]] = []
    (n,) = _U16.unpack_from(buf, 0)
    off = 2
    for _ in range(n):
        ln = buf[off]
        off += 1
        rid = buf[off:off + ln].decode()
        off += ln
        fin, nt = _UP.unpack_from(buf, off)
        off += 3
        if nt:
            toks = list(struct.unpack_from(f">{nt}i", buf, off))
            off += 4 * nt
        else:
            toks = []
        out.append((rid, toks, bool(fin)))
    return out


class FrameReader:
    """Buffered frame reader over a BLOCKING socket.

    The socket is deliberately left in blocking mode and the timeout is done
    with `select` instead of `socket.settimeout`. A timeout on the socket
    itself applies to `sendall` too, and a `sendall` that times out half way
    through a frame corrupts the stream with no error the peer can recover
    from. This is the engine process's only reader and its writer runs on the
    GPU thread, so that hazard is real rather than theoretical.
    """

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buf = bytearray()

    def next(self, timeout: float | None = None
             ) -> tuple[int, bytes] | None:
        """`(kind, payload)`, or `None` if `timeout` elapsed with no frame.

        Raises `ConnectionError` on a clean EOF — the peer is gone, and for
        both processes here that is fatal rather than something to retry.
        """
        while True:
            if len(self.buf) >= 4:
                (n,) = _HDR.unpack_from(self.buf, 0)
                if n > MAX_FRAME:
                    raise ConnectionError(f"frame of {n} bytes: stream lost")
                if len(self.buf) >= 4 + n:
                    p = bytes(self.buf[4:4 + n])
                    del self.buf[:4 + n]
                    return p[0], p[1:]
            if timeout is not None:
                r, _, _ = select.select([self.sock], [], [], timeout)
                if not r:
                    return None
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("peer closed the engine socket")
            self.buf += chunk
