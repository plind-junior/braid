"""SSE server and slot release on disconnect. ROADMAP Phase 4 item 1.

The roadmap names the failure this file is for: *"a client disconnecting
mid-generation must release **both** its KV blocks and its recurrent slot — the
reference engine leaked slots on exactly this path until it added a dedicated
reset."*

A leak here has no symptom until the pool is full, at which point the server
stops accepting work with no error raised anywhere. So the tests assert the
slot count directly rather than inferring health from a successful response,
and they hang up **mid-stream** rather than after completion — after completion
the slot comes back through the normal path and proves nothing.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

from braid.model.engine import Engine
from braid.model.loader import load_checkpoint
from braid.serve.scheduler import Request
from braid.serve.server import EngineService, serve

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

CAPACITY, MAX_LEN = 4, 1024
# Long enough that natural completion CANNOT be mistaken for a working
# disconnect path. A 64-token request finishes in ~2 s, well inside any
# reasonable wait, so a leak test built on one passes whether or not `cancel`
# does anything. These run ~900 steps -- tens of seconds -- and the release
# assertions are bounded at RELEASE_BUDGET, an order below that.
LONG = 900
RELEASE_BUDGET = 5.0


@pytest.fixture(scope="module")
def engine():
    ck = load_checkpoint(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    return Engine.from_checkpoint(ck, device="cuda", dtype=torch.bfloat16,
                                  use_kernels=True)


@pytest.fixture
def service(engine):
    svc = EngineService(engine, capacity=CAPACITY, max_len=MAX_LEN, graphed=False)
    yield svc
    svc.shutdown()


@pytest.fixture
def server(engine):
    httpd, svc = serve(engine, host="127.0.0.1", port=0, capacity=CAPACITY,
                       max_len=MAX_LEN, graphed=False)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd, svc, httpd.server_address[1]
    httpd.shutdown()
    svc.shutdown()


def _prompt(n=8, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1000, (n,), generator=g).tolist()


def _wait(pred, timeout=30.0, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


# --- the service ---------------------------------------------------------------

def test_a_stream_yields_tokens_then_a_sentinel(service):
    req = Request(prompt=_prompt(), max_new_tokens=5)
    q = service.submit(req)
    got = []
    while True:
        tok = q.get(timeout=60)
        if tok is None:
            break
        got.append(tok)
    assert len(got) == 5
    _wait(lambda: service.scheduler.free_slots == CAPACITY, what="slot release")


def test_closing_midstream_releases_the_slot(service):
    """The leak path. Hang up early, with ~900 tokens still to come."""
    req = Request(prompt=_prompt(), max_new_tokens=LONG)
    q = service.submit(req)
    assert q.get(timeout=60) is not None
    assert service.scheduler.free_slots == CAPACITY - 1

    t0 = time.time()
    service.close(req.id)
    _wait(lambda: service.scheduler.free_slots == CAPACITY,
          timeout=RELEASE_BUDGET, what="slot release after mid-stream close")
    elapsed = time.time() - t0
    assert elapsed < RELEASE_BUDGET, (
        f"slot came back after {elapsed:.1f}s — that is natural completion, "
        f"not `cancel`")
    assert service.scheduler.live_ids == []


def test_close_is_idempotent_and_safe_after_completion(service):
    """A client disconnecting on the last token races normal completion."""
    req = Request(prompt=_prompt(), max_new_tokens=3)
    q = service.submit(req)
    while q.get(timeout=60) is not None:
        pass
    service.close(req.id)
    service.close(req.id)
    _wait(lambda: service.scheduler.free_slots == CAPACITY, what="slot release")


def test_the_pool_survives_many_abandoned_streams(service):
    """A leak of one slot per disconnect wedges the server; this is the test
    that fails loudly instead of the server going quiet."""
    for _ in range(CAPACITY * 3):
        req = Request(prompt=_prompt(), max_new_tokens=LONG)
        q = service.submit(req)
        assert q.get(timeout=60) is not None
        service.close(req.id)
        _wait(lambda: service.scheduler.free_slots == CAPACITY,
              timeout=RELEASE_BUDGET, what="release")
    assert service.scheduler.free_slots == CAPACITY


def test_concurrent_streams_each_get_their_own_tokens(service):
    """Every CUDA call is on one thread; the queues must not cross."""
    reqs = [Request(prompt=_prompt(6 + i, seed=10 + i), max_new_tokens=6)
            for i in range(CAPACITY)]
    qs = [service.submit(r) for r in reqs]
    outs = []
    for q in qs:
        got = []
        while True:
            tok = q.get(timeout=60)
            if tok is None:
                break
            got.append(tok)
        outs.append(got)
    assert all(len(o) == 6 for o in outs)
    _wait(lambda: service.scheduler.free_slots == CAPACITY, what="slot release")


# --- over HTTP ------------------------------------------------------------------

def test_health(server):
    _, _, port = server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30) as r:
        body = json.loads(r.read())
    assert body["ok"] and body["capacity"] == CAPACITY


def test_sse_stream_ends_with_done(server):
    _, svc, port = server
    body = json.dumps({"prompt_tokens": _prompt(), "max_new_tokens": 4}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data: "):
                events.append(line[6:])
    assert events[-1] == "[DONE]"
    assert len(events) == 5                       # 4 tokens + [DONE]
    assert all("token" in json.loads(e) for e in events[:-1])
    _wait(lambda: svc.scheduler.free_slots == CAPACITY, what="slot release")


def test_an_http_client_that_hangs_up_releases_its_slot(server):
    """The real disconnect: kill the socket mid-stream, not via `close()`."""
    _, svc, port = server
    body = json.dumps({"prompt_tokens": _prompt(),
                       "max_new_tokens": LONG}).encode()

    sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    sock.sendall(
        b"POST /generate HTTP/1.1\r\nHost: localhost\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    # Read until at least one token has actually been streamed.
    seen = b""
    while b"token" not in seen:
        chunk = sock.recv(4096)
        assert chunk, "server closed before streaming anything"
        seen += chunk
    assert svc.scheduler.free_slots == CAPACITY - 1

    t0 = time.time()
    sock.close()          # hang up hard, ~900 tokens from the end
    _wait(lambda: svc.scheduler.free_slots == CAPACITY, timeout=RELEASE_BUDGET,
          what="slot release after an HTTP disconnect")
    elapsed = time.time() - t0
    assert elapsed < RELEASE_BUDGET, (
        f"slot came back after {elapsed:.1f}s — natural completion, not the "
        f"disconnect path")


def test_bad_requests_are_refused_without_taking_a_slot(server):
    _, svc, port = server
    for body in (b'{"max_new_tokens": 4}', b'{"prompt_tokens": [], "max_new_tokens": 4}',
                 b'not json'):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=30)
        assert e.value.code == 400
    assert svc.scheduler.free_slots == CAPACITY
