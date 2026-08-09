"""The server-vs-server load generator. One client, two wire formats.

**The client is the instrument here, and the instrument can lie.** The roadmap
records a threaded Python client inflating c=64 TTFT by 19x — the load
generator saturated its own GIL and published the queueing delay of the client
as if it were the server's. So this one is built to keep itself out of the
measurement, and to *say so* when it cannot:

  * Single-threaded asyncio over stdlib sockets. No thread pool, no requests/
    aiohttp dependency on a box provisioned for torch, no per-token thread
    hand-off. One coroutine per stream, one connection per request.
  * **The client publishes its own guilt figure.** Each point reports
    `client_busy` = process CPU time / wall clock. A client at 0.3 cores is an
    observer; a client at ~1.0 saturated its event loop and the point is
    labelled `client_bound: true` rather than silently published. That flag in
    the output is the difference between "braid's HTTP layer tops out at X"
    and "the measurement topped out at X".
  * Closed loop, like `serve_bench`: `c` streams, each running
    `--requests-per-stream` sequential requests, plus one discarded warmup
    wave. Aggregate tok/s = generated tokens / wall across the timed wave.

**Fairness.** Both servers get the same prompts as raw token-id arrays —
distinct random ids per request (llama.cpp has prefix caching; braid does not;
fresh prompts keep that lever out of the comparison), the same lengths, the
same generation budget, greedy sampling. braid's `/generate` takes
`prompt_tokens`; llama.cpp's `/completion` accepts an int array `prompt`,
which also bypasses its tokenizer so neither side pays or skips tokenization.

Token counting differs by wire format and is taken from each server's own
stream: braid emits exactly one SSE event per token; llama-server's final
event carries `timings.predicted_n`, which is used when present (event
counting is the fallback). TTFT is first-event arrival; ITL is inter-event
gaps within one stream.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import resource
import sys
import time
from dataclasses import dataclass, field


@dataclass
class ReqResult:
    tokens: int = 0
    ttft_ms: float = 0.0
    itl_ms: list[float] = field(default_factory=list)
    error: str = ""


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def _prompts(n: int, length: int, seed: int) -> list[list[int]]:
    """Distinct pseudo-random token ids, no torch needed client-side."""
    import random

    rng = random.Random(seed)
    return [[rng.randrange(100, 100_000) for _ in range(length)] for _ in range(n)]


def _body(server: str, prompt: list[int], max_new: int) -> tuple[str, bytes]:
    if server == "braid":
        path = "/generate"
        body = {"prompt_tokens": prompt, "max_new_tokens": max_new,
                "temperature": 0.0}
    else:
        path = "/completion"
        body = {"prompt": prompt, "n_predict": max_new, "stream": True,
                "temperature": 0.0, "cache_prompt": False}
    return path, json.dumps(body).encode()


async def _one_request(host: str, port: int, server: str, prompt: list[int],
                       max_new: int, timeout: float) -> ReqResult:
    r = ReqResult()
    try:
        reader, writer = await asyncio.open_connection(host, port)
    except OSError as e:
        r.error = f"connect: {e}"
        return r
    try:
        path, payload = _body(server, prompt, max_new)
        writer.write(
            f"POST {path} HTTP/1.1\r\nHost: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
            .encode() + payload)
        await writer.drain()

        t0 = time.perf_counter()
        last = t0
        predicted_n = None

        # Status line + headers first. llama-server streams with
        # `Transfer-Encoding: chunked`; a naive line reader would see SSE
        # events split by chunk framing — the token count would survive by
        # luck, but a split `"stop":true` or timings line would not. So the
        # framing is decoded properly and SSE lines are re-assembled from the
        # de-chunked byte stream.
        status = (await asyncio.wait_for(reader.readline(), timeout)
                  ).decode("utf-8", "replace").strip()
        if "200" not in status:
            r.error = f"http: {status}"
            return r
        chunked = False
        while True:
            h = (await asyncio.wait_for(reader.readline(), timeout)
                 ).decode("utf-8", "replace").strip()
            if not h:
                break
            if h.lower().replace(" ", "") == "transfer-encoding:chunked":
                chunked = True

        async def payload_lines():
            """SSE lines from the (possibly chunked) body, split on newline."""
            buf = b""
            while True:
                if chunked:
                    size_line = await asyncio.wait_for(reader.readline(), timeout)
                    if not size_line:
                        break
                    try:
                        n = int(size_line.strip().split(b";")[0], 16)
                    except ValueError:
                        break
                    if n == 0:
                        break
                    piece = await asyncio.wait_for(reader.readexactly(n + 2),
                                                   timeout)
                    buf += piece[:-2]     # strip the chunk's trailing CRLF
                else:
                    piece = await asyncio.wait_for(reader.read(65536), timeout)
                    if not piece:
                        break
                    buf += piece
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    yield line.decode("utf-8", "replace").strip()

        async for s in payload_lines():
            if not s.startswith("data:"):
                continue
            data = s[5:].strip()
            if data == "[DONE]":
                break
            now = time.perf_counter()
            if r.tokens == 0:
                r.ttft_ms = (now - t0) * 1e3
            else:
                r.itl_ms.append((now - last) * 1e3)
            last = now
            r.tokens += 1
            if server == "llama" and '"stop":true' in data.replace(" ", ""):
                try:
                    predicted_n = json.loads(data).get("timings", {}).get("predicted_n")
                except json.JSONDecodeError:
                    pass
                break
        # llama's final stop event repeats no new token; its own count wins.
        if predicted_n is not None:
            r.tokens = int(predicted_n)
    except (asyncio.TimeoutError, OSError) as e:
        r.error = f"{type(e).__name__}: {e}"
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
    return r


async def _wave(host, port, server, prompts, max_new, c, timeout):
    """`c` streams, each consuming its share of `prompts` sequentially.

    Stream starts are RAMPED, 10 ms apart, not fired in one event-loop tick.
    A same-tick start is a SYN flood the kernel answers for the server: both
    braid's socketserver and llama-server's httplib default to a listen
    backlog of 5, and past it the client measures resets and TCP retransmit
    ladders (1s, 2s, 4s...) instead of inference — measured as a 289-SECOND
    TTFT p50 against llama at c=128 with the client provably idle. Real load
    does not arrive in one tick either, so the ramp is fairness, not charity;
    it applies identically to both servers, and TTFT is stamped per request
    after its own start, so the ramp never enters any latency number.
    """
    per = len(prompts) // c
    results: list[ReqResult] = []

    async def stream(i: int):
        await asyncio.sleep(i * 0.010)
        for j in range(per):
            results.append(await _one_request(host, port, server,
                                              prompts[i * per + j], max_new,
                                              timeout))

    await asyncio.gather(*(stream(i) for i in range(c)))
    return results


async def run_point(args, c: int) -> dict:
    warm = _prompts(c, args.prompt_len, args.seed + 9999 + c)
    await _wave(args.host, args.port, args.server, warm, 8, c, args.timeout)

    prompts = _prompts(c * args.requests_per_stream, args.prompt_len,
                       args.seed + c)
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    results = await _wave(args.host, args.port, args.server, prompts,
                          args.max_new_tokens, c, args.timeout)
    wall = time.perf_counter() - t0
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    busy = ((cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)) / wall

    ok = [r for r in results if not r.error]
    errors = [r.error for r in results if r.error]
    toks = sum(r.tokens for r in ok)
    ttfts = [r.ttft_ms for r in ok if r.tokens]
    itls = [x for r in ok for x in r.itl_ms]
    return {
        "server": args.server, "concurrency": c, "requests": len(results),
        "errors": len(errors), "error_sample": errors[:2],
        "tokens": toks, "wall_s": round(wall, 3),
        "tok_s": round(toks / wall, 1) if wall else 0.0,
        "ttft_ms_p50": round(_pct(ttfts, 0.50), 1),
        "ttft_ms_p90": round(_pct(ttfts, 0.90), 1),
        "itl_ms_p50": round(_pct(itls, 0.50), 2),
        "itl_ms_p99": round(_pct(itls, 0.99), 2),
        "client_busy": round(busy, 3),
        # ~1.0 means the single-threaded client saturated: the point measures
        # the client, not the server, and must not be published as the server.
        "client_bound": busy > 0.85,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", choices=["braid", "llama"], required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=[1, 8, 16, 32, 64, 128])
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--requests-per-stream", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    for c in args.concurrency:
        point = asyncio.run(run_point(args, c))
        print(json.dumps(point), flush=True)
        if point["errors"]:
            print(f"  {point['errors']} errors at c={c}: "
                  f"{point['error_sample']}", file=sys.stderr)


if __name__ == "__main__":
    main()
