"""The server-vs-server load generator. One client, four wire formats.

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

**Fairness.** Every server gets the same prompts as raw token-id arrays —
distinct random ids per request (llama.cpp and vLLM have prefix caching; braid
does not; fresh prompts keep that lever out of the comparison without turning
a competitor's default off), the same lengths, the same generation budget,
greedy sampling. braid's `/generate` takes `prompt_tokens`; llama.cpp's
`/completion` and vLLM's `/v1/completions` accept an int array `prompt`;
SGLang's `/generate` takes `input_ids`. All four bypass the server's tokenizer,
so no side pays or skips tokenization.

Token counting differs by wire format and is taken from each server's own
stream:

  * braid emits exactly one SSE event per token.
  * llama-server's final event carries `timings.predicted_n`.
  * vLLM's `include_usage` trailer carries `usage.completion_tokens`. That
    trailer is an event with an EMPTY `choices` array and no token in it —
    counting it would add a phantom token and a phantom ITL gap to every
    request, so it is skipped explicitly.
  * SGLang's events carry a cumulative `meta_info.completion_tokens`; the last
    one wins.

Event counting is the fallback everywhere. JSON parsing is gated behind a cheap
substring test rather than run per event — the client publishes `client_busy`
as a guilt figure and must not inflate it decoding its own instrument.

TTFT is first-event arrival; ITL is inter-event gaps within one stream.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import resource
import sys
import time
import urllib.request
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


def _body(server: str, prompt: list[int], max_new: int,
          model: str) -> tuple[str, bytes]:
    # EVERY ARM IGNORES EOS. The prompts are random token ids, so a real model
    # emits EOS early on them at wildly different rates per engine — and an
    # engine whose batch drains early then finishes at reduced concurrency,
    # which LOWERS its measured throughput. The 2026-08-11 recon hit exactly
    # this: braid generated its full budget (8,192 tokens at c=64) while vLLM
    # stopped at 7,443, so the two arms were not running the same experiment.
    # braid ignores EOS implicitly (`eos_token_id=None` never compares equal to
    # a token, scheduler.py), which is why only the competitors truncated. The
    # id is pinned explicitly here anyway: the contract is "no arm stops early",
    # and it should not depend on a default in another file staying put.
    if server == "braid":
        path = "/generate"
        body = {"prompt_tokens": prompt, "max_new_tokens": max_new,
                "temperature": 0.0, "eos_token_id": -1}
    elif server == "llama":
        path = "/completion"
        body = {"prompt": prompt, "n_predict": max_new, "stream": True,
                "temperature": 0.0, "cache_prompt": False, "ignore_eos": True}
    elif server == "vllm":
        path = "/v1/completions"
        body = {"model": model, "prompt": prompt, "max_tokens": max_new,
                "temperature": 0.0, "stream": True, "ignore_eos": True,
                "stream_options": {"include_usage": True}}
    else:
        path = "/generate"
        body = {"input_ids": prompt, "stream": True,
                "sampling_params": {"max_new_tokens": max_new,
                                    "temperature": 0.0, "ignore_eos": True}}
    return path, json.dumps(body).encode()


async def _one_request(host: str, port: int, server: str, prompt: list[int],
                       max_new: int, timeout: float,
                       model: str = "") -> ReqResult:
    r = ReqResult()
    try:
        reader, writer = await asyncio.open_connection(host, port)
    except OSError as e:
        r.error = f"connect: {e}"
        return r
    try:
        path, payload = _body(server, prompt, max_new, model)
        writer.write(
            f"POST {path} HTTP/1.1\r\nHost: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
            .encode() + payload)
        await writer.drain()

        t0 = time.perf_counter()
        last = t0
        predicted_n = None
        last_event = ""

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
            # vLLM's usage trailer is an event carrying no token: empty
            # `choices`, a `usage` block. Counting it would add one phantom
            # token and one phantom ITL gap to every single request. The
            # substring test keeps the parse off the per-token path.
            if server == "vllm" and '"usage"' in data:
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict) and obj.get("usage"):
                    n = obj["usage"].get("completion_tokens")
                    if n is not None:
                        predicted_n = n
                    if not obj.get("choices"):
                        continue
            now = time.perf_counter()
            if r.tokens == 0:
                r.ttft_ms = (now - t0) * 1e3
            else:
                r.itl_ms.append((now - last) * 1e3)
            last = now
            r.tokens += 1
            if server == "sglang":
                last_event = data
            if server == "llama" and '"stop":true' in data.replace(" ", ""):
                try:
                    predicted_n = json.loads(data).get("timings", {}).get("predicted_n")
                except json.JSONDecodeError:
                    pass
                break
        # SGLang's meta_info count is cumulative; the last event carries the
        # total. Parsed once per request, not once per token.
        if server == "sglang" and last_event:
            try:
                predicted_n = json.loads(last_event).get(
                    "meta_info", {}).get("completion_tokens")
            except json.JSONDecodeError:
                pass
        # A server's own count wins: llama's final stop event repeats no new
        # token, and vLLM/SGLang report the number they actually decoded.
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


async def _wave(host, port, server, prompts, max_new, c, timeout, model="",
                ramp_s: float = 0.010):
    """`c` streams, each consuming its share of `prompts` sequentially.

    Stream starts are RAMPED, 10 ms apart by default, not fired in one
    event-loop tick.
    A same-tick start is a SYN flood the kernel answers for the server: both
    braid's socketserver and llama-server's httplib default to a listen
    backlog of 5, and past it the client measures resets and TCP retransmit
    ladders (1s, 2s, 4s...) instead of inference — measured as a 289-SECOND
    TTFT p50 against llama at c=128 with the client provably idle. Real load
    does not arrive in one tick either, so the ramp is fairness, not charity;
    it applies identically to both servers, and TTFT is stamped per request
    after its own start, so the ramp never enters any latency number.

    `ramp_s=0` fires every stream in one tick. That is NOT a fair default —
    see the backlog story above — but it is the only way to compare an HTTP
    arm against `braid.bench.serve_bench`, which submits all `c` at once. The
    2026-08-11 phase-0 run could not attribute its −36.6% because the two
    harnesses differed in arrival pattern (TTFT p50 1,709 ms vs 171 ms) and
    nothing isolated it. Use 0 for that comparison only, on a server whose
    listen backlog is known to be large enough (braid's is 512).
    """
    per = len(prompts) // c
    results: list[ReqResult] = []

    async def stream(i: int):
        if ramp_s:
            await asyncio.sleep(i * ramp_s)
        for j in range(per):
            results.append(await _one_request(host, port, server,
                                              prompts[i * per + j], max_new,
                                              timeout, model))

    await asyncio.gather(*(stream(i) for i in range(c)))
    return results


def _health(url: str | None) -> dict | None:
    """One GET, best-effort. Used to bracket the MEASURED wave only.

    Sampling the server's counters from the outside (before/after the whole
    client invocation) folds the warmup wave into the window, and at c=64 the
    warmup is another 64 requests — enough turnaround to dominate an idle
    measurement. Bracketing here scopes the window to exactly the wave whose
    tok/s is reported. Failure is a None, never an exception: a server without
    the endpoint must not take the benchmark down.
    """
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


async def run_point(args, c: int) -> dict:
    ramp_s = getattr(args, "ramp_ms", 10.0) / 1000.0
    warm = _prompts(c, args.prompt_len, args.seed + 9999 + c)
    await _wave(args.host, args.port, args.server, warm, 8, c, args.timeout,
                args.model, ramp_s)

    prompts = _prompts(c * args.requests_per_stream, args.prompt_len,
                       args.seed + c)
    h0 = _health(getattr(args, "health_url", None))
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    results = await _wave(args.host, args.port, args.server, prompts,
                          args.max_new_tokens, c, args.timeout, args.model,
                          ramp_s)
    wall = time.perf_counter() - t0
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    h1 = _health(getattr(args, "health_url", None))
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
        # What `tokens` WOULD be if no request stopped early. A JSON field an
        # engine does not recognise is silently dropped, so `ignore_eos` in
        # _body cannot be trusted to have taken effect — it has to be observed.
        # tokens < tokens_expected on one arm and not another means the arms ran
        # different experiments, which is the defect this pair of fields exists
        # to make impossible to publish by accident.
        "tokens_expected": len(ok) * args.max_new_tokens,
        "tok_s": round(toks / wall, 1) if wall else 0.0,
        "ttft_ms_p50": round(_pct(ttfts, 0.50), 1),
        "ttft_ms_p90": round(_pct(ttfts, 0.90), 1),
        "itl_ms_p50": round(_pct(itls, 0.50), 2),
        "itl_ms_p99": round(_pct(itls, 0.99), 2),
        "client_busy": round(busy, 3),
        # ~1.0 means the single-threaded client saturated: the point measures
        # the client, not the server, and must not be published as the server.
        "client_bound": busy > 0.85,
        **_tick_split(h0, h1),
    }


def _tick_split(h0: dict | None, h1: dict | None) -> dict:
    """Server-side per-tick split over the measured wave, or {} if unavailable.

    `idle_pct` is the share of the window in which the scheduler had nothing to
    run. That is the number that separates "the serving layer is slow" from
    "the load pattern never filled the batch" — the question phase 0 left open.
    """
    if not h0 or not h1 or "ticks" not in h1:
        return {}
    dt = h1["ticks"] - h0["ticks"]
    if dt <= 0:
        return {"srv_ticks": dt}
    d = {k: h1.get(k, 0.0) - h0.get(k, 0.0)
         for k in ("step_s", "fanout_s", "busy_s", "idle_s")}
    window = d["busy_s"] + d["idle_s"]
    return {
        "srv_ticks": dt,
        "srv_step_ms": round(d["step_s"] / dt * 1e3, 3),
        "srv_fanout_ms": round(d["fanout_s"] / dt * 1e3, 3),
        "srv_other_ms": round(
            (d["busy_s"] - d["step_s"] - d["fanout_s"]) / dt * 1e3, 3),
        "srv_busy_s": round(d["busy_s"], 4),
        "srv_idle_s": round(d["idle_s"], 4),
        "srv_idle_pct": round(d["idle_s"] / window * 100, 1) if window else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", choices=["braid", "llama", "vllm", "sglang"],
                   required=True)
    p.add_argument("--model", default="recon",
                   help="served model name; vLLM's OpenAI route requires the "
                        "request to name it. Launch vLLM with "
                        "--served-model-name so this is deterministic.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=[1, 8, 16, 32, 64, 128])
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--requests-per-stream", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ramp-ms", type=float, default=10.0,
                   help="delay between stream starts. 10 is the fair default "
                        "(a same-tick start overruns a small listen backlog "
                        "and measures TCP, not inference); 0 matches "
                        "serve_bench's all-at-once submission and exists only "
                        "for that comparison.")
    p.add_argument("--health-url", default=None,
                   help="if set, GET it immediately before and after the "
                        "MEASURED wave and fold the server's own per-tick "
                        "counters into the row. braid: "
                        "http://127.0.0.1:<port>/health")
    args = p.parse_args()

    for c in args.concurrency:
        point = asyncio.run(run_point(args, c))
        print(json.dumps(point), flush=True)
        if point["errors"]:
            print(f"  {point['errors']} errors at c={c}: "
                  f"{point['error_sample']}", file=sys.stderr)


if __name__ == "__main__":
    main()
