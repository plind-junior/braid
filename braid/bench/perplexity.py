"""Perplexity against an HF reference on a pinned corpus. ROADMAP Phase 2 item 4.

**This measures correctness, not speed.** No timing is reported here and none of
`benchmark-remote-5090`'s clock/noise-floor machinery applies: perplexity on a
pinned token list is deterministic, so the run either reproduces bit-for-bit or
something changed. What it *does* inherit is the publishing contract — every
number lands in `docs/runbooks/perplexity.md` with its method, date and commit.

Method, stated so the absolute value means something:

  corpus     wikitext-2-raw-v1 **test** split, documents joined with "\\n\\n",
             tokenized with the checkpoint's own tokenizer, first N tokens taken.
             The token list is pinned by SHA-256 — a corpus that silently changed
             would move the perplexity and look like a regression.
  windows    non-overlapping, `window` tokens each, each a fresh forward with no
             carried state. Position 0 of each window is not predicted.
  reduction  total NLL over all predicted positions / count, exponentiated.
             Token-level, natural log, no length normalisation beyond the mean.

Non-overlapping windows are the cheap convention, and they understate absolute
perplexity relative to a stride-1 sliding window because most tokens are
predicted from a short prefix. That does not matter here: the gate is braid
against HF **on the identical protocol**, and the absolute number is recorded
rather than compared to anyone else's leaderboard.
"""
from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))
CORPUS_REPO = "Salesforce/wikitext"
CORPUS_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


@dataclass(frozen=True)
class Corpus:
    ids: torch.Tensor          # [n_tokens], int64
    sha256: str
    n_tokens: int
    source: str

    def describe(self) -> str:
        return (f"{self.source}: {self.n_tokens} tokens, sha256={self.sha256[:16]}")


def load_corpus(
    n_tokens: int = 16384,
    model_dir: Path = DEFAULT_MODEL_DIR,
    cache_dir: str | Path = "/root/corpus",
) -> Corpus:
    """Download, tokenize and pin the first `n_tokens` of wikitext-2 test."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(repo_id=CORPUS_REPO, filename=CORPUS_FILE,
                           repo_type="dataset", local_dir=str(cache_dir))
    text = "\n\n".join(t for t in pq.read_table(path)["text"].to_pylist() if t.strip())

    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    # `tokenizers` caps a single encode; chunk the raw text, not the token stream.
    ids: list[int] = []
    for i in range(0, len(text), 200_000):
        ids.extend(tok.encode(text[i:i + 200_000]).ids)
        if len(ids) >= n_tokens:
            break
    if len(ids) < n_tokens:
        raise ValueError(f"corpus yielded {len(ids)} tokens, need {n_tokens}")
    ids = ids[:n_tokens]

    digest = hashlib.sha256(
        torch.tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest()
    return Corpus(ids=torch.tensor(ids, dtype=torch.int64), sha256=digest,
                  n_tokens=n_tokens, source=f"{CORPUS_REPO}/{CORPUS_FILE}")


@dataclass
class PPLResult:
    perplexity: float
    nll: float
    predicted_tokens: int
    windows: int
    window: int

    def __str__(self) -> str:
        return (f"ppl={self.perplexity:.4f} nll={self.nll:.6f} "
                f"over {self.predicted_tokens} tokens in {self.windows}x{self.window}")


def _window_nll(
    hidden: torch.Tensor,       # [1, T, hidden], post-final-norm
    lm_head: torch.Tensor,      # [vocab, hidden]
    targets: torch.Tensor,      # [T]
    slice_size: int = 256,
) -> tuple[float, int]:
    """Sum of -log p(target[t] | ..<t) over t in 1..T-1.

    The head is applied in slices so peak memory is `slice_size x vocab` rather
    than `T x vocab`; at T=2048 that is 250 MB instead of 2 GB in fp32. Cross
    entropy is taken in fp32 regardless of the activation dtype — a bf16
    log-softmax over 248,320 logits throws away most of the precision the
    comparison is trying to measure.
    """
    total, count = 0.0, 0
    T = hidden.shape[1]
    for s in range(0, T - 1, slice_size):
        e = min(s + slice_size, T - 1)
        logits = torch.nn.functional.linear(hidden[0, s:e], lm_head).float()
        total += torch.nn.functional.cross_entropy(
            logits, targets[s + 1:e + 1], reduction="sum").item()
        count += e - s
    return total, count


def perplexity(
    hidden_fn: Callable[[torch.Tensor], torch.Tensor],
    lm_head: torch.Tensor,
    corpus: Corpus,
    window: int = 2048,
    device: str | torch.device = "cuda",
    progress: bool = False,
) -> PPLResult:
    """`hidden_fn(ids[1, T]) -> [1, T, hidden]`, post-final-norm."""
    ids = corpus.ids.to(device)
    n_windows = corpus.n_tokens // window
    if n_windows == 0:
        raise ValueError(f"corpus of {corpus.n_tokens} is shorter than window {window}")

    total, count = 0.0, 0
    for w in range(n_windows):
        chunk = ids[w * window:(w + 1) * window]
        nll, n = _window_nll(hidden_fn(chunk[None]), lm_head, chunk)
        total += nll
        count += n
        if progress:
            print(f"    window {w + 1}/{n_windows}: running ppl "
                  f"{math.exp(total / count):.4f}", flush=True)

    return PPLResult(perplexity=math.exp(total / count), nll=total / count,
                     predicted_tokens=count, windows=n_windows, window=window)


# --- the two arms -------------------------------------------------------------

def braid_perplexity(corpus: Corpus, model_dir: Path = DEFAULT_MODEL_DIR,
                     dtype: torch.dtype = torch.bfloat16, window: int = 2048,
                     progress: bool = False, drop_final_norm_offset: bool = False,
                     quant_mlp: bool = False):
    """`drop_final_norm_offset` is the deliberate-bug arm: it undoes the `1+W`
    fold on the final norm only, which the roadmap predicts costs ~2x perplexity.
    It exists so the gate is shown to detect the thing it is for."""
    from braid.model.engine import Engine
    from braid.model.loader import load_checkpoint

    ck = load_checkpoint(model_dir, device="cuda", dtype=dtype)
    if drop_final_norm_offset:
        ck.tensors["norm"] = ck.tensors["norm"] - 1.0
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=dtype,
                                 quant_mlp=quant_mlp)
    res = perplexity(lambda ids: eng.hidden_states(ids, cache=None, last_only=False),
                     eng.lm_head, corpus, window=window, progress=progress)
    return res, eng


def hf_perplexity(corpus: Corpus, model_dir: Path = DEFAULT_MODEL_DIR,
                  dtype: torch.dtype = torch.bfloat16, window: int = 2048,
                  progress: bool = False):
    """The reference arm, built on `meta` and loaded with `assign=True`.

    Not negotiable: `module.to(bfloat16)` before `load_state_dict` copies into
    the bf16 parameter and truncates the tensors this checkpoint stores as F32,
    which makes the "reference" a measurably worse model than braid.
    """
    import json

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    tcfg = AutoConfig.from_pretrained(model_dir).get_text_config()
    tcfg._attn_implementation = "sdpa"

    wm = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    prefix = "model.language_model."
    by_shard: dict[str, list[str]] = {}
    for k in wm:
        if k.startswith(prefix):
            by_shard.setdefault(wm[k], []).append(k)
    sd = {}
    for shard, ks in by_shard.items():
        with safe_open(Path(model_dir) / shard, framework="pt") as f:
            for k in ks:
                t = f.get_tensor(k)
                sd[k[len(prefix):]] = t.to(
                    "cuda", torch.float32 if t.dtype == torch.float32 else dtype)

    with torch.device("meta"):
        model = hf.Qwen3_5TextModel(tcfg)
    model.load_state_dict(sd, assign=True)
    model.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(tcfg, device="cuda")
    model.eval()

    # The head is tied on Qwen3.5-4B and **not** on Qwen3.5-9B / Qwen3.6-27B,
    # where it ships as a top-level `lm_head.weight` outside the text tower —
    # so the `model.language_model.` filter above never collects it. Reading
    # `embed_tokens` unconditionally is right for the tied case and a *fluent*
    # disaster on an untied one: it decodes through the input embedding matrix,
    # which is a real matrix of the right shape and utter nonsense as a head.
    # Measured on the 9B before this branch existed: reference perplexity
    # **931,600** against braid's 7.13 — which presents as braid being broken,
    # when the reference was.
    if tcfg.tie_word_embeddings:
        lm_head = sd["embed_tokens.weight"]
    else:
        with safe_open(Path(model_dir) / wm["lm_head.weight"], framework="pt") as f:
            t = f.get_tensor("lm_head.weight")
        lm_head = t.to("cuda", torch.float32 if t.dtype == torch.float32 else dtype)

    @torch.no_grad()
    def hidden_fn(ids):
        return model(input_ids=ids, use_cache=False).last_hidden_state

    res = perplexity(hidden_fn, lm_head, corpus, window=window, progress=progress)
    return res, model


def main() -> None:
    import argparse
    import gc

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--ablate-final-norm", action="store_true",
                   help="also run braid with the final norm's 1+W offset removed")
    p.add_argument("--quant-mlp", action="store_true",
                   help="also run braid with FP8 W8A8 on the MLP projections")
    p.add_argument("--skip-hf", action="store_true",
                   help="skip the HF arm (it is the slow one and does not move)")
    args = p.parse_args()

    corpus = load_corpus(args.tokens, args.model_dir)
    print(f"corpus: {corpus.describe()}")
    print(f"windows: {corpus.n_tokens // args.window} x {args.window}\n")

    hf_res = None
    if not args.skip_hf:
        hf_res, hf_model = hf_perplexity(corpus, args.model_dir, window=args.window,
                                         progress=True)
        print(f"  HF    {hf_res}")
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

    br_res, eng = braid_perplexity(corpus, args.model_dir, window=args.window,
                                   progress=True)
    print(f"  braid {br_res}")
    del eng
    gc.collect()
    torch.cuda.empty_cache()

    if hf_res is not None:
        delta = abs(br_res.perplexity - hf_res.perplexity) / hf_res.perplexity
        print(f"\n  braid {br_res.perplexity:.4f} vs HF {hf_res.perplexity:.4f}  "
              f"-> {delta * 100:.4f}% (gate: within 20%)")

    if args.quant_mlp:
        # The comparison that matters is against braid-bf16 on the SAME corpus
        # and window, not against HF: this isolates what fp8 costs from what
        # braid already differs by.
        q, eng3 = braid_perplexity(corpus, args.model_dir, window=args.window,
                                   progress=True, quant_mlp=True)
        n_q = sum(1 for l in eng3.layers if l.mlp.quantized)
        rel = (q.perplexity - br_res.perplexity) / br_res.perplexity
        print(f"\n  FP8 W8A8 on MLP ({n_q}/{len(eng3.layers)} layers quantized)")
        print(f"    braid bf16 {br_res.perplexity:.4f} -> fp8 {q.perplexity:.4f}"
              f"   {rel * 100:+.2f}%")
        del eng3
        gc.collect()
        torch.cuda.empty_cache()

    if args.ablate_final_norm:
        ab, eng2 = braid_perplexity(corpus, args.model_dir, window=args.window,
                                    drop_final_norm_offset=True)
        ratio = ab.perplexity / hf_res.perplexity
        print(f"\n  ABLATION, final norm 1+W removed: ppl {ab.perplexity:.4f} "
              f"= {ratio:.2f}x the reference")
        del eng2


if __name__ == "__main__":
    main()
