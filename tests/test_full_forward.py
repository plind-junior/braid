"""The whole 32-layer hybrid against HF, plus proof of life. ROADMAP Phase 2 item 3.

Four things are checked, in increasing order of what they'd catch:

  1. **One GDN layer** vs `Qwen3_5GatedDeltaNet` — the conv, the [Q|K|V] split,
     the gates, the scan, the gated norm.
  2. **The full stack** vs `Qwen3_5TextModel` — layer schedule, residual order,
     rope, the final norm and the tied head. A single mixer wired to the wrong
     layer index survives (1) and dies here.
  3. **Decode == prefill.** T tokens fed one at a time through the caches must
     reproduce the single T-token prefill. This is the cache test: the conv
     window, the recurrent state and the KV buffer all have to be right, and
     it is the standard "generation drifts after the first token" bug.
  4. **Proof of life** — `"The capital of France is"` -> `" Paris"`.

**The two full-stack fp32 arms run on the CPU, and that is not a compromise.**
Two fp32 copies of the 9B are 72 GiB — no card here holds them, and truncating
the stack would retire exactly the claim these two tests exist to make, that
every one of the 32 layers is wired to the right mixer. The box has 245 GiB of
host RAM and 64 cores, a 16-token fp32 forward costs ~1.5 s per arm, and fp32 on
CPU is if anything the more faithful reference. The *batched* fp32 gates
elsewhere in the suite stay on the GPU and truncate (`conftest.fp32_engine`),
because what they assert is a property of the GPU batching itself.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
from conftest import assert_greedy, cuda_reclaim, weight_bytes

from braid.model.config import ModelConfig
from braid.model.engine import Engine
from braid.model.gdn import GatedDeltaNet
from braid.model.loader import load_checkpoint

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

MODEL_DIR = Path(os.environ.get("BRAID_MODEL_DIR", "/root/models/Qwen3.5-4B"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU"),
    pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"no checkpoint at {MODEL_DIR}"),
]

REL_L2_MAX = 5e-3
COSINE_MIN = 0.99999
GDN_LAYER = 0
DTYPE = torch.bfloat16


def _metrics(mine, ref):
    a, b = mine.double().flatten(), ref.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a @ b / (a.norm() * b.norm())).item()


def _assert_parity(mine, ref, what, rel_max=REL_L2_MAX, cos_min=COSINE_MIN):
    r, c = _metrics(mine, ref)
    assert r <= rel_max and c >= cos_min, (
        f"{what}: rel_l2={r:.3e} (max {rel_max:.1e}), cosine={c:.9f} (min {cos_min})"
    )
    return r, c


# `_assert_greedy` moved to `conftest.assert_greedy` once a second module
# needed it (`test_graph_decode`, the kv_len arm). The reasoning is long and
# load-bearing enough that two copies of it would drift.
_assert_greedy = assert_greedy


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def hf_text_config():
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(MODEL_DIR).get_text_config()
    c._attn_implementation = "sdpa"
    return c


@pytest.fixture(scope="module")
def engine():
    ck = load_checkpoint(MODEL_DIR, device="cuda")
    eng = Engine.from_checkpoint(ck, device="cuda", dtype=DTYPE)
    yield eng


# --- 1. one GDN layer ---------------------------------------------------------

def test_gdn_layer_matches_hf(cfg, hf_text_config):
    """Prefill through one `linear_attention` layer, real weights, no cache."""
    dev = "cuda"
    B, T = 1, 24

    hf_gdn = _load_hf_gdn(hf_text_config, GDN_LAYER)
    ck = load_checkpoint(MODEL_DIR, device=dev, layers=(GDN_LAYER,),
                         include_embeddings=False)
    g = torch.Generator(device=dev).manual_seed(7)
    x = torch.randn(B, T, cfg.hidden_size, generator=g, device=dev, dtype=DTYPE)

    with torch.no_grad():
        ref = hf_gdn(x, cache_params=None, attention_mask=None)
        mine = GatedDeltaNet(cfg, ck.layer(GDN_LAYER))(x)

    assert mine.shape == ref.shape
    r, c = _assert_parity(mine, ref, f"gdn[{GDN_LAYER}]")
    print(f"\n  gdn layer: rel_l2={r:.3e} cosine={c:.9f}")


# --- 2. the full stack --------------------------------------------------------

PROBE_IDS = [151, 9284, 501, 62, 8, 4410, 77, 1201, 33, 990, 12, 7788, 45, 2, 9001, 640]

# Accumulated bf16 rounding over 32 layers, measured. NOT a correctness
# threshold — `test_full_forward_matches_hf_fp32` is where the 5e-3 / 0.99999
# gate is actually applied, and it clears it by four orders of magnitude.
BF16_STACK_DRIFT_MAX = 2e-2


def test_full_forward_matches_hf(engine, hf_text_config):
    """All 32 layers, braid in bf16 on the GPU against an **fp32 CPU** HF.

    An L2 gate on the bf16 arm would be measuring rounding, not correctness. The
    per-layer trace (`scripts/layer_trace_diag.py`) shows the residual growing
    smoothly from 1.9e-4 at layer 0 and plateauing near 1e-2 — no step at any
    layer, which is what a wrongly-wired sublayer would produce.

    The reference moved to fp32-on-CPU when the 9B stopped fitting beside the
    engine. It is the better reference for the same reason it is the cheaper
    one: the drift being bounded is then a statement about braid's arm alone,
    with no rounding of its own on the other side.
    """
    ids = torch.tensor([PROBE_IDS], device="cuda")

    with torch.no_grad():
        mine = engine.forward(ids, cache=None, last_only=False)

    ref_hidden = _hf_hidden(hf_text_config, ids)
    # Read out on the CPU: an fp32 lm_head is 4 GiB on this vocabulary and there
    # is no reason to put it next to the engine.
    ref = torch.nn.functional.linear(ref_hidden, engine.lm_head.cpu().float())
    mine = mine.cpu()
    assert mine.shape == ref.shape
    r, c = _metrics(mine, ref)
    print(f"\n  full forward (bf16 vs fp32 HF): rel_l2={r:.3e} cosine={c:.9f}")
    _assert_greedy(mine, ref, "full forward (bf16)")
    assert r <= BF16_STACK_DRIFT_MAX, f"drift {r:.3e} exceeds the bf16 tripwire"


@pytest.mark.slow
def test_full_forward_matches_hf_fp32(hf_text_config):
    """**The Phase 2 item-3 gate.** All 32 layers in fp32: rel L2 <= 5e-3,
    cosine >= 0.99999, greedy tokens identical.

    fp32 removes the accumulation confound entirely, so what is left is whether
    braid computes the same function as HF. Measured 6.4e-7 on Qwen3.5-4B and
    4.5e-7 on Qwen3.5-9B.

    Both arms on the CPU: two fp32 copies of the 9B are 72 GiB of weights, and
    the alternative — truncating the stack — would retire the one claim this
    test exists to make.
    """
    ids = torch.tensor([PROBE_IDS])
    ref_hidden = _hf_hidden(hf_text_config, ids)

    ck = load_checkpoint(MODEL_DIR, device="cpu", dtype=torch.float32)
    eng = Engine.from_checkpoint(ck, device="cpu", dtype=torch.float32)
    del ck
    try:
        with torch.no_grad():
            mine = eng.forward(ids, cache=None, last_only=False)
            ref = torch.nn.functional.linear(ref_hidden, eng.lm_head)
        r, c = _assert_parity(mine, ref, "full forward logits (fp32)")
        print(f"\n  full forward (fp32, {eng.config.num_hidden_layers} layers, cpu): "
              f"rel_l2={r:.3e} cosine={c:.9f}")
        assert torch.equal(mine.argmax(-1), ref.argmax(-1))
    finally:
        del eng
        gc.collect()


def _read(prefix: str, device: str = "cuda", dtype=None) -> dict[str, torch.Tensor]:
    """Checkpoint tensors under `prefix`.

    With `dtype=None` each tensor keeps its stored dtype, and that is
    load-bearing. `module.to(bfloat16)` followed by `load_state_dict` copies INTO
    the existing bf16 parameter, which silently truncates the tensors this
    checkpoint deliberately stores as F32 — `linear_attn.norm` moves by 2.4e-3
    absolute, and the resulting "reference" is a worse model than braid.
    `assign=True` plus per-tensor dtypes keeps HF faithful to the file, which is
    the only thing worth calling ground truth. Passing `dtype=torch.float32`
    upcasts everything instead, which is faithful in the other direction.
    """
    import json

    from safetensors import safe_open

    wm = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k in wm:
        if k.startswith(prefix):
            by_shard.setdefault(wm[k], []).append(k)

    sd: dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        with safe_open(MODEL_DIR / shard, framework="pt") as f:
            for k in ks:
                t = f.get_tensor(k)
                want = dtype or (torch.float32 if t.dtype == torch.float32 else DTYPE)
                sd[k[len(prefix):]] = t.to(device, want)
    return sd


def _load_hf_gdn(text_config, layer: int):
    with torch.device("meta"):
        mod = hf.Qwen3_5GatedDeltaNet(text_config, layer_idx=layer)
    mod.load_state_dict(
        _read(f"model.language_model.layers.{layer}.linear_attn."), assign=True)
    return mod.eval()


def _hf_hidden(text_config, ids):
    """HF's last hidden state for `ids`, computed in fp32 on the CPU and freed.

    Just the text tower — building the VL wrapper would also load 297 visual
    tensors braid never runs. CPU because two 9B fp32 towers do not fit on the
    card and this box has 245 GiB of host RAM; ~1.5 s for a 16-token forward.
    """
    torch.set_num_threads(os.cpu_count() or 8)
    with torch.device("meta"):
        model = hf.Qwen3_5TextModel(text_config)
    model.load_state_dict(
        _read("model.language_model.", device="cpu", dtype=torch.float32), assign=True)
    # `inv_freq` is a non-persistent buffer, so it is absent from the state dict
    # and `assign=True` leaves it on meta. Rebuild the module on the real device.
    model.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(text_config, device="cpu")
    model = model.eval()
    try:
        with torch.no_grad():
            return model(input_ids=ids.cpu(), use_cache=False).last_hidden_state.float()
    finally:
        del model
        gc.collect()


# --- 3. decode == prefill -----------------------------------------------------

@pytest.mark.parametrize("layer_idx", [GDN_LAYER, 3], ids=["gdn", "attention"])
def test_caches_are_exact_in_fp32(cfg, layer_idx):
    """The cache correctness test, run in fp32 so bf16 noise cannot mask a bug.

    Feeding T tokens one at a time through the conv window / recurrent state /
    KV buffer must reproduce the single T-token call. In fp32 this is exact to
    ~5e-7, which is the only tolerance under which the claim means anything:
    the same comparison in bf16 sits at 4e-4 (GDN) and 1.3e-3 (attention) purely
    because a T=8 GEMM and eight T=1 GEMMs accumulate in different orders, and a
    real bug would hide comfortably under that.

    Catches a conv window holding post-conv outputs, a recurrent state not
    carried between steps, and a KV buffer written at the wrong offset — each of
    which leaves prefill correct and generation subtly wrong.
    """
    from braid.model.attention import Attention, RotaryEmbedding
    from braid.model.cache import KVCache, RecurrentCache
    from braid.model.gdn import GatedDeltaNet

    dev, dt, T = "cuda", torch.float32, 8
    cuda_reclaim()
    ck = load_checkpoint(MODEL_DIR, device=dev, layers=(layer_idx,),
                         include_embeddings=False)
    w = {k: (v.to(dt) if v.is_floating_point() and v.dtype != torch.float32 else v)
         for k, v in ck.layer(layer_idx).items()}
    g = torch.Generator(device=dev).manual_seed(4)
    x = torch.randn(1, T, cfg.hidden_size, generator=g, device=dev, dtype=dt)

    with torch.no_grad():
        if cfg.is_gdn(layer_idx):
            mod = GatedDeltaNet(cfg, w)
            bulk = mod(x)
            c = RecurrentCache(cfg, 1, dev, dt)
            slot = torch.zeros(1, dtype=torch.int64, device=dev)
            step = torch.cat([mod(x[:, t:t + 1], cache=c, slots=slot) for t in range(T)],
                             dim=1)
        else:
            mod = Attention(cfg, w)
            rope = RotaryEmbedding(cfg, dev, dt)
            pos = torch.arange(T, device=dev)[None]
            cos, sin = rope(pos)
            bulk = mod(x, cos, sin)
            kv = KVCache(cfg, 1, T + 2, dev, dt)
            slot = torch.zeros(1, dtype=torch.int64, device=dev)
            outs = []
            for t in range(T):
                ct, st = rope(pos[:, t:t + 1])
                where = torch.tensor([t], device=dev)
                outs.append(mod(x[:, t:t + 1], ct, st, cache=kv, slots=slot,
                                positions=where, kv_len=t + 1))
            step = torch.cat(outs, dim=1)

    r, c = _assert_parity(step, bulk, f"cache layer {layer_idx}",
                          rel_max=1e-5, cos_min=0.999999999)
    print(f"\n  cache layer {layer_idx} (fp32): rel_l2={r:.3e} cosine={c:.9f}")


# bf16 drift between a T-token call and T single-token calls, measured over all
# 32 layers. Not a correctness threshold — see test_caches_are_exact_in_fp32 for
# that — just a tripwire on the accumulated GEMM-shape noise.
BF16_CACHE_DRIFT_MAX = 3e-2


CACHE_IDS = [[7, 4410, 77, 1201, 33, 990, 12, 7788]]


def test_decode_matches_prefill(engine):
    """Full model, bf16: greedy tokens must agree where bf16 resolves them,
    drift merely bounded. See `_assert_greedy` for why "where bf16 resolves
    them" and not "everywhere" — on the 9B this sequence contains an exact bf16
    tie, and the tie-break is not a property of the cache."""
    ids = torch.tensor(CACHE_IDS, device="cuda")
    T = ids.shape[1]

    with torch.no_grad():
        bulk = engine.forward(ids, cache=None, last_only=False)
        cache = engine.allocate_cache(max_len=T + 4)
        step = torch.cat([engine.forward(ids[:, t: t + 1], cache) for t in range(T)], dim=1)

    assert int(cache.lengths[0]) == T
    r, c = _metrics(step, bulk)
    print(f"\n  decode vs prefill (bf16, {engine.config.num_hidden_layers} layers): "
          f"rel_l2={r:.3e} cosine={c:.9f}")
    _assert_greedy(step, bulk, "decode vs prefill (bf16)")
    assert r <= BF16_CACHE_DRIFT_MAX, f"drift {r:.3e} exceeds the bf16 tripwire"


@pytest.mark.slow
def test_decode_matches_prefill_exactly_in_fp32():
    """The same claim with the confound removed: **full stack, fp32, on the CPU,
    token-for-token identical.**

    `test_caches_are_exact_in_fp32` above pins one layer of each kind at 5e-7;
    this pins all 32 wired together, which is where a cache that is right per
    layer and wrong across the schedule would show. It exists because the bf16
    version of this test could only ever be a tripwire, and the fp32 version
    could not be run on the GPU at 9B — 33 GiB of weights against a 31 GiB card.
    Measured 9.3e-7 on Qwen3.5-9B.
    """
    ids = torch.tensor(CACHE_IDS)
    T = ids.shape[1]
    torch.set_num_threads(os.cpu_count() or 8)

    ck = load_checkpoint(MODEL_DIR, device="cpu", dtype=torch.float32)
    eng = Engine.from_checkpoint(ck, device="cpu", dtype=torch.float32)
    del ck
    try:
        with torch.no_grad():
            bulk = eng.forward(ids, cache=None, last_only=False)
            cache = eng.allocate_cache(max_len=T + 4)
            step = torch.cat([eng.forward(ids[:, t:t + 1], cache) for t in range(T)],
                             dim=1)
        r, c = _metrics(step, bulk)
        print(f"\n  decode vs prefill (fp32, {eng.config.num_hidden_layers} layers, "
              f"cpu): rel_l2={r:.3e} cosine={c:.9f}")
        assert int(cache.lengths[0]) == T
        assert torch.equal(step.argmax(-1), bulk.argmax(-1)), "greedy tokens differ"
        assert r <= 1e-5, f"fp32 decode/prefill residual {r:.3e}"
    finally:
        del eng
        gc.collect()


def test_prefill_then_decode_is_continuous(engine):
    """Prefill T-1 tokens, then step the last one; must match a full prefill."""
    ids = torch.tensor(CACHE_IDS, device="cuda")

    with torch.no_grad():
        bulk = engine.forward(ids, cache=None)
        cache = engine.allocate_cache(max_len=16)
        engine.forward(ids[:, :-1], cache)
        tail = engine.forward(ids[:, -1:], cache)

    r, _ = _metrics(tail, bulk)
    _assert_greedy(tail, bulk, "prefill-then-decode (bf16)")
    assert r <= BF16_CACHE_DRIFT_MAX


# --- 4. proof of life ---------------------------------------------------------

def test_capital_of_france(engine):
    tok = pytest.importorskip("tokenizers")
    t = tok.Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    ids = torch.tensor([t.encode("The capital of France is").ids], device="cuda")

    out = engine.generate(ids, max_new_tokens=4, temperature=0.0)
    text = t.decode(out[0].tolist(), skip_special_tokens=False)
    print(f"\n  'The capital of France is' -> {text!r}")
    assert "Paris" in text, f"expected Paris, got {text!r}"


def test_generates_coherent_continuation(engine):
    """128 greedy tokens with no degeneration: not a single repeated 8-gram."""
    tok = pytest.importorskip("tokenizers")
    t = tok.Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    prompt = "The process of photosynthesis in plants can be described as follows:"
    ids = torch.tensor([t.encode(prompt).ids], device="cuda")

    out = engine.generate(ids, max_new_tokens=128, temperature=0.0)[0].tolist()
    text = t.decode(out, skip_special_tokens=True)
    print(f"\n  128 greedy tokens: {text[:300]!r}")

    assert len(out) == 128
    grams = [tuple(out[i: i + 8]) for i in range(len(out) - 7)]
    assert len(set(grams)) == len(grams), "repeated 8-gram: greedy decode degenerated"


def test_weight_footprint(engine, cfg):
    """The engine holds the text tower and nothing else.

    This used to pin a literal 9.0 GiB, which is a Qwen3.5-4B fact and failed on
    the 9B for being a different model rather than for being wrong. What it was
    really asserting is that no part of the visual tower was loaded, so assert
    that: the measured footprint must match what the *text* config predicts. On
    Qwen3.5-9B the visual tower is 333 of 775 tensors, so a leak is not a
    rounding error and 2% is a generous window.
    """
    gb = engine.weight_bytes() / 2 ** 30
    want = weight_bytes(cfg, itemsize=engine.dtype.itemsize) / 2 ** 30
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"\n  weights {gb:.2f} GiB (text config predicts {want:.2f}), "
          f"peak allocated {peak:.2f} GiB")
    assert abs(gb - want) / want < 0.02, (
        f"engine holds {gb:.2f} GiB where the text tower is {want:.2f} GiB — "
        f"the visual tower may have leaked in")
