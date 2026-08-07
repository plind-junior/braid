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

Loading two copies of a 4B model costs ~17 GB, so these are module-scoped and
the HF reference is freed before generation runs.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

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
    """All 32 layers in bf16: greedy tokens identical, drift bounded.

    An L2 gate on the bf16 arm would be measuring rounding, not correctness. The
    per-layer trace (`scripts/layer_trace_diag.py`) shows the residual growing
    smoothly from 1.9e-4 at layer 0 and plateauing near 1e-2 — no step at any
    layer, which is what a wrongly-wired sublayer would produce.
    """
    dev = "cuda"
    ids = torch.tensor([PROBE_IDS], device=dev)

    with torch.no_grad():
        mine = engine.forward(ids, cache=None, last_only=False)

    hf_model = _load_hf_text_model(hf_text_config)
    try:
        with torch.no_grad():
            ref = torch.nn.functional.linear(
                hf_model(input_ids=ids, use_cache=False).last_hidden_state,
                engine.lm_head)
        assert mine.shape == ref.shape
        r, c = _metrics(mine, ref)
        print(f"\n  full forward (bf16): rel_l2={r:.3e} cosine={c:.9f}")
        assert torch.equal(mine.argmax(-1), ref.argmax(-1)), (
            "greedy tokens differ from HF somewhere in the sequence")
        assert r <= BF16_STACK_DRIFT_MAX, f"drift {r:.3e} exceeds the bf16 tripwire"
    finally:
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.slow
def test_full_forward_matches_hf_fp32(hf_text_config):
    """**The Phase 2 item-3 gate.** All 32 layers in fp32: rel L2 <= 5e-3,
    cosine >= 0.99999.

    fp32 removes the accumulation confound entirely, so what is left is whether
    braid computes the same function as HF. It does, to 6.4e-7.

    Loads two fp32 copies of a 4B model (~16 GiB each), so they are built and
    freed in sequence rather than held together.
    """
    dev = "cuda"
    ids = torch.tensor([PROBE_IDS], device=dev)

    hf_model = _load_hf_text_model(hf_text_config).float()
    try:
        with torch.no_grad():
            ref_hidden = hf_model(input_ids=ids, use_cache=False).last_hidden_state.float()
    finally:
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

    ck = load_checkpoint(MODEL_DIR, device=dev, dtype=torch.float32)
    eng = Engine.from_checkpoint(ck, device=dev, dtype=torch.float32)
    try:
        with torch.no_grad():
            mine = eng.forward(ids, cache=None, last_only=False)
            ref = torch.nn.functional.linear(ref_hidden, eng.lm_head)
        r, c = _assert_parity(mine, ref, "full forward logits (fp32)")
        print(f"\n  full forward (fp32): rel_l2={r:.3e} cosine={c:.9f}")
        assert torch.equal(mine.argmax(-1), ref.argmax(-1))
    finally:
        del eng, ck
        gc.collect()
        torch.cuda.empty_cache()


def _read(prefix: str) -> dict[str, torch.Tensor]:
    """Checkpoint tensors under `prefix`, keeping each one's stored dtype.

    Preserving dtype is load-bearing. `module.to(bfloat16)` followed by
    `load_state_dict` copies INTO the existing bf16 parameter, which silently
    truncates the tensors this checkpoint deliberately stores as F32 —
    `linear_attn.norm` moves by 2.4e-3 absolute, and the resulting "reference"
    is a worse model than braid. `assign=True` plus per-tensor dtypes keeps HF
    faithful to the file, which is the only thing worth calling ground truth.
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
                sd[k[len(prefix):]] = t.to(
                    "cuda", torch.float32 if t.dtype == torch.float32 else DTYPE)
    return sd


def _load_hf_gdn(text_config, layer: int):
    with torch.device("meta"):
        mod = hf.Qwen3_5GatedDeltaNet(text_config, layer_idx=layer)
    mod.load_state_dict(
        _read(f"model.language_model.layers.{layer}.linear_attn."), assign=True)
    return mod.eval()


def _load_hf_text_model(text_config):
    """Just the text tower — building the VL wrapper would also load 297 visual
    tensors braid never runs, and the pair would not fit alongside the engine."""
    with torch.device("meta"):
        model = hf.Qwen3_5TextModel(text_config)
    model.load_state_dict(_read("model.language_model."), assign=True)
    # `inv_freq` is a non-persistent buffer, so it is absent from the state dict
    # and `assign=True` leaves it on meta. Rebuild the module on the real device.
    model.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(text_config, device="cuda")
    return model.eval()


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


def test_decode_matches_prefill(engine):
    """Full model, bf16: greedy tokens must be identical, drift merely bounded.

    Token identity is the claim that matters — it is the same gate Phase 3 uses
    for the batch axis — and unlike an L2 threshold it does not have to be
    calibrated against bf16 noise.
    """
    dev = "cuda"
    ids = torch.tensor([[7, 4410, 77, 1201, 33, 990, 12, 7788]], device=dev)
    T = ids.shape[1]

    with torch.no_grad():
        bulk = engine.forward(ids, cache=None, last_only=False)
        cache = engine.allocate_cache(max_len=T + 4)
        step = torch.cat([engine.forward(ids[:, t: t + 1], cache) for t in range(T)], dim=1)

    assert int(cache.lengths[0]) == T
    r, c = _metrics(step, bulk)
    print(f"\n  decode vs prefill (bf16, 32 layers): rel_l2={r:.3e} cosine={c:.9f}")
    assert torch.equal(step.argmax(-1), bulk.argmax(-1)), "greedy tokens differ"
    assert r <= BF16_CACHE_DRIFT_MAX, f"drift {r:.3e} exceeds the bf16 tripwire"


def test_prefill_then_decode_is_continuous(engine):
    """Prefill T-1 tokens, then step the last one; must match a full prefill."""
    dev = "cuda"
    ids = torch.tensor([[7, 4410, 77, 1201, 33, 990, 12, 7788]], device=dev)

    with torch.no_grad():
        bulk = engine.forward(ids, cache=None)
        cache = engine.allocate_cache(max_len=16)
        engine.forward(ids[:, :-1], cache)
        tail = engine.forward(ids[:, -1:], cache)

    r, _ = _metrics(tail, bulk)
    assert torch.equal(tail.argmax(-1), bulk.argmax(-1)), "greedy token differs"
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


def test_weight_footprint(engine):
    """8.8 GB of BF16 weights — the rescoped Phase 2 VRAM budget is 12 GB."""
    gb = engine.weight_bytes() / 2 ** 30
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"\n  weights {gb:.2f} GiB, peak allocated {peak:.2f} GiB")
    assert gb < 9.0, f"text tower is {gb:.2f} GiB; the visual tower may have leaked in"
