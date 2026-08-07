"""Load the HF safetensors checkpoint into braid's flat tensor namespace.

The published Qwen3.5-4B is a vision-language checkpoint: **738 tensors, of
which braid runs 426.** The other 312 are a 24-block visual tower (297) and a
multi-token-prediction head (15). Both are absent from the GGUF llama.cpp runs,
so a loader that walks the index naively does not merely waste VRAM — it makes
the head-to-head compare two different models. `load_checkpoint` therefore
*filters by prefix and reports what it dropped*, rather than loading whatever it
finds.

Four load-time transforms, each silently wrong if missed:

  1. `A = -exp(A_log)`, keyed on the **source tensor name**. Neither dtype nor
     value works here: this checkpoint ships `A_log` as F32 *and* entirely
     negative, so both published heuristics read "already transformed" and skip
     the exp. See `_apply_transform` for the measurement.
  2. RMSNorm `gamma = 1 + W`, in fp32, on every plain norm — and **not** on
     `linear_attn.norm`, which is the gated form and stores gamma directly.
     (Settled empirically in `tests/test_hf_parity.py`, not read off the
     reference engine's source.)
  3. `conv1d.weight` is `[C, 1, K]`; the kernel wants `[C, K]`.
  4. `tie_word_embeddings` — there is no `lm_head` tensor in this file at all.

The NVFP4-specific transforms in ROADMAP Phase 2 item 1 (llm-compressor suffix
renames, the `tensor_scale = 1/weight_global_scale` reciprocal-direction trap)
do not apply: this checkpoint is BF16 throughout.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

from braid.model.config import ModelConfig

TEXT_PREFIX = "model.language_model."
VISUAL_PREFIX = "model.visual."
MTP_PREFIX = "mtp."

# Norms whose stored weight is a DELTA. `linear_attn.norm` is deliberately
# absent: it is `Qwen3_5RMSNormGated` and stores gamma directly.
_PLAIN_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)

_ATTN_TENSORS = (
    "input_layernorm", "post_attention_layernorm",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.q_norm", "self_attn.k_norm",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
_GDN_TENSORS = (
    "input_layernorm", "post_attention_layernorm",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
    "linear_attn.in_proj_a", "linear_attn.in_proj_b",
    "linear_attn.out_proj", "linear_attn.conv1d",
    "linear_attn.A", "linear_attn.dt_bias", "linear_attn.norm",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")


@dataclass
class LoadReport:
    """What the loader saw and what it refused, so "unfair comparison" is visible."""

    total_in_index: int = 0
    text: int = 0
    visual_skipped: int = 0
    mtp_skipped: int = 0
    unrecognised: tuple[str, ...] = ()
    loaded: int = 0
    layers_loaded: tuple[int, ...] = ()
    a_log_transformed: int = 0
    norms_offset: int = 0
    tied_lm_head: bool = False

    def summary(self) -> str:
        return (
            f"{self.loaded} tensors over {len(self.layers_loaded)} layers "
            f"(index has {self.total_in_index}: {self.text} text, "
            f"{self.visual_skipped} visual skipped, {self.mtp_skipped} MTP skipped); "
            f"A_log->A on {self.a_log_transformed}, 1+W on {self.norms_offset}, "
            f"tied lm_head={self.tied_lm_head}"
        )


@dataclass
class Checkpoint:
    config: ModelConfig
    tensors: dict[str, torch.Tensor]
    report: LoadReport = field(default_factory=LoadReport)

    def layer(self, idx: int) -> dict[str, torch.Tensor]:
        """This layer's tensors, keyed without the `layers.N.` prefix."""
        pre = f"layers.{idx}."
        out = {k[len(pre):]: v for k, v in self.tensors.items() if k.startswith(pre)}
        if not out:
            raise KeyError(f"layer {idx} not loaded (loaded: {self.report.layers_loaded})")
        return out

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key]


def braid_name(hf_key: str) -> str | None:
    """HF key -> braid flat name. `None` means "not part of the text tower"."""
    if hf_key.startswith((VISUAL_PREFIX, MTP_PREFIX)):
        return None
    if not hf_key.startswith(TEXT_PREFIX):
        return None
    name = hf_key[len(TEXT_PREFIX):]
    # `A_log` and `dt_bias` are bare parameters; everything else is `<mod>.weight`.
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    if name.endswith(".A_log"):
        name = name[: -len("A_log")] + "A"
    return name


def _apply_transform(
    hf_key: str, name: str, t: torch.Tensor, report: LoadReport
) -> torch.Tensor:
    leaf = _LAYER_RE.sub("", name)

    if leaf.endswith("linear_attn.A"):
        # Keyed on the SOURCE NAME, which is unambiguous for a safetensors load:
        # `A_log` is stored, `A = -exp(A_log)` is what the recurrence wants.
        #
        # The original spec prescribed deciding by VALUE instead ("any element
        # >= 0 => raw HF"). That heuristic is unsound on this checkpoint and
        # measured so: every one of layer 0's 32 `A_log` entries is negative
        # (-4.22 .. -0.96), so the value test reads "already transformed", skips
        # the exp, and leaves A = -2.7 where it should be -0.067. The state then
        # decays ~40x too fast — it collapses silently rather than diverging
        # loudly, so the absmax tell the doc describes never fires. The
        # value heuristic belongs to the GGUF path, whose conversion script may
        # have folded the transform already; braid reads HF safetensors only.
        if hf_key.endswith(".A_log"):
            t = -torch.exp(t.float())
            report.a_log_transformed += 1
        t = t.float()
        if not bool((t < 0).all()) or not bool(torch.isfinite(t).all()):
            raise ValueError(
                f"{name}: A must be finite and strictly negative after transform; "
                f"got min={t.min().item()}, max={t.max().item()}"
            )
        return t

    if leaf.endswith("linear_attn.dt_bias"):
        return t.float()

    if leaf.endswith("linear_attn.conv1d"):
        # [C, 1, K] -> [C, K], K contiguous per channel.
        return t.squeeze(1).contiguous()

    if leaf in _PLAIN_NORMS or leaf == "norm":
        # fp32 fold; see norm.py for the measured headroom it buys.
        report.norms_offset += 1
        return 1.0 + t.float()

    if leaf.endswith("linear_attn.norm"):
        return t.float()  # gated form: gamma stored directly, no offset

    return t


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cuda",
    layers: Iterable[int] | None = None,
    include_embeddings: bool = True,
    dtype: torch.dtype | None = None,
) -> Checkpoint:
    """mmap the safetensors shards and materialise the text tower on `device`.

    `layers` restricts to a subset — the parity tests need two layers, not 8.8 GB.
    `include_embeddings=False` skips the 1.27 GB tied embedding table with it.

    `dtype` recasts the BF16 weights on the way through, on the host and before
    the transfer, so an fp32 run never has a bf16 and an fp32 copy resident at
    once — that doubling is what puts a 4B model over a 32 GB card. Tensors the
    checkpoint stores as fp32 (`A_log`, `linear_attn.norm`) are left alone; they
    are fp32 deliberately, and widening is not the same as not narrowing.
    """
    root = Path(path)
    cfg = ModelConfig.from_pretrained(root)

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map: dict[str, str] = json.load(open(index_path))["weight_map"]
    else:
        single = root / "model.safetensors"
        with safe_open(single, framework="pt") as f:
            weight_map = {k: single.name for k in f.keys()}

    want_layers = None if layers is None else set(layers)
    report = LoadReport(total_in_index=len(weight_map))
    unrecognised: list[str] = []

    # Group by shard so each file is opened once; `safe_open` mmaps, so only the
    # slices actually requested are faulted in.
    per_file: dict[str, list[tuple[str, str]]] = {}
    for hf_key, shard in weight_map.items():
        if hf_key.startswith(VISUAL_PREFIX):
            report.visual_skipped += 1
            continue
        if hf_key.startswith(MTP_PREFIX):
            report.mtp_skipped += 1
            continue
        name = braid_name(hf_key)
        if name is None:
            unrecognised.append(hf_key)
            continue
        report.text += 1

        m = _LAYER_RE.match(name)
        if m is not None:
            if want_layers is not None and int(m.group(1)) not in want_layers:
                continue
        elif not include_embeddings and name in ("embed_tokens",):
            continue
        per_file.setdefault(shard, []).append((hf_key, name))

    report.unrecognised = tuple(sorted(unrecognised))
    if report.unrecognised:
        raise ValueError(
            "checkpoint contains tensors outside the known prefixes: "
            f"{report.unrecognised[:8]}"
        )

    tensors: dict[str, torch.Tensor] = {}
    for shard, items in per_file.items():
        with safe_open(root / shard, framework="pt") as f:
            for hf_key, name in items:
                t = f.get_tensor(hf_key)
                if dtype is not None and t.dtype != torch.float32 and t.is_floating_point():
                    t = t.to(dtype)
                t = t.to(device, non_blocking=True)
                tensors[name] = _apply_transform(hf_key, name, t, report)

    if cfg.tie_word_embeddings and "embed_tokens" in tensors:
        tensors["lm_head"] = tensors["embed_tokens"]
        report.tied_lm_head = True
    elif "lm_head" not in tensors and include_embeddings and want_layers is None:
        raise ValueError("no lm_head tensor and tie_word_embeddings is false")

    report.loaded = len(tensors)
    loaded_layers = sorted({int(m.group(1)) for k in tensors if (m := _LAYER_RE.match(k))})
    report.layers_loaded = tuple(loaded_layers)

    _validate(cfg, tensors, loaded_layers)
    return Checkpoint(config=cfg, tensors=tensors, report=report)


def _validate(cfg: ModelConfig, tensors: dict[str, torch.Tensor], layers: list[int]) -> None:
    """Every expected tensor present, and every shape as the config predicts.

    Shape checks are the cheap half of the head_dim trap: `q_proj` must be
    `[2 * n_heads * head_dim, hidden]` = `[8192, 2560]`, which only holds for
    head_dim 256. The inferred 160 would give 5120 and fail here rather than
    reshaping into fluent garbage.
    """
    for i in layers:
        expected = _GDN_TENSORS if cfg.is_gdn(i) else _ATTN_TENSORS
        missing = [n for n in expected if f"layers.{i}.{n}" not in tensors]
        if missing:
            raise ValueError(f"layer {i} ({cfg.layer_types[i]}) missing: {missing}")

    h, hd = cfg.hidden_size, cfg.head_dim
    g = cfg.gdn
    want = {
        "self_attn.q_proj": (cfg.q_proj_out, h),
        "self_attn.k_proj": (cfg.kv_dim, h),
        "self_attn.v_proj": (cfg.kv_dim, h),
        "self_attn.o_proj": (h, cfg.q_dim),
        "self_attn.q_norm": (hd,),
        "self_attn.k_norm": (hd,),
        "mlp.gate_proj": (cfg.intermediate_size, h),
        "mlp.up_proj": (cfg.intermediate_size, h),
        "mlp.down_proj": (h, cfg.intermediate_size),
        "input_layernorm": (h,),
        "post_attention_layernorm": (h,),
        "linear_attn.in_proj_qkv": (g.conv_channels, h),
        "linear_attn.in_proj_z": (g.inner_size, h),
        "linear_attn.in_proj_a": (g.n_heads, h),
        "linear_attn.in_proj_b": (g.n_heads, h),
        "linear_attn.out_proj": (h, g.inner_size),
        "linear_attn.conv1d": (g.conv_channels, g.conv_kernel),
        "linear_attn.A": (g.n_heads,),
        "linear_attn.dt_bias": (g.n_heads,),
        "linear_attn.norm": (g.head_dim,),
    }
    for name, t in tensors.items():
        leaf = _LAYER_RE.sub("", name)
        if leaf in want and tuple(t.shape) != want[leaf]:
            raise ValueError(f"{name}: shape {tuple(t.shape)}, config predicts {want[leaf]}")

    if "embed_tokens" in tensors:
        if tuple(tensors["embed_tokens"].shape) != (cfg.vocab_size, h):
            raise ValueError(f"embed_tokens: {tuple(tensors['embed_tokens'].shape)}")
