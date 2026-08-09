from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GDNConfig:
    """Gated DeltaNet shape parameters.

    Names follow HF config.json; the reference engine's kernel argument names
    are noted for cross-referencing, marked `ref:`:
      n_heads    == linear_num_value_heads  (ref: n_heads / ssm_dt_rank)
      head_dim   == linear_value_head_dim   (ref: head_dim_ssm, HD)
      state_size == linear_key_head_dim     (ref: state_size, SS)
      n_groups   == linear_num_key_heads    (ref: n_groups)
    """

    n_heads: int
    head_dim: int
    state_size: int
    n_groups: int
    conv_kernel: int = 4
    n_gdn_layers: int = 30

    @property
    def heads_per_group(self) -> int:
        return self.n_heads // self.n_groups

    @property
    def inner_size(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def conv_channels(self) -> int:
        """[ Q(n_groups*SS) | K(n_groups*SS) | V(n_heads*HD) ] — Q FIRST.

        The buffer is named `xBC` in the reference engine's shared Mamba2 code
        and its header comment mentions the order only in passing; reading it
        as [K|Q|V] swaps the delta-rule key with the readout query and produces
        fluent garbage. See ARCHITECTURE.md §6 — the order was DISPUTED between
        two readings of that code and is now settled empirically by
        tests/test_hf_parity.py, not asserted here.
        """
        return 2 * self.n_groups * self.state_size + self.inner_size

    @staticmethod
    def _align256(n: int) -> int:
        return (n + 255) & ~255

    @property
    def state_bytes_per_seq_per_layer(self) -> int:
        return self._align256(self.n_heads * self.state_size * self.head_dim * 4)

    @property
    def conv_bytes_per_seq_per_layer(self) -> int:
        return self._align256(self.conv_channels * self.conv_kernel * 4)

    @property
    def recurrent_bytes_per_seq(self) -> int:
        """Total recurrent footprint for one sequence, all GDN layers.

        63.8 MiB on Qwen3.6-35B-A3B, matching the reference engine's measured
        figure (docs/BENCHMARKS.md:313).
        """
        per_layer = self.state_bytes_per_seq_per_layer + self.conv_bytes_per_seq_per_layer
        return per_layer * self.n_gdn_layers

    @classmethod
    def qwen35_4b(cls) -> GDNConfig:
        """The MVP target: Qwen3.5-4B, a DENSE GDN hybrid.

        32 layers as 3x(linear_attention) -> 1x(full_attention), so 24 GDN and
        8 attention. hidden 2560, MLP 9216 dense (mlp_only_layers == []),
        attention 16 heads / head_dim 256 / 4 KV heads, vocab 248320, tied
        embeddings, mamba_ssm_dtype float32.

        Its GDN block is dimensionally IDENTICAL to qwen36_35b_a3b, which is
        why it is the MVP target: the decode kernel transfers unchanged while
        the MoE and NVFP4 paths stay off the critical path.

        The published checkpoint is vision-language (738 tensors, a visual
        tower and an MTP head); braid uses the text tower only, matching what
        llama.cpp's GGUF contains.
        """
        return cls(n_heads=32, head_dim=128, state_size=128, n_groups=16, n_gdn_layers=24)

    @classmethod
    def qwen35_9b(cls) -> GDNConfig:
        """Qwen3.5-9B — **GDN-identical to the 4B**, which is the point.

        Verified against the published `config.json`: `linear_num_value_heads`
        32, `linear_num_key_heads` 16, both head dims 128, conv kernel 4, and
        32 layers at `full_attention_interval` 4 = 24 GDN. Every one of those
        matches `qwen35_4b`, so this returns an equal object and the decode
        kernel, the conv kernel and the graph buckets all transfer untouched.

        What differs is outside the GDN block: `hidden_size` 4096 (vs 2560),
        `intermediate_size` 12288 (vs 9216), and **`tie_word_embeddings` is
        false** — the 9B ships a real `lm_head.weight`, at the top level rather
        than under the text tower. See `braid/model/loader.py`.

        Consequence for the throughput model: weights roughly double (~18 GB
        against 8.44) while **per-sequence recurrent state is unchanged** at
        51 MiB, so the fixed term grows and the linear term does not.
        """
        return cls(n_heads=32, head_dim=128, state_size=128, n_groups=16, n_gdn_layers=24)

    @classmethod
    def qwen36_35b_a3b(cls) -> GDNConfig:
        return cls(n_heads=32, head_dim=128, state_size=128, n_groups=16)

    @classmethod
    def qwen36_27b(cls) -> GDNConfig:
        """Qwen3.6-27B — dense, 64 layers at interval 4, so **48 GDN layers**.

        Corrected 2026-08-09 against the published `config.json`: this
        previously took the class default of 30 GDN layers, which was never
        checked against the real checkpoint. Also `hidden_size` 5120,
        `intermediate_size` 17408, 24 attention heads, `head_dim` 256, 4 KV
        heads, untied embeddings, no MoE.

        **This is the most thesis-favourable model in the family and does not
        fit the card.** 48 heads over 48 layers put per-sequence state at
        ~152 MiB, 3x the 4B/9B, which is the term braid's advantage feeds on —
        but ~27B parameters is ~54 GB in BF16 and ~27 GB even at 8-bit, leaving
        too little for state and KV at the B >= 32 where braid wins. It needs
        4-bit weights first (ROADMAP Phase 5+).
        """
        return cls(n_heads=48, head_dim=128, state_size=128, n_groups=16, n_gdn_layers=48)
