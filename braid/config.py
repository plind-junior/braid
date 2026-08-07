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
        fluent garbage. See ARCHITECTURE.md §5 — this order is DISPUTED between
        two readings of that code and is settled empirically at Phase 2, not
        here.
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
    def qwen35_4b(cls) -> "GDNConfig":
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
    def qwen36_35b_a3b(cls) -> "GDNConfig":
        return cls(n_heads=32, head_dim=128, state_size=128, n_groups=16)

    @classmethod
    def qwen36_27b(cls) -> "GDNConfig":
        return cls(n_heads=48, head_dim=128, state_size=128, n_groups=16)
