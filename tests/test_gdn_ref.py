import torch

from braid.config import GDNConfig
from braid.reference.gdn_ref import gdn_decode_naive, gdn_decode_vectorized


def _inputs(B, cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        state=torch.randn(B, cfg.n_heads, cfg.state_size, cfg.head_dim, generator=g, dtype=torch.float32),
        q=torch.randn(B, cfg.n_groups, cfg.state_size, generator=g, dtype=torch.float32),
        k=torch.randn(B, cfg.n_groups, cfg.state_size, generator=g, dtype=torch.float32),
        v=torch.randn(B, cfg.n_heads, cfg.head_dim, generator=g, dtype=torch.float32),
        alpha=torch.rand(B, cfg.n_heads, generator=g, dtype=torch.float32) * 0.5 + 0.5,
        beta=torch.rand(B, cfg.n_heads, generator=g, dtype=torch.float32),
    )


def test_config_matches_qwen36_35b():
    cfg = GDNConfig.qwen36_35b_a3b()
    assert (cfg.n_heads, cfg.head_dim, cfg.state_size, cfg.n_groups) == (32, 128, 128, 16)


def test_state_bytes_match_reference_measurement():
    """The reference engine measures 63.8 MiB of recurrent state per sequence
    on this model.

    30 GDN layers of (256B-aligned h_state + 256B-aligned conv_state).
    Reproducing their number is the cheapest possible check that our shape
    constants describe the same model theirs do.
    """
    cfg = GDNConfig.qwen36_35b_a3b()
    total = (cfg.state_bytes_per_seq_per_layer + cfg.conv_bytes_per_seq_per_layer) * cfg.n_gdn_layers
    assert abs(total / 2**20 - 63.75) < 0.1, f"{total / 2**20:.2f} MiB, expected ~63.8"


def test_naive_and_vectorized_agree():
    cfg = GDNConfig(n_heads=4, head_dim=8, state_size=8, n_groups=2)
    a, b = _inputs(3, cfg), _inputs(3, cfg)
    y_naive = gdn_decode_naive(**a, cfg=cfg)
    y_vec = gdn_decode_vectorized(**b, cfg=cfg)
    torch.testing.assert_close(y_naive, y_vec, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(a["state"], b["state"], rtol=1e-5, atol=1e-6)


def test_state_is_mutated_in_place():
    cfg = GDNConfig(n_heads=2, head_dim=4, state_size=4, n_groups=1)
    d = _inputs(1, cfg)
    before = d["state"].clone()
    gdn_decode_vectorized(**d, cfg=cfg)
    assert not torch.allclose(before, d["state"])


def test_zero_beta_is_pure_decay():
    """beta=0 means no write; the state should just scale by alpha."""
    cfg = GDNConfig(n_heads=2, head_dim=4, state_size=4, n_groups=1)
    d = _inputs(1, cfg)
    d["beta"] = torch.zeros_like(d["beta"])
    before = d["state"].clone()
    gdn_decode_vectorized(**d, cfg=cfg)
    expected = before * d["alpha"][0, :, None, None]
    torch.testing.assert_close(d["state"], expected, rtol=1e-5, atol=1e-6)


def test_batch_rows_are_independent():
    """Row 1 must be unaffected by row 0's inputs. This is the property the
    batched kernel exists to preserve."""
    cfg = GDNConfig(n_heads=2, head_dim=4, state_size=4, n_groups=1)
    two = _inputs(2, cfg)
    solo = {kk: vv[1:2].clone() for kk, vv in two.items()}
    y_two = gdn_decode_vectorized(**two, cfg=cfg)
    y_solo = gdn_decode_vectorized(**solo, cfg=cfg)
    torch.testing.assert_close(y_two[1:2], y_solo, rtol=1e-5, atol=1e-6)


def test_head_to_group_mapping_is_grouped_not_tiled():
    """HF SafeTensors is the GROUPED layout: g = h // (n_heads // n_groups).

    The reference engine supports both (gdn.cu:55) and picks by checkpoint
    format. Both are valid permutations of the same index range, so a mismatch
    produces
    plausible-looking garbage and never a crash — worth pinning explicitly.
    """
    cfg = GDNConfig(n_heads=4, head_dim=4, state_size=4, n_groups=2)
    d = _inputs(1, cfg)
    # Zero group 1 entirely; under grouped layout that must affect heads 2,3
    # and leave heads 0,1 alone. Under tiled (h % n_groups) it would hit 1,3.
    d["k"][:, 1] = 0.0
    d["q"][:, 1] = 0.0
    base = {kk: vv.clone() for kk, vv in d.items()}
    base["k"] = _inputs(1, cfg)["k"]
    base["q"] = _inputs(1, cfg)["q"]

    y_zeroed = gdn_decode_vectorized(**d, cfg=cfg)
    y_base = gdn_decode_vectorized(**base, cfg=cfg)
    same = [torch.allclose(y_zeroed[0, h], y_base[0, h]) for h in range(cfg.n_heads)]
    assert same == [True, True, False, False], f"grouped layout violated: {same}"
