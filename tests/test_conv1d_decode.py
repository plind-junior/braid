import pytest
import torch

from braid.config import GDNConfig
from braid.kernels.loader import load_gdn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def conv1d_ref(state, x, weight, bias):
    """Torch reference. Mutates `state` in place, returns SiLU(conv + bias).

    state  [B, C, K] fp32 -- per-channel window, K contiguous, oldest first
    x      [B, C]    fp32 -- this token's value per channel
    weight [C, K]    fp32
    bias   [C]       fp32
    """
    state[:, :, :-1] = state[:, :, 1:].clone()
    state[:, :, -1] = x
    acc = (state * weight).sum(-1) + bias
    return acc * torch.sigmoid(acc)


def _cfg_small():
    return GDNConfig(n_heads=4, head_dim=8, state_size=8, n_groups=2, conv_kernel=4)


def test_matches_reference_single_step():
    mod = load_gdn()
    C, K, B = 512, 4, 3
    pool = torch.randn(8, C, K, device="cuda")
    slots = torch.tensor([5, 0, 2], dtype=torch.int32, device="cuda")
    x = torch.randn(B, C, device="cuda")
    w = torch.randn(C, K, device="cuda")
    bias = torch.randn(C, device="cuda")

    ref_state = pool[slots.long()].clone()
    y_ref = conv1d_ref(ref_state, x, w, bias)

    out = torch.empty(B, C, device="cuda")
    mod.conv1d_decode(pool, slots, x, w, bias, out)

    torch.testing.assert_close(out, y_ref, rtol=2e-5, atol=2e-6)
    for row, slot in enumerate(slots.tolist()):
        torch.testing.assert_close(pool[slot], ref_state[row], rtol=2e-5, atol=2e-6)


def test_eight_steps_with_rotating_slots():
    """A single step cannot catch a window-orientation error; eight can.

    If the window were shifted the wrong way, or the new value appended at
    index 0 instead of K-1, step 1 still produces a plausible number — the
    error only compounds once the window has turned over. Rotating the slot
    assignment between steps additionally proves each row's history follows
    its slot rather than its row index.
    """
    mod = load_gdn()
    C, K, B, S = 256, 4, 4, 16
    pool = torch.randn(S, C, K, device="cuda")
    ref_pool = pool.clone()
    w = torch.randn(C, K, device="cuda")
    bias = torch.randn(C, device="cuda")
    out = torch.empty(B, C, device="cuda")

    gen = torch.Generator(device="cuda").manual_seed(3)
    for step in range(8):
        assignment = [(step * 3 + i) % S for i in range(B)]
        slots = torch.tensor(assignment, dtype=torch.int32, device="cuda")
        x = torch.randn(B, C, generator=gen, device="cuda")

        idx = slots.long()
        ref_state = ref_pool[idx].clone()
        y_ref = conv1d_ref(ref_state, x, w, bias)
        ref_pool[idx] = ref_state

        mod.conv1d_decode(pool, slots, x, w, bias, out)

        torch.testing.assert_close(out, y_ref, rtol=2e-5, atol=2e-6,
                                   msg=f"output mismatch at step {step}")
    torch.testing.assert_close(pool, ref_pool, rtol=2e-5, atol=2e-6)


def test_silu_is_applied_to_every_channel():
    """SiLU covers Q, K and V alike, matching causal_conv1d_fn(activation='silu').

    Applying it only to the V block is a plausible reading -- the gate lives on
    the value path -- and produces a model that runs and is wrong.
    """
    mod = load_gdn()
    cfg = GDNConfig.qwen36_35b_a3b()
    C, K = cfg.conv_channels, cfg.conv_kernel
    assert C == 8192 and K == 4

    pool = torch.zeros(2, C, K, device="cuda")
    slots = torch.tensor([0], dtype=torch.int32, device="cuda")
    # Drive every channel to a strongly negative pre-activation: SiLU must
    # squash it toward zero everywhere, including the Q and K blocks.
    x = torch.full((1, C), -6.0, device="cuda")
    w = torch.zeros(C, K, device="cuda")
    w[:, -1] = 1.0
    bias = torch.zeros(C, device="cuda")

    out = torch.empty(1, C, device="cuda")
    mod.conv1d_decode(pool, slots, x, w, bias, out)

    expected = -6.0 * torch.sigmoid(torch.tensor(-6.0))
    assert out.min().item() == pytest.approx(expected.item(), rel=1e-4)
    assert out.max().item() == pytest.approx(expected.item(), rel=1e-4)


def test_graph_replay_survives_slot_reassignment():
    mod = load_gdn()
    C, K, B = 512, 4, 4
    pool = torch.zeros(16, C, K, device="cuda")
    slots = torch.arange(B, dtype=torch.int32, device="cuda")
    x = torch.zeros(B, C, device="cuda")
    w = torch.randn(C, K, device="cuda")
    bias = torch.randn(C, device="cuda")
    out = torch.zeros(B, C, device="cuda")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            mod.conv1d_decode(pool, slots, x, w, bias, out)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mod.conv1d_decode(pool, slots, x, w, bias, out)

    gen = torch.Generator(device="cuda").manual_seed(19)
    for assignment in ([9, 2, 15, 0], [1, 3, 5, 7]):
        pool.normal_(generator=gen)
        x.normal_(generator=gen)
        slots.copy_(torch.tensor(assignment, dtype=torch.int32, device="cuda"))

        ref_state = pool[slots.long()].clone()
        y_ref = conv1d_ref(ref_state, x.clone(), w, bias)

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(out, y_ref, rtol=2e-5, atol=2e-6)
        for row, slot in enumerate(assignment):
            torch.testing.assert_close(pool[slot], ref_state[row], rtol=2e-5, atol=2e-6)
