"""The hook must move exactly the intended positions by exactly a * scale."""
import pytest
import torch

from steer.hooks import PositionSteerer

D, P, SCALE = 8, 5, 3.0


class Block(torch.nn.Module):
    def forward(self, x):
        return (x, None)  # HF blocks return tuples


def run(mode, k=0, coeffs=2.0, cached=True, n_new=6, B=2):
    """Return per-position shift norms [B, P+n_new-1] for one simulated generate."""
    block, direction = Block(), torch.randn(D)
    st = PositionSteerer(block, direction, SCALE, coeffs, mode=mode, k=k,
                         prompt_len=None if cached else P)
    T = P + n_new - 1
    x = torch.randn(B, T, D)
    with st:
        if cached:  # prompt in one call, then one position per call
            outs = [block(x[:, :P])[0]] + [block(x[:, t : t + 1])[0] for t in range(P, T)]
            y = torch.cat(outs, dim=1)
        else:  # one full-sequence call
            y = block(x)[0]
    assert block._forward_hooks == {}  # hook removed on exit
    return (y - x).norm(dim=-1)


def expected(mode, k, T):
    pos = torch.arange(T)
    g0 = P - 1  # last prompt token = generation position 1
    return {
        "all": pos >= 0,
        "none": pos < 0,
        "prompt": pos < g0,
        "response": pos >= g0,
        "first_k": (pos >= g0) & (pos < g0 + k),
        "after_k": pos >= g0 + k,
    }[mode]


@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("mode,k", [("all", 0), ("none", 0), ("prompt", 0), ("response", 0),
                                    ("first_k", 1), ("first_k", 3), ("after_k", 0), ("after_k", 2)])
def test_positions_and_size(mode, k, cached):
    shift = run(mode, k, cached=cached)
    want = expected(mode, k, shift.shape[1]).float() * 2.0 * SCALE
    assert torch.allclose(shift, want.expand_as(shift), atol=1e-4)


def test_partition():
    """prompt + response == all; first_k + after_k == response."""
    assert torch.allclose(run("prompt") + run("response"), run("all"), atol=1e-4)
    assert torch.allclose(run("first_k", 3) + run("after_k", 3), run("response"), atol=1e-4)


def test_per_row_strengths_and_zero():
    shift = run("all", coeffs=torch.tensor([0.0, 0.5]))
    assert torch.all(shift[0] == 0)
    assert torch.allclose(shift[1], torch.full_like(shift[1], 0.5 * SCALE), atol=1e-4)


def test_reusable():
    block = Block()
    st = PositionSteerer(block, torch.randn(D), SCALE, 1.0, mode="prompt")
    for _ in range(2):  # second use must re-infer the prompt length
        with st:
            y = block(torch.zeros(1, P, D))[0]
        assert (y.norm(dim=-1) > 0).sum() == P - 1
