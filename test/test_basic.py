"""
Tests to check cpu/gpu with gamma=1 agrees with torch.cumsum with different shapes and dims.
For gamma != 1, test that both cpu and gpu implementations agree.

For small numbers, there can be a difference which is greater than the default
tolerance (e.g. in sample of 1024 nelem, 1 is 'not close' with the default atol)
"""

from typing import Sequence
import pytest
import torch
from torch import Tensor

from torch_discounted_cumsum_nd import discounted_cumsum


def make_data(shape: Sequence[int], device: str):
    """Create random data from shape and device"""
    return torch.randn(shape, device=device, dtype=torch.float32)


_TEST_DEVICES = ["cpu", "cuda"]
_TEST_SHAPES = [(4, 32), (12, 128, 30), (4, 32, 64), (4, 101, 99)]


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("device", _TEST_DEVICES)
@pytest.mark.parametrize("shape", _TEST_SHAPES)
def test_forward_no_weighting(dim: int, device: str, shape: tuple[int, ...]):
    """Test forward method is correct compared against torch.cumsum with gamma=1"""
    data = make_data(shape, device)
    baseline = torch.cumsum(data, dim=dim)
    target = discounted_cumsum(data, dim=dim, gamma=1)
    assert torch.allclose(baseline, target, atol=1e-5)


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("device", _TEST_DEVICES)
@pytest.mark.parametrize("shape", _TEST_SHAPES)
def test_backward_no_weighting(dim: int, device: str, shape: tuple[int, ...]):
    """Test backward method is correct compared against torch.cumsum with gamma=1"""
    data = make_data(shape, device)
    test_grad = torch.empty_like(data)
    baseline_grad = torch.empty_like(data)

    data.requires_grad = True
    handle = data.register_hook(baseline_grad.copy_)
    out = torch.cumsum(data, dim=dim)
    out = out + 1
    out.sum().backward()

    handle.remove()
    handle = data.register_hook(test_grad.copy_)
    out = discounted_cumsum(data, dim=dim, gamma=1)
    out = out + 1
    out.sum().backward()

    assert torch.allclose(baseline_grad, test_grad, atol=1e-5)


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("shape", _TEST_SHAPES)
@pytest.mark.parametrize("gamma", [2, 5])
def test_gamma_factor(dim: int, gamma: float, shape: tuple[int, ...]):
    """Check both CPU and GPU implementations agree with the result.
    CPU impl is more trustworthy as it is a simple serial algorithm."""
    fw_list: list[Tensor] = []
    bw_list: list[Tensor] = []

    data = make_data(shape, device="cpu")
    data.requires_grad = True
    for device in _TEST_DEVICES:
        data = data.to(device=device)
        grad = torch.empty_like(data)
        handle = data.register_hook(grad.copy_)
        out = discounted_cumsum(data, dim=dim, gamma=gamma)
        fw_list.append(out)
        out = out + 1
        out.sum().backward()
        bw_list.append(grad)
        handle.remove()

    for a, b in [fw_list, bw_list]:
        assert torch.allclose(a.cpu(), b.cpu(), atol=1e-5)
