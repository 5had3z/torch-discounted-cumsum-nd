from typing import Sequence
import pytest
import torch

from torch_discounted_cumsum_nd import discounted_cumsum


def make_data(shape: Sequence[int], device: str):
    """Create random data from shape and device"""
    return torch.randn(shape, device=device, dtype=torch.float32)


_TEST_DEVICES = ["cpu", "cuda"]
_TEST_SHAPES = [(4, 32), (12, 128, 30), (4, 32, 64), (4, 128, 99)]


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("device", _TEST_DEVICES)
@pytest.mark.parametrize("shape", _TEST_SHAPES)
def test_no_weighting_last_dim(dim: int, device: str, shape: tuple[int, ...]):
    """Test forward method is correct compared against simple cumsum on last dim"""
    data = make_data(shape, device)
    baseline = torch.cumsum(data, dim=dim)
    target = discounted_cumsum(data, dim=dim, gamma=1)
    # For small numbers, there can be a difference which is greater than the default
    # tolerance (e.g. in sample of 1024 nelem, 1 is 'not close' with the default atol)
    assert torch.allclose(baseline, target, atol=1e-5)


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("device", _TEST_DEVICES)
@pytest.mark.parametrize("shape", _TEST_SHAPES)
def test_backward_no_weighting(dim: int, device: str, shape: tuple[int, ...]):
    """Test backward method is correct compared against simple cumsum on last dim"""
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
