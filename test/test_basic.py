from typing import Sequence
import pytest
import torch

from torch_discounted_cumsum_nd import discounted_cumsum


def make_data(shape: Sequence[int], device: str):
    """Create random data from shape and device"""
    return torch.randn(shape, device=device, dtype=torch.float32)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_no_weighting(device: str, dim: int):
    """Test forward method is correct compared against simple cumsum on last dim"""
    data = make_data((4, 124), device)
    baseline = torch.cumsum(data, dim=dim)
    target = discounted_cumsum(data, dim=dim, gamma=1)
    # For small numbers, there can be a difference which is greater than the default
    # tolerance (e.g. in sample of 1024 nelem, 1 is 'not close' with the default atol)
    assert torch.allclose(baseline, target, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_backward_no_weighting(device: str, dim: int):
    """Test backward method is correct compared against simple cumsum on last dim"""
    data = make_data((4, 124), device)
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
