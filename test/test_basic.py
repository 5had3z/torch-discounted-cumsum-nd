import pytest
import torch
from torch import Tensor

from torch_discounted_cumsum_nd import discounted_cumsum


@pytest.fixture
def basic_data():
    data = torch.randn(4, 124, device="cuda", dtype=torch.float32)
    return data


def test_no_weighting(basic_data: Tensor):
    """Test forward method is correct compared against simple cumsum on last dim"""
    baseline = torch.cumsum(basic_data, dim=-1)
    target = discounted_cumsum(basic_data, dim=1, gamma=1)
    # For small numbers, there can be a difference which is greater than the default
    # tolerance (e.g. in sample of 1024 nelem, 1 is 'not close' with the default atol)
    assert torch.allclose(baseline, target, atol=1e-5)


def test_backward_no_weighting(basic_data: Tensor):
    """Test backward method is correct compared against simple cumsum on last dim"""
    test_grad = torch.empty_like(basic_data)
    baseline_grad = torch.empty_like(basic_data)

    basic_data.requires_grad = True
    handle = basic_data.register_hook(baseline_grad.copy_)
    out = torch.cumsum(basic_data, dim=-1)
    out = out + 1
    out.sum().backward()

    handle.remove()
    handle = basic_data.register_hook(test_grad.copy_)
    out = discounted_cumsum(basic_data, dim=-1, gamma=1)
    out = out + 1
    out.sum().backward()

    assert torch.allclose(baseline_grad, test_grad, atol=1e-5)
