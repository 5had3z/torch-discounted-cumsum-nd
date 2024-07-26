import pytest
import torch
from torch import Tensor

from torch_discounted_cumsum_nd import discounted_cumsum


@pytest.fixture
def basic_data():
    data = torch.arange(0, 256, device="cuda")
    data = data.reshape(4, 256 // 4)
    return data


def test_no_weighting(basic_data: Tensor):
    baseline = torch.cumsum(basic_data, dim=-1)
    target = discounted_cumsum(basic_data, discount=1)
    assert torch.isclose(baseline, target).all()
