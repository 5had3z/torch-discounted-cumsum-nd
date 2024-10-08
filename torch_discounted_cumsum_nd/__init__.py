"""Pytorch Discounted Cumsum ND Bindings"""

from pathlib import Path

import torch
from torch import Tensor

# Grab binding from goofy-ahh folder above after installation
torch.ops.load_library(
    next(
        str(p)
        for p in Path(__file__).parent.parent.iterdir()
        if p.suffix == ".so" and p.stem.startswith("torch_discounted_cumsum_nd")
    )
)


def _backward(ctx, grad: Tensor):
    """Backward"""
    g = torch.ops.discounted_cumsum._discounted_cumsum_bw(grad, ctx.dim, ctx.gamma)
    return g, None, None


def _setup_context(ctx, inputs: tuple[Tensor, int, float], output: Tensor):
    """Save target dimension and discount factor for backward"""
    _, ctx.dim, ctx.gamma = inputs


torch.library.register_autograd(
    "discounted_cumsum::discounted_cumsum", _backward, setup_context=_setup_context
)


def discounted_cumsum(x: Tensor, dim: int = -1, gamma: float = 2) -> Tensor:
    r"""
    Discounted cumsum where each element is calculated with the formula
    .. math::
        \text{{out}}_i = \sum_{j=0}^{i} (\frac{1}{\gamma})^{i-j}\text{in}_j.
    Gamma == 1 is a normal cumsum, gamma < 1 will blow up quickly so is disabled.

    Args:
        x (Tensor): N-D Tensor to apply operation
        dim (int, optional): Dimension to apply discounted cumsum over. Defaults to -1.
        discount (float, optional): Gamma factor to the cumsum, must be >=1. Defaults to 2.

    Returns:
        Tensor: Discounted cumsum result
    """
    assert gamma >= 1, "Gamma should be >=1, you'll get inf/nan quickly otherwise"
    return torch.ops.discounted_cumsum.discounted_cumsum(x, dim, gamma)
