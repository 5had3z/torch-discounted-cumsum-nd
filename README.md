# torch-discounted-cumsum-nd

$$y_i = \sum_{j=0}^{i} (\frac{1}{\gamma})^{i-j}x_j$$

## Installation

This package utilizes the new Pytorch 2.4 API for C++/Autograd bindings, hence that is a must for compatibility. Otherwise should be installable with pip, preferably with `--no-build-isolation` so that an entire copy of PyTorch isn't downloaded to build this, rather your usual global version is used.

```bash
pip3 install . --no-build-isolation
```

..and then used by importing the function from the package.

```python
import torch
from torch_discounted_cumsum_nd import discounted_cumsum

dummy = torch.arange(32, 96, device="cuda", dtype=torch.float32, requires_grad=True)
dummy = dummy[None].repeat(2, 1)
dummy.register_hook(lambda x: print(f"backward: {x}"))
ret = discounted_cumsum(dummy, dim=-1, gamma=2)
print(f"forward: {ret}")
ret = ret + 1
ret.sum().backward()
```


## Usage Notes

Simple auto-grad supported operation that applies a weighted inclusive-scan on the input data across dimension `D` of a tensor of any number of dimension. I designed the implementation with my own selfish needs foremost, hence it is optimised for `D << other dims`. The CUDA implementation iterates over the target dimension with cub::WarpScan so should be reasonably quick. Block-Parallelism is mainly utilized used batch over the other dimensions. Future work to target the case where `D` is not small could use cub::BlockScan instead. CPU implementation uses std::inclusive_scan with std::execution::unseq so you'll need TBB or whatever other std::execution provider installed on your machine.

```python
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
```

## Other Notes

To validate correctness of the inclusive-scan operation over multiple shapes and dimensions, the program is checked against a normal cumsum with gamma=1. To validate the gamma weighting is producing the correct result, the CUDA implementation is checked against the simpler CPU implementation.

There is a CMakeLists that builds a standalone CUDA/C++ program to perform a simple profile on the operation with Nsight-Compute. This will inform how tweaks to the kernel impact performance and hopefully improve performance closer to memcpy speed.
