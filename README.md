# torch-discounted-cumsum-nd

## Currently still WIP and not thoroughly tested - CPU has error when target dim isn't last dim.

## Installation

This Package utilizes the new Pytorch 2.4 API for C++/Autograd bindings, hence that is a must for compatibility. Otherwise should be installable with setup.py...

```bash
python3 setup.py install --user
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


## General Notes

Simple auto-grad supported operation that applies a weighted inclusive-scan on the input data across dimension `D` of a tensor of any number of dimension. I designed the implementation with my own selfish needs foremost, hence it is optimised for `D << other dims`. The CUDA implementation iterates over the target dimension with cub::WarpScan so should be reasonably quick. Block-Parallelism is mainly utilized used batch over the other dimensions. Future work to target the case where `D` is not small could use cub::BlockScan instead. CPU implementation uses std::inclusive_scan with std::execution::unseq so you'll need TBB or whatever other std::execution provider installed on your machine.

$$ \text{{out}}_i = \sum_{j=0}^{i} (\frac{1}{\gamma})^{i-j}\text{in}_j.$$

```python
def discounted_cumsum(x: Tensor, dim: int = -1, gamma: float = 2) -> Tensor:
    r"""
    Discounted cumsum where each element is calculated with the formula
    .. math::
        \text{{out}}_i = \sum_{j=0}^{i} (\frac{1}{\gamma})^{i-j}\text{in}_j.

    Args:
        x (Tensor): N-D Tensor to apply operation
        dim (int, optional): Dimension to apply discounted cumsum over. Defaults to -1.
        discount (float, optional): Gamma factor to the cumsum. Defaults to 2.

    Returns:
        Tensor: Discounted cumsum result
    """
    return torch.ops.discounted_cumsum.discounted_cumsum(x, dim, gamma)
```
