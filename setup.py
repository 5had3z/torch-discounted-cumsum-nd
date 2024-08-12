#!/usr/bin/env python3
"""
Change the NVCC args to the correct SM you have and install with the command
> python3 setup.py install
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == "__main__":
    setup(
        ext_modules=[
            CUDAExtension(
                "torch_discounted_cumsum_nd",
                ["torch_discounted_cumsum_nd/operator.cu"],
                extra_compile_args={
                    "cxx": ["-O2", "-std=c++17"],
                    "nvcc": ["-O2", "-std=c++17"],
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
