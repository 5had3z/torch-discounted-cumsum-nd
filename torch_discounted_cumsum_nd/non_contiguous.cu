#include "common.cuh"

#include "cooperative_groups.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

template <typename scalar_t>
using TensorAcc3R = torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>;

namespace cg = cooperative_groups;

template <typename T>
union ShMemLayout
{
    WarpScan::TempStorage scan[gThreadBlockDim];
    T tpose[gThreadBlockDim][gThreadBlockDim + 1];
};

template <typename T>
__global__ void __launch_bounds__(gThreadBlockDim* gThreadBlockDim)
    forward_noncontig_kernel(const TensorAcc3R<T> input, TensorAcc3R<T> output, const float invGamma)
{
    __shared__ ShMemLayout<T> smem;

    auto fn = [invGamma](float2 a, float2 b)
    {
        const float c = __powf(invGamma, b.x - a.x);
        b.y = __fmaf_rn(a.y, c, b.y);
        return b;
    };

    const auto tblock = cg::this_thread_block();
    const auto input_ = input[blockIdx.x];
    auto output_ = output[blockIdx.x];

    const int outerDim = blockIdx.y * gThreadBlockDim + threadIdx.x;
    const int scanEnd = ceil_div(input.size(1), gThreadBlockDim) * gThreadBlockDim;

    float warpAgg{0};
    for (int scanDim = threadIdx.y; scanDim < scanEnd; scanDim += gThreadBlockDim)
    {
        T sample = 0;
        const bool isValid = outerDim < input.size(2) && scanDim < input.size(1);
        if (isValid)
        {
            sample = input_[scanDim][outerDim];
        }
        smem.tpose[threadIdx.y][threadIdx.x] = sample;
        tblock.sync();

        float2 data = {
            .x = __int2float_rn(threadIdx.x), .y = __fmaf_rn(invGamma, warpAgg, smem.tpose[threadIdx.x][threadIdx.y])};

        float2 result;
        WarpScan(smem.scan[threadIdx.y]).InclusiveScan(data, result, fn);
        smem.tpose[threadIdx.x][threadIdx.y] = static_cast<T>(result.y);
        tblock.sync();

        if (isValid)
        {
            output_[scanDim][outerDim] = smem.tpose[threadIdx.y][threadIdx.x];
        }
        warpAgg = (threadIdx.x == 0) * __shfl_sync(0xFFFFFFFF, result.y, 31);
    }
}

void forward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output)
{
    TORCH_CHECK_EQ(input.ndimension(), 3);
    TORCH_CHECK_EQ(input.is_contiguous(), true);
    TORCH_CHECK_EQ(output.ndimension(), 3);
    TORCH_CHECK_EQ(output.is_contiguous(), true);

    const dim3 blocksGrid(input.size(0), ceil_div(static_cast<int>(input.size(2)), gThreadBlockDim));
    const dim3 threadsPerBlock(gThreadBlockDim, gThreadBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_forward_cuda",
        [&]()
        {
            auto inputAcc = input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
            auto outputAcc = output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
            forward_noncontig_kernel<scalar_t><<<blocksGrid, threadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
                inputAcc, outputAcc, 1.f / gamma);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
__global__ void __launch_bounds__(gThreadBlockDim* gThreadBlockDim)
    backward_noncontig_kernel(const TensorAcc3R<T> input, TensorAcc3R<T> output, const float invGamma)
{
    __shared__ ShMemLayout<T> smem;

    auto fn = [invGamma](float2 a, float2 b)
    {
        const float c = __powf(invGamma, a.x - b.x);
        b.y = __fmaf_rn(a.y, c, b.y);
        return b;
    };

    const auto tblock = cg::this_thread_block();
    const auto input_ = input[blockIdx.x];
    auto output_ = output[blockIdx.x];

    const int outerDim = blockIdx.y * gThreadBlockDim + threadIdx.x;
    const int scanEnd = ceil_div(input.size(1), gThreadBlockDim) * gThreadBlockDim;

    float warpAgg{0};
    for (int scanDim = scanEnd - threadIdx.y - 1; scanDim >= 0; scanDim -= gThreadBlockDim)
    {
        T sample = 0;
        const bool isValid = outerDim < input.size(2) && scanDim < input.size(1);
        if (isValid)
        {
            sample = input_[scanDim][outerDim];
        }
        smem.tpose[threadIdx.y][threadIdx.x] = sample;
        tblock.sync();

        float2 data = {.x = __int2float_rn(gThreadBlockDim - threadIdx.x),
            .y = __fmaf_rn(invGamma, warpAgg, smem.tpose[threadIdx.x][threadIdx.y])};

        float2 result;
        WarpScan(smem.scan[threadIdx.y]).InclusiveScan(data, result, fn);
        smem.tpose[threadIdx.x][threadIdx.y] = static_cast<T>(result.y);
        tblock.sync();

        if (isValid)
        {
            output_[scanDim][outerDim] = smem.tpose[threadIdx.y][threadIdx.x];
        }
        warpAgg = (threadIdx.x == 0) * __shfl_sync(0xFFFFFFFF, result.y, 31);
    }
}

void backward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output)
{
    TORCH_CHECK_EQ(input.ndimension(), 3);
    TORCH_CHECK_EQ(input.is_contiguous(), true);
    TORCH_CHECK_EQ(output.ndimension(), 3);
    TORCH_CHECK_EQ(output.is_contiguous(), true);

    const dim3 blocksGrid(input.size(0), ceil_div(static_cast<int>(input.size(2)), gThreadBlockDim));
    const dim3 threadsPerBlock(gThreadBlockDim, gThreadBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_cuda",
        [&]()
        {
            auto inputAcc = input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
            auto outputAcc = output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
            backward_noncontig_kernel<scalar_t><<<blocksGrid, threadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
                inputAcc, outputAcc, 1.f / gamma);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
