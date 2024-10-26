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
    T tpose[gThreadBlockDim][gThreadBlockDim];
};

template <typename T>
__global__ void forward_noncontig_kernel(const TensorAcc3R<T> input, TensorAcc3R<T> output, const float invGamma)
{
    extern __shared__ __align__(alignof(ShMemLayout<T>)) char smem[];
    auto& blockTemp = reinterpret_cast<float(&)[gThreadBlockDim][gThreadBlockDim]>(smem);

    auto fn = [invGamma](float2 a, float2 b)
    {
        const float c = __powf(invGamma, b.x - a.x);
        b.y = __fmaf_rn(a.y, c, b.y);
        return b;
    };

    const auto tblock = cg::this_thread_block();
    const auto input_ = input[blockIdx.x];
    auto output_ = output[blockIdx.x];

    for (int outerDim = threadIdx.y; outerDim < input.size(2); outerDim += gThreadBlockDim)
    {
        float warpAgg{0};
        for (int scanDim = threadIdx.x; scanDim < input.size(1); scanDim += gThreadBlockDim)
        {
            blockTemp[threadIdx.y][threadIdx.x] = input_[scanDim][outerDim];
            tblock.sync();

            float2 data = {
                .x = __int2float_rn(scanDim), .y = __fmaf_rn(invGamma, warpAgg, blockTemp[threadIdx.x][threadIdx.y])};

            float2 result;
            WarpScan(reinterpret_cast<WarpScan::TempStorage(&)[gThreadBlockDim]>(smem)[threadIdx.y])
                .InclusiveScan(data, result, fn);
            blockTemp[threadIdx.x][threadIdx.y] = static_cast<T>(result.y);
            tblock.sync();

            output_[scanDim][outerDim] = blockTemp[threadIdx.y][threadIdx.x];
            warpAgg = (threadIdx.x == 0) * __shfl_sync(0xFFFFFFFF, result.y, 31);
        }
    }
}

void forward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output)
{
    TORCH_CHECK_EQ(input.ndimension(), 3);
    TORCH_CHECK_EQ(input.is_contiguous(), true);
    TORCH_CHECK_EQ(output.ndimension(), 3);
    TORCH_CHECK_EQ(output.is_contiguous(), true);

    const dim3 blocksGrid(input.size(0));
    const dim3 threadsPerBlock(gThreadBlockDim, gThreadBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_forward_cuda",
        [&]()
        {
            auto inputAcc = input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
            auto outputAcc = output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
            forward_noncontig_kernel<scalar_t>
                <<<blocksGrid, threadsPerBlock, sizeof(ShMemLayout<scalar_t>), at::cuda::getCurrentCUDAStream()>>>(
                    inputAcc, outputAcc, 1.f / gamma);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output) {}