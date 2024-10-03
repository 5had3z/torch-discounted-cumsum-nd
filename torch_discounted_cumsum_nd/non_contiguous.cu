#include "common.cuh"

#include <cub/block/block_load.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

template <typename T>
__global__ void forward_noncontig_kernel(const T* __restrict__ inPtr, const int64_t inOuterPitch,
    const int64_t inInnerPitch, T* __restrict__ outPtr, const int64_t outOuterPitch, int64_t outInnerPitch,
    const float inv_gamma, const int64_t scanDimSize, const int64_t outerBatch)
{
    using BlockExchange = cub::BlockExchange<float, 32, 1, false, 32>;
    using BlockTemp = typename BlockExchange::TempStorage;

    union ShMemLayout
    {
        WarpScan::TempStorage warp;
        BlockTemp block;
    };

    extern __shared__ __align__(alignof(ShMemLayout)) char smem[];
    auto& blockTemp = reinterpret_cast<BlockTemp&>(smem);

    auto fn = [inv_gamma](float2 a, float2 b)
    {
        const float c = __powf(inv_gamma, b.x - a.x);
        b.y = __fmaf_rn(a.y, c, b.y);
        return b;
    };

    float data[1];
    data[0] = static_cast<float>(*inPtr);
    BlockExchange(blockTemp).WarpStripedToBlocked(data);
}

void forward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output)
{
    TORCH_CHECK_EQ(input.ndimension(), 3);
    TORCH_CHECK_EQ(output.ndimension(), 3);

    const auto maxBlockSize = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    const auto batchSize = static_cast<int>(input.size(0));
    const int yBlockDim = std::min(maxBlockSize / gThreadBlockDim, batchSize);
    const dim3 blocksGrid(ceil_div(batchSize, yBlockDim));
    const dim3 threadsPerBlock(gThreadBlockDim, yBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_forward_cuda",
        [&]()
        {
            // smem = yBlockDim since sizeof(StorageT) = 1
            forward_noncontig_kernel<scalar_t>
                <<<blocksGrid, threadsPerBlock, yBlockDim, at::cuda::getCurrentCUDAStream()>>>(
                    input.const_data_ptr<scalar_t>(), input.stride(0), input.stride(1),
                    output.mutable_data_ptr<scalar_t>(), output.stride(0), output.stride(1), 1.f / gamma, input.size(1),
                    input.size(0));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output) {}