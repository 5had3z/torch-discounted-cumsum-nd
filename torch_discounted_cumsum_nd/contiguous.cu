#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

template <typename T>
__global__ void forward_contiguous_kernel(const T* __restrict__ inPtr, int64_t inPitch, T* __restrict__ outPtr,
    int64_t outPitch, const float inv_gamma, const int64_t scanDimSize, const int64_t totalBatch)
{
    extern __shared__ WarpScan::TempStorage tempStorage[];
    auto fn = [inv_gamma](float2 a, float2 b)
    {
        const float c = __powf(inv_gamma, b.x - a.x);
        b.y = __fmaf_rn(a.y, c, b.y);
        return b;
    };

    const auto batchOffset = blockIdx.x * blockDim.y + threadIdx.y;
    const bool hasWork = batchOffset < totalBatch;

    const auto inOffset = inPtr + inPitch * batchOffset;
    const auto outOffset = outPtr + outPitch * batchOffset;

    float warp_agg{0};
    for (int idx = threadIdx.x; idx < scanDimSize; idx += gThreadBlockDim)
    {
        if (hasWork)
        {
            float data = static_cast<float>(inOffset[idx]);
            data += (threadIdx.x == 0) * inv_gamma * warp_agg;
            float2 result;
            WarpScan(tempStorage[threadIdx.y]).InclusiveScan(float2{__int2float_rn(idx), data}, result, fn);
            outOffset[idx] = static_cast<T>(result.y);
            warp_agg = __shfl_sync(0xFFFFFFFF, result.y, 31);
        }
    }
}

void forward_cuda_contig(const torch::Tensor& inputFlat, double gamma, torch::Tensor& output)
{
    const auto maxBlockSize = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock / 2;
    const auto batchSize = static_cast<int>(inputFlat.size(0));
    const int yBlockDim = std::min(maxBlockSize / gThreadBlockDim, batchSize);
    const dim3 blocksGrid(ceil_div(batchSize, yBlockDim));
    const dim3 threadsPerBlock(gThreadBlockDim, yBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputFlat.scalar_type(), "discounted_cumsum_forward_cuda",
        [&]()
        {
            // smem = yBlockDim since sizeof(StorageT) = 1
            forward_contiguous_kernel<scalar_t>
                <<<blocksGrid, threadsPerBlock, yBlockDim, at::cuda::getCurrentCUDAStream()>>>(
                    inputFlat.const_data_ptr<scalar_t>(), inputFlat.stride(0), output.mutable_data_ptr<scalar_t>(),
                    output.stride(0), 1.f / gamma, inputFlat.size(1), inputFlat.size(0));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
__global__ void backward_contiguous_kernel(const T* __restrict__ inPtr, int64_t inPitch, T* __restrict__ outPtr,
    int64_t outPitch, const float inv_gamma, const int64_t scanDimSize, const int64_t totalBatch)
{
    extern __shared__ WarpScan::TempStorage tempStorage[];
    auto fn = [inv_gamma](float2 a, float2 b)
    {
        const float c = __powf(inv_gamma, a.x - b.x);
        b.y = __fmaf_rn(a.y, c, b.y);
        return b;
    };

    const auto batchOffset = blockIdx.x * blockDim.y + threadIdx.y;
    const bool hasWork = batchOffset < totalBatch;

    const auto inOffset = inPtr + inPitch * batchOffset;
    const auto outOffset = outPtr + outPitch * batchOffset;

    float warp_agg{0};
    for (int idx = scanDimSize - threadIdx.x - 1; idx >= 0; idx -= gThreadBlockDim)
    {
        if (hasWork)
        {
            float data = static_cast<float>(inOffset[idx]);
            data += (threadIdx.x == 0) * inv_gamma * warp_agg;
            float2 result;
            WarpScan(tempStorage[threadIdx.y]).InclusiveScan({__int2float_rn(idx), data}, result, fn);
            outOffset[idx] = static_cast<T>(result.y);
            warp_agg = __shfl_sync(0xFFFFFFFF, result.y, 31);
        }
    }
}

void backward_cuda_contig(const torch::Tensor& inputFlat, double gamma, torch::Tensor& output)
{
    const auto maxBlockSize = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock / 2;
    const auto batchSize = static_cast<int>(inputFlat.size(0));
    const int yBlockDim = std::min(maxBlockSize / gThreadBlockDim, batchSize);
    const dim3 blocksGrid(ceil_div(batchSize, yBlockDim));
    const dim3 threadsPerBlock(gThreadBlockDim, yBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputFlat.scalar_type(), "discounted_cumsum_backward_cuda",
        [&]()
        {
            // smem = yBlockDim since sizeof(StorageT) = 1
            backward_contiguous_kernel<scalar_t>
                <<<blocksGrid, threadsPerBlock, yBlockDim, at::cuda::getCurrentCUDAStream()>>>(
                    inputFlat.const_data_ptr<scalar_t>(), inputFlat.stride(0), output.mutable_data_ptr<scalar_t>(),
                    output.stride(0), 1.f / gamma, inputFlat.size(1), inputFlat.size(0));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}