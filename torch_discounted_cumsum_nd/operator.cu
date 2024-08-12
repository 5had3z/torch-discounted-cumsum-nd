#include <cub/warp/warp_scan.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

#include <execution>

template <typename scalar_t>
using TensorAcc2R = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;

constexpr auto gThreadBlockDim = 32;

template <typename integer>
constexpr inline integer ceil_div(integer n, integer m)
{
    return (n + m - 1) / m;
}

[[nodiscard]] auto getForwardPermutation(int64_t ndim, int64_t dim) -> std::vector<int64_t>
{
    std::vector<int64_t> dims;
    dims.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i)
    {
        if (i != dim)
        {
            dims.emplace_back(i);
        }
    }
    dims.emplace_back(dim);
    return dims;
}

[[nodiscard]] auto getReversePermutation(int64_t ndim, int64_t dim) -> std::vector<int64_t>
{
    // Allocate target size then remove last element by resizing
    std::vector<int64_t> dims(ndim);
    dims.resize(dims.size() - 1);
    std::iota(dims.begin(), dims.end(), 0);
    dims.insert(dims.begin() + dim, ndim - 1);
    return dims;
}

[[nodiscard]] auto prepareInput(const torch::Tensor& input, int64_t dim) -> torch::Tensor
{
    torch::Tensor input_ = input;
    if (dim != (input.ndimension() - 1))
    {
        input_ = input_.permute(getForwardPermutation(input.ndimension(), dim));
    }
    return input_.flatten(0, -2).contiguous();
}

[[nodiscard]] auto getPermutedShape(c10::IntArrayRef inShape, int64_t dim) -> std::vector<int64_t>
{
    std::vector permShape(inShape.begin(), inShape.end());
    if (dim != inShape.size() - 1)
    {
        const auto targetDim = permShape.begin() + dim;
        std::rotate(std::execution::unseq, targetDim, targetDim + 1, permShape.end());
    }
    return permShape;
}

void restoreOutputShape(torch::Tensor& output, c10::IntArrayRef inShape, int64_t dim)
{
    output = output.reshape(getPermutedShape(inShape, dim));
    if (dim != (output.ndimension() - 1))
    {
        output = output.permute(getReversePermutation(output.ndimension(), dim));
    }
    output = output.contiguous();
}

TORCH_LIBRARY(discounted_cumsum, m)
{
    m.def("discounted_cumsum(Tensor input, int dim, float gamma) -> Tensor");
    m.def("_discounted_cumsum_bw(Tensor input, int dim, float gamma) -> Tensor");
}

using WarpScan = cub::WarpScan<float2>;
using StorageT = typename WarpScan::TempStorage;

template <typename T>
__global__ void forward_kernel(const T* __restrict__ inPtr, int64_t inPitch, T* __restrict__ outPtr, int64_t outPitch,
    const float inv_gamma, const int64_t scanDimSize, const int64_t totalBatch)
{
    extern __shared__ StorageT tempStorage[];
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

auto forward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor inputFlat = prepareInput(input, dim);
    auto output = torch::zeros_like(inputFlat);

    const auto maxBlockSize = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock / 2;
    const auto batchSize = static_cast<int>(inputFlat.size(0));
    const int yBlockDim = std::min(maxBlockSize / gThreadBlockDim, batchSize);
    const dim3 blocksGrid(ceil_div(batchSize, yBlockDim));
    const dim3 threadsPerBlock(gThreadBlockDim, yBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_forward_cuda",
        [&]()
        {
            // smem = yBlockDim since sizeof(StorageT) = 1
            forward_kernel<scalar_t><<<blocksGrid, threadsPerBlock, yBlockDim, at::cuda::getCurrentCUDAStream()>>>(
                inputFlat.const_data_ptr<scalar_t>(), inputFlat.stride(0), output.mutable_data_ptr<scalar_t>(),
                output.stride(0), 1.f / gamma, inputFlat.size(1), inputFlat.size(0));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    restoreOutputShape(output, input.sizes(), dim);
    return output;
}

template <typename T>
__global__ void backward_kernel(const T* __restrict__ inPtr, int64_t inPitch, T* __restrict__ outPtr, int64_t outPitch,
    const float inv_gamma, const int64_t scanDimSize, const int64_t totalBatch)
{
    extern __shared__ StorageT tempStorage[];
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

auto backward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor inputFlat = prepareInput(input, dim);
    auto output = torch::zeros_like(inputFlat);

    const auto maxBlockSize = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock / 2;
    const auto batchSize = static_cast<int>(inputFlat.size(0));
    const int yBlockDim = std::min(maxBlockSize / gThreadBlockDim, batchSize);
    const dim3 blocksGrid(ceil_div(batchSize, yBlockDim));
    const dim3 threadsPerBlock(gThreadBlockDim, yBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_cuda",
        [&]()
        {
            // smem = yBlockDim since sizeof(StorageT) = 1
            backward_kernel<scalar_t><<<blocksGrid, threadsPerBlock, yBlockDim, at::cuda::getCurrentCUDAStream()>>>(
                inputFlat.const_data_ptr<scalar_t>(), inputFlat.stride(0), output.mutable_data_ptr<scalar_t>(),
                output.stride(0), 1.f / gamma, inputFlat.size(1), inputFlat.size(0));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    restoreOutputShape(output, input.sizes(), dim);
    return output;
}

TORCH_LIBRARY_IMPL(discounted_cumsum, CUDA, m)
{
    m.impl("discounted_cumsum", &forward_cuda);
    m.impl("_discounted_cumsum_bw", &backward_cuda);
}

auto forward_cpu(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor inputFlat = prepareInput(input, dim);
    auto output = torch::zeros_like(inputFlat);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_forward_cpu",
        [&]()
        {
            for (auto bidx = 0; bidx < inputFlat.size(0); ++bidx)
            {
                const auto inPtr = static_cast<scalar_t*>(inputFlat.data_ptr()) + bidx * inputFlat.stride(0);
                auto outPtr = static_cast<scalar_t*>(output.data_ptr()) + bidx * output.stride(0);
                std::inclusive_scan(std::execution::unseq, inPtr, inPtr + inputFlat.stride(0), outPtr,
                    [&](scalar_t a, scalar_t b) -> scalar_t
                    { return std::fma(a, static_cast<scalar_t>(1 / gamma), b); });
            }
        });

    restoreOutputShape(output, input.sizes(), dim);
    return output;
}

auto backward_cpu(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor inputFlat = prepareInput(input, dim);
    auto output = torch::zeros_like(inputFlat);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_cpu",
        [&]()
        {
            for (auto bidx = 0; bidx < inputFlat.size(0); ++bidx)
            {
                const auto inPtr = static_cast<scalar_t*>(inputFlat.data_ptr()) + bidx * inputFlat.stride(0);
                auto outPtr = static_cast<scalar_t*>(output.data_ptr()) + bidx * output.stride(0);
                std::inclusive_scan(std::execution::unseq, std::make_reverse_iterator(inPtr + inputFlat.stride(0)),
                    std::make_reverse_iterator(inPtr), std::make_reverse_iterator(outPtr + output.stride(0)),
                    [&](scalar_t a, scalar_t b) -> scalar_t
                    { return std::fma(a, static_cast<scalar_t>(1 / gamma), b); });
            }
        });

    restoreOutputShape(output, input.sizes(), dim);
    return output;
}

TORCH_LIBRARY_IMPL(discounted_cumsum, CPU, m)
{
    m.impl("discounted_cumsum", &forward_cpu);
    m.impl("_discounted_cumsum_bw", &backward_cpu);
}
