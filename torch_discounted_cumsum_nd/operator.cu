#include <cub/block/block_scan.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

#include <execution>

template <typename scalar_t>
using TensorAcc2R = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;

constexpr auto gThreadBlockDim = 32;

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

struct P
{
    float i; // index
    float v; // value
};
using WarpScan = cub::WarpScan<P>;
using StorageT = typename WarpScan::TempStorage;

template <typename T>
__global__ void forward_kernel(const TensorAcc2R<T> input, TensorAcc2R<T> output, float inv_gamma, int64_t scanDimSize)
{
    __shared__ StorageT tempStorage;

    P warp_agg{0, 0};
    for (auto idx = threadIdx.x; idx < scanDimSize; idx += gThreadBlockDim)
    {
        float data = static_cast<float>(input[blockIdx.x][idx]);
        data += (threadIdx.x == 0) * inv_gamma * warp_agg.v;
        P pair{static_cast<float>(idx), data};
        auto fn = [&](const P& a, const P& b)
        {
            float c = powf(inv_gamma, b.i - a.i);
            return P{b.i, fma(a.v, c, b.v)};
        };
        P result;
        WarpScan(tempStorage).InclusiveScan(pair, result, fn, warp_agg);
        output[blockIdx.x][idx] = static_cast<T>(result.v);
        __syncwarp();
    }
}

auto forward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor input_ = prepareInput(input, dim);
    auto output_ = torch::zeros_like(input_);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocksGrid(static_cast<unsigned int>(input_.size(0)));
    const dim3 threadsPerBlock(gThreadBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_forward_cuda",
        [&]()
        {
            auto input_acc = input_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto output_acc = output_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            forward_kernel<scalar_t>
                <<<blocksGrid, threadsPerBlock, 0, stream>>>(input_acc, output_acc, 1.f / gamma, input_.size(1));
        });

    restoreOutputShape(output_, input.sizes(), dim);
    return output_;
}

template <typename T>
__global__ void backward_kernel(const TensorAcc2R<T> input, TensorAcc2R<T> output, float inv_gamma, int64_t scanDimSize)
{
    __shared__ StorageT tempStorage;

    P warp_agg{0, 0};
    for (int idx = scanDimSize - threadIdx.x - 1; idx >= 0; idx -= gThreadBlockDim)
    {
        float data = static_cast<float>(input[blockIdx.x][idx]);
        data += (threadIdx.x == 0) * inv_gamma * warp_agg.v;
        P pair{static_cast<float>(idx), data};
        auto fn = [&](const P& a, const P& b)
        {
            float c = powf(inv_gamma, a.i - b.i);
            return P{b.i, fma(a.v, c, b.v)};
        };
        P result;
        WarpScan(tempStorage).InclusiveScan(pair, result, fn, warp_agg);
        output[blockIdx.x][idx] = static_cast<T>(result.v);
        __syncwarp();
    }
}

auto backward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor input_ = prepareInput(input, dim);
    auto output_ = torch::zeros_like(input_);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocksGrid(static_cast<unsigned int>(input_.size(0)));
    const dim3 threadsPerBlock(gThreadBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_cuda",
        [&]()
        {
            auto input_acc = input_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto output_acc = output_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            backward_kernel<scalar_t>
                <<<blocksGrid, threadsPerBlock, 0, stream>>>(input_acc, output_acc, 1.f / gamma, input_.size(1));
        });

    restoreOutputShape(output_, input.sizes(), dim);
    return output_;
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
    torch::Tensor input_ = prepareInput(input, dim);
    auto output_ = torch::zeros_like(input_);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_cpu",
        [&]()
        {
            for (auto bidx = 0; bidx < input_.size(0); ++bidx)
            {
                const auto inPtr = static_cast<scalar_t*>(input_.data_ptr()) + bidx * input_.stride(0);
                auto outPtr = static_cast<scalar_t*>(output_.data_ptr()) + bidx * output_.stride(0);
                std::inclusive_scan(std::execution::unseq, inPtr, inPtr + input.stride(0), outPtr,
                    [&](scalar_t a, scalar_t b) -> scalar_t
                    { return std::fma(a, static_cast<scalar_t>(1 / gamma), b); });
            }
        });

    restoreOutputShape(output_, input.sizes(), dim);
    return output_;
}

auto backward_cpu(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor input_ = prepareInput(input, dim);
    auto output_ = torch::zeros_like(input_);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_cpu",
        [&]()
        {
            for (auto bidx = 0; bidx < input_.size(0); ++bidx)
            {
                const auto inPtr = static_cast<scalar_t*>(input_.data_ptr()) + bidx * input_.stride(0);
                auto outPtr = static_cast<scalar_t*>(output_.data_ptr()) + bidx * output_.stride(0);
                std::inclusive_scan(std::execution::unseq, std::make_reverse_iterator(inPtr + input_.stride(0)),
                    std::make_reverse_iterator(inPtr), std::make_reverse_iterator(outPtr + output_.stride(0)),
                    [&](scalar_t a, scalar_t b) -> scalar_t
                    { return std::fma(a, static_cast<scalar_t>(1 / gamma), b); });
            }
        });

    restoreOutputShape(output_, input.sizes(), dim);
    return output_;
}

TORCH_LIBRARY_IMPL(discounted_cumsum, CPU, m)
{
    m.impl("discounted_cumsum", &forward_cpu);
    m.impl("_discounted_cumsum_bw", &backward_cpu);
}