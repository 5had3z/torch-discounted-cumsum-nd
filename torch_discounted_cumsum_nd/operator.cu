#include <cub/block/block_scan.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

#include <ranges>

template <typename scalar_t>
using TensorAcc2R = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;

constexpr auto gThreadBlockDim = 32;

[[nodiscard]] auto permute_target_last(const torch::Tensor& input, int64_t dim) -> torch::Tensor
{
    std::vector<int64_t> dims;
    dims.reserve(input.ndimension());
    for (int64_t i = 0; i < input.ndimension(); ++i)
    {
        if (i != dim)
        {
            dims.emplace_back(i);
        }
    }
    dims.emplace_back(dim);
    return input.permute(dims);
}

[[nodiscard]] auto restore_input_shape(const torch::Tensor& input, int64_t dim) -> torch::Tensor
{
    // Allocate target size then remove last element by resizing
    std::vector<int64_t> dims(input.ndimension());
    dims.resize(dims.size() - 1);
    std::iota(dims.begin(), dims.end(), 0);
    dims.insert(dims.begin() + dim, input.ndimension() - 1);
    return input.permute(dims);
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
    const bool isLastDim = dim == (input.ndimension() - 1);
    const auto scanDimSize = input.size(dim);
    const std::vector inputShape(input.sizes().begin(), input.sizes().end());

    torch::Tensor input_;
    if (!isLastDim)
    {
        input_ = permute_target_last(input, dim);
    }
    else
    {
        input_ = input;
    }

    // Create initially as permuted input
    auto output = torch::zeros_like(input_);

    // Flatten to batch and ensure contiguous
    input_ = input_.flatten(0, -2).contiguous();
    auto output_ = output.flatten(0, -2).contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocksGrid(static_cast<unsigned int>(input_.size(0)));
    const dim3 threadsPerBlock(gThreadBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_forward_kernel",
        [&]()
        {
            auto input_acc = input_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto output_acc = output_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            forward_kernel<scalar_t>
                <<<blocksGrid, threadsPerBlock, 0, stream>>>(input_acc, output_acc, 1.f / gamma, scanDimSize);
        });

    if (!isLastDim)
    {
        output = restore_input_shape(output, dim);
    }

    return output;
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
    const bool isLastDim = dim == (input.ndimension() - 1);
    const auto scanDimSize = input.size(dim);
    const std::vector inputShape(input.sizes().begin(), input.sizes().end());

    torch::Tensor input_;
    if (!isLastDim)
    {
        input_ = permute_target_last(input, dim);
    }
    else
    {
        input_ = input;
    }

    // Create initially as permuted input
    auto output = torch::zeros_like(input_);

    // Flatten to batch and ensure contiguous
    input_ = input_.flatten(0, -2).contiguous();
    auto output_ = output.flatten(0, -2).contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocksGrid(static_cast<unsigned int>(input_.size(0)));
    const dim3 threadsPerBlock(gThreadBlockDim);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_kernel",
        [&]()
        {
            auto input_acc = input_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto output_acc = output_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            backward_kernel<scalar_t>
                <<<blocksGrid, threadsPerBlock, 0, stream>>>(input_acc, output_acc, 1.f / gamma, scanDimSize);
        });

    if (!isLastDim)
    {
        output = restore_input_shape(output, dim);
    }

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
    const bool isLastDim = dim == (input.ndimension() - 1);
    const auto scanDimSize = input.size(dim);
    const std::vector inputShape(input.sizes().begin(), input.sizes().end());

    torch::Tensor input_;
    if (!isLastDim)
    {
        input_ = permute_target_last(input, dim);
    }
    else
    {
        input_ = input;
    }

    // Create initially as permuted input
    auto output = torch::zeros_like(input_);

    // Flatten to batch and ensure contiguous
    input_ = input_.flatten(0, -2).contiguous();
    auto output_ = output.flatten(0, -2).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_kernel",
        [&]()
        {
            for (auto bidx = 0; bidx < input_.size(0); ++bidx)
            {
                const auto inPtr = static_cast<scalar_t*>(input_.data_ptr()) + bidx * input_.stride(0);
                auto outPtr = static_cast<scalar_t*>(output_.data_ptr()) + bidx * output_.stride(0);
                std::inclusive_scan(inPtr, inPtr + input.stride(0), outPtr, [&](scalar_t a, scalar_t b) -> scalar_t
                    { return std::fma(a, static_cast<scalar_t>(1 / gamma), b); });
            }
        });

    if (!isLastDim)
    {
        output = restore_input_shape(output, dim);
    }

    return output;
}

auto backward_cpu(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    const bool isLastDim = dim == (input.ndimension() - 1);
    const auto scanDimSize = input.size(dim);
    const std::vector inputShape(input.sizes().begin(), input.sizes().end());

    torch::Tensor input_;
    if (!isLastDim)
    {
        input_ = permute_target_last(input, dim);
    }
    else
    {
        input_ = input;
    }

    // Create initially as permuted input
    auto output = torch::zeros_like(input_);

    // Flatten to batch and ensure contiguous
    input_ = input_.flatten(0, -2).contiguous();
    auto output_ = output.flatten(0, -2).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_kernel",
        [&]()
        {
            for (auto bidx = 0; bidx < input_.size(0); ++bidx)
            {
                const auto inPtr = static_cast<scalar_t*>(input_.data_ptr()) + bidx * input_.stride(0);
                auto outPtr = static_cast<scalar_t*>(output_.data_ptr()) + bidx * output_.stride(0);
                std::inclusive_scan(std::make_reverse_iterator(inPtr + input_.stride(0)),
                    std::make_reverse_iterator(inPtr), std::make_reverse_iterator(outPtr + output_.stride(0)),
                    [&](scalar_t a, scalar_t b) -> scalar_t
                    { return std::fma(a, static_cast<scalar_t>(1 / gamma), b); });
            }
        });

    if (!isLastDim)
    {
        output = restore_input_shape(output, dim);
    }

    return output;
}

TORCH_LIBRARY_IMPL(discounted_cumsum, CPU, m)
{
    m.impl("discounted_cumsum", &forward_cpu);
    m.impl("_discounted_cumsum_bw", &backward_cpu);
}