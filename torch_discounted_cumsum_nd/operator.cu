#include <cub/block/block_scan.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

template <typename scalar_t>
using TensorAcc4R = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;

constexpr auto gBlockDim = 256;

template <typename T>
__global__ void forward_kernel(const TensorAcc4R<T> input, TensorAcc4R<T> output, double gamma)
{
    using BlockScan = cub::BlockScan<T, gBlockDim>;
    __shared__ typename BlockScan::TempStorage tempStorage;

    T data;
    BlockScan(tempStorage).InclusiveSum(data, data);
}

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
    return input.permute(dims).contiguous();
}

[[nodiscard]] auto restore_input_shape(const torch::Tensor& input, int64_t dim) -> torch::Tensor
{
    std::vector<int64_t> dims;
    dims.reserve(input.ndimension());
    std::iota(dims.begin(), dims.end() - 1, 0);
    dims.insert(dims.begin() + dim, input.ndimension() - 1);
    return input.permute(dims).contiguous();
}

auto forward(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    const auto channelDim = input.size(dim);
    const std::vector inputShape(input.sizes().begin(), input.sizes().end());

    torch::Tensor input_;
    if (dim != input.ndimension() - 1)
    {
        input_ = permute_target_last(input, dim);
    }
    else
    {
        input_ = input;
    }

    // Create initially as permuted input
    auto output = torch::zeros_like(input_);

    // Flatten to batch
    input_ = input_.flatten(0, -2);
    auto output_ = output.flatten(0, -2);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_forward_kernel",
        [&]()
        {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            const dim3 blocksGrid(input.size(0), input.size(1), 1);
            const dim3 threadsPerBlock(gBlockDim);

            auto input_acc = input_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto output_acc = output_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            forward_kernel<scalar_t><<<blocksGrid, threadsPerBlock, 0, stream>>>(input_acc, output_acc, gamma);
        });

    if (dim != input.ndimension() - 1)
    {
        output = restore_input_shape(output, dim);
    }

    return output;
}

template <typename T>
__global__ void backward_kernel(const TensorAcc4R<T> input, TensorAcc4R<T> output, double gamma)
{
}

auto backward(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocksGrid(input.size(0), input.size(1), 1);
    const dim3 threadsPerBlock(gBlockDim);

    auto output = torch::zeros_like(input);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_backward_kernel",
        [&]()
        {
            auto input_acc = input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto output_acc = output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            backward_kernel<scalar_t><<<blocksGrid, threadsPerBlock, 0, stream>>>(input_acc, output_acc, gamma);
        });

    return output;
}

TORCH_LIBRARY(discounted_cumsum, m)
{
    m.def("discounted_cumsum(Tensor input, int dim, float gamma) -> Tensor");
    m.def("_discounted_cumsum_bw(Tensor input, int dim, float gamma) -> Tensor");
}

TORCH_LIBRARY_IMPL(discounted_cumsum, CUDA, m)
{
    m.impl("discounted_cumsum", &forward);
    m.impl("_discounted_cumsum_bw", &backward);
}
