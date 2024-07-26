#include <cub/block/block_scan.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

#include <ranges>

template <typename scalar_t>
using TensorAcc2R = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;

constexpr auto gThreadBlockDim = 32;

template <typename T>
__global__ void forward_kernel(const TensorAcc2R<T> input, TensorAcc2R<T> output, float inv_gamma, int64_t scanDimSize)
{
    struct P
    {
        float i; // index
        float v; // value
    };
    using WarpScan = cub::WarpScan<P>;
    __shared__ typename WarpScan::TempStorage tempStorage;

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

    // Flatten to batch
    input_ = input_.flatten(0, -2);
    auto output_ = output.flatten(0, -2);
    if (!output_.is_contiguous())
    {
        throw std::runtime_error("expected output to be contiguous");
    }
    if (!input_.is_contiguous())
    {
        throw std::runtime_error("expected input to be contiguous");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocksGrid(static_cast<unsigned int>(input_.size(0)));
    const dim3 threadsPerBlock(gThreadBlockDim);
    std::cout << "launching with grid " << blocksGrid.x << std::endl;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "discounted_cumsum_forward_kernel",
        [&]()
        {
            auto input_acc = input_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto output_acc = output_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            forward_kernel<scalar_t>
                <<<blocksGrid, threadsPerBlock, 0, stream>>>(input_acc, output_acc, 1.f / gamma, scanDimSize);
        });
    cudaStreamSynchronize(stream);

    std::cout << "done " << cudaGetErrorName(cudaGetLastError()) << std::endl;

    if (!isLastDim)
    {
        output = restore_input_shape(output, dim);
    }

    return output;
}

template <typename T>
__global__ void backward_kernel(const TensorAcc2R<T> input, TensorAcc2R<T> output, double gamma)
{
}

auto backward(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocksGrid(input.size(0), input.size(1), 1);
    const dim3 threadsPerBlock(gThreadBlockDim);

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
