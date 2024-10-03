#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

#include <execution>

void forward_cuda_contig(const torch::Tensor& inputFlat, double gamma, torch::Tensor& output);

void forward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output);

void backward_cuda_contig(const torch::Tensor& inputFlat, double gamma, torch::Tensor& output);

void backward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output);

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

auto forward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor inputFlat = prepareInput(input, dim);
    auto output = torch::zeros_like(inputFlat);

    forward_cuda_contig(inputFlat, gamma, output);

    restoreOutputShape(output, input.sizes(), dim);
    return output;
}

auto backward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor inputFlat = prepareInput(input, dim);
    auto output = torch::zeros_like(inputFlat);

    backward_cuda_contig(inputFlat, gamma, output);

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
