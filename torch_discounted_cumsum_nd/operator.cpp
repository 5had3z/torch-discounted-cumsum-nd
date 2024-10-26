#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

#include <execution>

void forward_cuda_contig(const torch::Tensor& inputFlat, double gamma, torch::Tensor& output);

void forward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output);

void backward_cuda_contig(const torch::Tensor& inputFlat, double gamma, torch::Tensor& output);

void backward_cuda_noncontig(const torch::Tensor& input, double gamma, torch::Tensor& output);

/**
 * @brief Get the permutation to make `dim` last for shape of `ndims`.
 *
 * @param ndim Number of dimensions.
 * @param dim Dimension index to permute to the end.
 * @return Permutation to make dim last.
 */
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

/**
 * @brief Get the permutation required to reverse making `dim` last for shape of `ndims`.
 *
 * @param ndim Number of dimensions.
 * @param dim Original dimension that was permuted to last.
 * @return Permutation to restore original shape.
 */
[[nodiscard]] auto getReversePermutation(int64_t ndim, int64_t dim) -> std::vector<int64_t>
{
    // Allocate target size then remove last element by resizing
    std::vector<int64_t> dims(ndim);
    dims.resize(dims.size() - 1);
    std::iota(dims.begin(), dims.end(), 0);
    dims.insert(dims.begin() + dim, ndim - 1);
    return dims;
}

/**
 * @brief Permutes dim to be the last and flattens other dimensions. Also enforces the output to be contiguous.
 *
 * @param input Tensor to permute
 * @param dim Dimension to permute to the end.
 * @return Contiguous two dimension tensor where dim is now last.
 */
[[nodiscard]] auto permuteDimToLast(const torch::Tensor& input, int64_t dim) -> torch::Tensor
{
    torch::Tensor input_ = input;
    if (dim != (input.ndimension() - 1))
    {
        input_ = input_.permute(getForwardPermutation(input.ndimension(), dim));
    }
    return input_.flatten(0, -2).contiguous();
}

/**
 * @brief Permute dim of inShape to be last.
 *
 * @param inShape Shape to permute
 * @param dim Dimension to permute last
 * @return Permuted shape.
 */
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

/**
 * @brief Restore output to the original inShape where dim of inShape was permuted last.
 *
 * @param output Tensor to restore (modified inplace).
 * @param inShape Original shape to restore.
 * @param dim Dimension that was permuted to last.
 */
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

/**
 * @brief Performs weighted inclusive-scan on input over dim with gamma.
 *
 * @param input data to scan
 * @param dim dimension to scan over
 * @param gamma decay rate of scan
 * @return Tensor weighted sum of input over dim
 */
auto forward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    auto output = torch::empty_like(input);

    if (dim == (input.ndimension() - 1))
    {
        torch::Tensor inputFlat = input.flatten(0, -2).contiguous();
        output = output.reshape_as(inputFlat);
        forward_cuda_contig(inputFlat, gamma, output);
    }
    else
    {
        torch::Tensor inputFlat = input.flatten(0, dim - 1).flatten(2, -1).contiguous();
        output = output.reshape_as(inputFlat);
        forward_cuda_noncontig(inputFlat, gamma, output);
    }

    output = output.reshape_as(input);
    return output;
}

/**
 * @brief Calculates gradient for inputs of weighted scan.
 *
 * @param input gradient of output
 * @param dim dimension to scan over
 * @param gamma decay rate of scan
 * @return Tensor gradient of input
 */
auto backward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor inputFlat = permuteDimToLast(input, dim);
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

/**
 * @brief Performs weighted inclusive-scan on input over dim with gamma.
 *
 * @param input data to scan
 * @param dim dimension to scan over
 * @param gamma decay rate of scan
 * @return Tensor result
 */
auto forward_cpu(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor inputFlat = permuteDimToLast(input, dim);
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

/**
 * @brief Calculates gradient for inputs of weighted scan.
 *
 * @param input gradient of output
 * @param dim dimension to scan over
 * @param gamma decay rate of scan
 * @return Tensor gradient of input
 */
auto backward_cpu(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor
{
    if (dim < 0) // wrap to [0,ndim]
    {
        dim += input.ndimension();
    }
    torch::Tensor inputFlat = permuteDimToLast(input, dim);
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
