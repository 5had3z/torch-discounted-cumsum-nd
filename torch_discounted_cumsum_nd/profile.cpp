#include <torch/torch.h>

#include <cxxopts.hpp>

#include <iostream>

auto forward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor;

auto backward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor;

[[nodiscard]] auto permuteDimToLast(const torch::Tensor& input, int64_t dim) -> torch::Tensor;

void restoreOutputShape(torch::Tensor& output, c10::IntArrayRef inShape, int64_t dim);

int main(int argc, char* argv[])
{
    cxxopts::Options options("Profile-Discounted-Cumsum",
        "Run discounted cumsum operation with args specifying the test data shape and parameters");

    // clang-format off
    options.add_options()
        ("shape", "Shape of tensor to profile", cxxopts::value<std::vector<int64_t>>())
        ("dim", "Dimension of tensor to profile", cxxopts::value<int64_t>())
        ("gamma", "Value of gamma", cxxopts::value<double>()->default_value("2.0"))
        ("backward", "Do backward (not forward)", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "This help");
    // clang-format on

    auto parsed_opts = options.parse(argc, argv);

    if (parsed_opts.count("help"))
    {
        std::cout << options.help();
        return 0;
    }

    auto gamma = parsed_opts["gamma"].as<double>();
    auto dim = parsed_opts["dim"].as<int64_t>();
    auto testShape = parsed_opts["shape"].as<std::vector<int64_t>>();
    if (std::any_of(testShape.begin(), testShape.end(), [](int64_t i) { return i <= 0; }))
    {
        std::cerr << "Input shape dimensions must be positive and non-zero, got: " << testShape << "\n";
        return -1;
    }
    auto opts = c10::TensorOptions().device(c10::Device("cuda:0")).dtype(c10::ScalarType::Float);
    auto testData = torch::randn(c10::IntArrayRef(testShape), opts);

    auto tposefw = permuteDimToLast(testData, dim);
    restoreOutputShape(tposefw, testData.sizes(), dim);

    torch::Tensor baseResult = torch::cumsum(testData, dim);
    torch::Tensor customResult;
    if (parsed_opts.count("backward"))
    {
        backward_cuda(testData, dim, gamma);
    }
    else
    {
        customResult = forward_cuda(testData, dim, gamma);
    }

    if (customResult.numel() > 0 && gamma == 1.0)
    {
        if (!torch::allclose(baseResult, customResult))
        {
            std::cerr << "Got unequal results beteen baseline and custom kernel\n";
        }
    }

    return 0;
}
