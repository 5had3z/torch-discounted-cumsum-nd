#include <torch/torch.h>

#include <cxxopts.hpp>

#include <iostream>

auto forward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor;

auto backward_cuda(const torch::Tensor& input, int64_t dim, double gamma) -> torch::Tensor;

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

    torch::cumsum(testData, dim);
    if (parsed_opts.count("backward"))
    {
        backward_cuda(testData, dim, gamma);
    }
    else
    {
        forward_cuda(testData, dim, gamma);
    }

    return 0;
}
