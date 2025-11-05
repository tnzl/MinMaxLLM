#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cpu_ops/rmsnorm.h>
#include "../test_utils.cpp"
#include <chrono>

// Naive RMSNorm implementation for validation
std::vector<float> naive_rmsnorm_forward(const float *input, const float *weight, int batch_size, int hidden_size, float eps)
{
    std::vector<float> output(batch_size * hidden_size);
    for (int b = 0; b < batch_size; ++b)
    {
        const float *in = input + b * hidden_size;
        float mean_sq = 0.0f;
        for (int d = 0; d < hidden_size; ++d)
        {
            mean_sq += in[d] * in[d];
        }
        mean_sq /= hidden_size;
        float denom = 1.0f / std::sqrt(mean_sq + eps);
        for (int d = 0; d < hidden_size; ++d)
        {
            output[b * hidden_size + d] = weight[d] * in[d] * denom;
        }
    }
    return output;
}

int main()
{
    const int batch_size = 8;
    const int hidden_size = 32;
    float eps = 1e-6f;

    std::vector<float> input(batch_size * hidden_size);
    std::vector<float> weight(hidden_size, 1.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : input)
        x = dist(gen);
    for (auto &x : weight)
        x = dist(gen);

    // More accurate timing: nanoseconds and multiple iterations
    constexpr int num_iters = 10000;

    // Naive reference timing
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> ref;
    for (int i = 0; i < num_iters; ++i)
    {
        ref = naive_rmsnorm_forward(input.data(), weight.data(), batch_size, hidden_size, eps);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double naive_time_avg_ns = (double)naive_time_ns / num_iters;

    // ===== Test 1: Separate input/output =====
    std::cout << "=== Test 1: Separate input/output ===\n";
    std::vector<float> out(batch_size * hidden_size);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i)
    {
        rmsnorm_avx2(input.data(), weight.data(), out.data(), batch_size, hidden_size, eps);
    }
    end = std::chrono::high_resolution_clock::now();
    auto avx_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avx_time_avg_ns = (double)avx_time_ns / num_iters;

    // Validate
    bool pass = validateResults(ref.data(), out.data(), batch_size, hidden_size, 0.001);
    printErrorAnalysis(ref.data(), out.data(), batch_size, hidden_size);
    if (!pass)
    {
        std::cerr << "Error: RMSNorm results don't match (separate buffers)!\n";
        return 1;
    }
    std::cout << "RMSNorm correctness test passed (separate buffers)!\n";
    std::cout << "Naive RMSNorm Avg Latency: " << naive_time_avg_ns << " ns\n";
    std::cout << "AVX RMSNorm Avg Latency: " << avx_time_avg_ns << " ns\n";
    std::cout << "Speedup: " << naive_time_avg_ns / avx_time_avg_ns << "x\n\n";

    // ===== Test 2: In-place operation (input == output) =====
    std::cout << "=== Test 2: In-place operation (input == output) ===\n";
    std::vector<float> inplace_data = input; // Copy for in-place test
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i)
    {
        // Reset data for each iteration to ensure consistent timing
        if (i > 0)
            inplace_data = input;
        rmsnorm_avx2(inplace_data.data(), weight.data(), inplace_data.data(), batch_size, hidden_size, eps);
    }
    end = std::chrono::high_resolution_clock::now();
    auto inplace_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double inplace_time_avg_ns = (double)inplace_time_ns / num_iters;

    // Validate in-place result
    pass = validateResults(ref.data(), inplace_data.data(), batch_size, hidden_size, 0.001);
    printErrorAnalysis(ref.data(), inplace_data.data(), batch_size, hidden_size);
    if (!pass)
    {
        std::cerr << "Error: RMSNorm results don't match (in-place)!\n";
        return 1;
    }
    std::cout << "RMSNorm correctness test passed (in-place)!\n";
    std::cout << "In-place RMSNorm Avg Latency: " << inplace_time_avg_ns << " ns\n";
    std::cout << "Speedup (naive vs in-place): " << naive_time_avg_ns / inplace_time_avg_ns << "x\n";
    std::cout << "In-place vs separate buffers: " << avx_time_avg_ns / inplace_time_avg_ns << "x\n";

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}