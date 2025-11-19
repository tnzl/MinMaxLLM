#include <iostream>
#include <random>
#include <chrono>
#include <cpu_ops/rmsnorm.h>
#include <tensor/tensor.h>
#include "../test_utils.cpp"
#include <malloc.h>

int main()
{
    const int batch_size = 8;
    const int hidden_size = 32;
    float eps = 1e-6f;

    // Allocate aligned memory
    float *input_data = static_cast<float *>(_aligned_malloc(batch_size * hidden_size * sizeof(float), 32));
    float *weight_data = static_cast<float *>(_aligned_malloc(hidden_size * sizeof(float), 32));
    float *output_golden = static_cast<float *>(_aligned_malloc(batch_size * hidden_size * sizeof(float), 32));
    float *output_naive_op = static_cast<float *>(_aligned_malloc(batch_size * hidden_size * sizeof(float), 32));
    float *output_avx2_func = static_cast<float *>(_aligned_malloc(batch_size * hidden_size * sizeof(float), 32));
    float *output_avx2_op = static_cast<float *>(_aligned_malloc(batch_size * hidden_size * sizeof(float), 32));

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < batch_size * hidden_size; ++i)
        input_data[i] = dist(gen);
    for (int i = 0; i < hidden_size; ++i)
        weight_data[i] = dist(gen);

    constexpr int num_iters = 10000;

    std::cout << "=== RMSNorm Test Suite ===\n";
    std::cout << "Batch size: " << batch_size << ", Hidden size: " << hidden_size << ", Epsilon: " << eps << "\n";
    std::cout << "Iterations per test: " << num_iters << "\n\n";

    // ===== Test 1: Naive function (golden output and latency baseline) =====
    std::cout << "=== Test 1: Naive Function (Golden Output & Baseline) ===\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i)
    {
        rmsnorm_naive(input_data, weight_data, output_golden, batch_size, hidden_size, eps);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_func_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double naive_func_time_avg_ns = static_cast<double>(naive_func_time_ns) / num_iters;
    std::cout << "Naive function avg latency: " << naive_func_time_avg_ns << " ns\n";
    std::cout << "Naive function total time: " << naive_func_time_ns / 1e6 << " ms\n\n";

    // ===== Test 2: Naive Op Class =====
    std::cout << "=== Test 2: Naive Op Class ===\n";
    Tensor input_tensor(static_cast<void *>(input_data), {static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::F32, false, false);
    Tensor weight_tensor(static_cast<void *>(weight_data), {static_cast<size_t>(hidden_size)}, DataType::F32, false, false);
    Tensor output_tensor_naive_op(static_cast<void *>(output_naive_op), {static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::F32, false, false);

    RMSNormOp rmsnorm_naive_op(OpBackend::NAIVE, eps);
    rmsnorm_naive_op.prepare();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i)
    {
        rmsnorm_naive_op.run(input_tensor, weight_tensor, output_tensor_naive_op);
    }
    end = std::chrono::high_resolution_clock::now();
    auto naive_op_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double naive_op_time_avg_ns = static_cast<double>(naive_op_time_ns) / num_iters;
    std::cout << "Naive op class avg latency: " << naive_op_time_avg_ns << " ns\n";
    std::cout << "Naive op class total time: " << naive_op_time_ns / 1e6 << " ms\n";

    // Validate against golden
    bool pass = validateResults(output_golden, output_naive_op, batch_size, hidden_size, 0.001f);
    printErrorAnalysis(output_golden, output_naive_op, batch_size, hidden_size, "Naive Op vs Golden");
    if (!pass)
    {
        std::cerr << "ERROR: Naive op class results don't match golden output!\n";
        _aligned_free(input_data);
        _aligned_free(weight_data);
        _aligned_free(output_golden);
        _aligned_free(output_naive_op);
        _aligned_free(output_avx2_func);
        _aligned_free(output_avx2_op);
        return 1;
    }
    std::cout << "✓ Naive op class correctness test PASSED\n";
    std::cout << "Speedup (naive func vs naive op): " << naive_func_time_avg_ns / naive_op_time_avg_ns << "x\n\n";

    // ===== Test 3: AVX2 Function =====
    std::cout << "=== Test 3: AVX2 Function ===\n";
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i)
    {
        rmsnorm_avx2(input_data, weight_data, output_avx2_func, batch_size, hidden_size, eps);
    }
    end = std::chrono::high_resolution_clock::now();
    auto avx2_func_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avx2_func_time_avg_ns = static_cast<double>(avx2_func_time_ns) / num_iters;
    std::cout << "AVX2 function avg latency: " << avx2_func_time_avg_ns << " ns\n";
    std::cout << "AVX2 function total time: " << avx2_func_time_ns / 1e6 << " ms\n";

    // Validate against golden
    pass = validateResults(output_golden, output_avx2_func, batch_size, hidden_size, 0.001f);
    printErrorAnalysis(output_golden, output_avx2_func, batch_size, hidden_size, "AVX2 Function vs Golden");
    if (!pass)
    {
        std::cerr << "ERROR: AVX2 function results don't match golden output!\n";
        _aligned_free(input_data);
        _aligned_free(weight_data);
        _aligned_free(output_golden);
        _aligned_free(output_naive_op);
        _aligned_free(output_avx2_func);
        _aligned_free(output_avx2_op);
        return 1;
    }
    std::cout << "✓ AVX2 function correctness test PASSED\n";
    std::cout << "Speedup (naive func vs avx2 func): " << naive_func_time_avg_ns / avx2_func_time_avg_ns << "x\n\n";

    // ===== Test 4: AVX2 Op Class =====
    std::cout << "=== Test 4: AVX2 Op Class ===\n";
    Tensor output_tensor_avx2_op(static_cast<void *>(output_avx2_op), {static_cast<size_t>(batch_size), static_cast<size_t>(hidden_size)}, DataType::F32, false, false);

    RMSNormOp rmsnorm_avx2_op(OpBackend::AVX2, eps);
    rmsnorm_avx2_op.prepare();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i)
    {
        rmsnorm_avx2_op.run(input_tensor, weight_tensor, output_tensor_avx2_op);
    }
    end = std::chrono::high_resolution_clock::now();
    auto avx2_op_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avx2_op_time_avg_ns = static_cast<double>(avx2_op_time_ns) / num_iters;
    std::cout << "AVX2 op class avg latency: " << avx2_op_time_avg_ns << " ns\n";
    std::cout << "AVX2 op class total time: " << avx2_op_time_ns / 1e6 << " ms\n";

    // Validate against golden
    pass = validateResults(output_golden, output_avx2_op, batch_size, hidden_size, 0.001f);
    printErrorAnalysis(output_golden, output_avx2_op, batch_size, hidden_size, "AVX2 Op vs Golden");
    if (!pass)
    {
        std::cerr << "ERROR: AVX2 op class results don't match golden output!\n";
        _aligned_free(input_data);
        _aligned_free(weight_data);
        _aligned_free(output_golden);
        _aligned_free(output_naive_op);
        _aligned_free(output_avx2_func);
        _aligned_free(output_avx2_op);
        return 1;
    }
    std::cout << "✓ AVX2 op class correctness test PASSED\n";
    std::cout << "Speedup (naive func vs avx2 op): " << naive_func_time_avg_ns / avx2_op_time_avg_ns << "x\n";
    std::cout << "Speedup (avx2 func vs avx2 op): " << avx2_func_time_avg_ns / avx2_op_time_avg_ns << "x\n\n";

    // ===== Summary =====
    std::cout << "=== Summary ===\n";
    std::cout << "1. Naive function (golden):     " << naive_func_time_avg_ns << " ns\n";
    std::cout << "2. Naive op class:              " << naive_op_time_avg_ns << " ns (vs golden: " << naive_func_time_avg_ns / naive_op_time_avg_ns << "x)\n";
    std::cout << "3. AVX2 function:               " << avx2_func_time_avg_ns << " ns (vs golden: " << naive_func_time_avg_ns / avx2_func_time_avg_ns << "x)\n";
    std::cout << "4. AVX2 op class:               " << avx2_op_time_avg_ns << " ns (vs golden: " << naive_func_time_avg_ns / avx2_op_time_avg_ns << "x)\n";
    std::cout << "\n=== All tests passed! ===\n";

    // Cleanup
    _aligned_free(input_data);
    _aligned_free(weight_data);
    _aligned_free(output_golden);
    _aligned_free(output_naive_op);
    _aligned_free(output_avx2_func);
    _aligned_free(output_avx2_op);

    return 0;
}
