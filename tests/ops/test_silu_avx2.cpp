#include <iostream>
#include <random>
#include <chrono>
#include <malloc.h>
#include <ops/silu.h>
#include <tensor/tensor.h>
#include "../test_utils.cpp"

int main() {
    constexpr size_t N = 1024;
    constexpr int num_iters = 100;

    // Allocate aligned memory
    float *input_data = static_cast<float *>(_aligned_malloc(N * sizeof(float), 32));
    float *output_golden = static_cast<float *>(_aligned_malloc(N * sizeof(float), 32));
    float *output_naive_op = static_cast<float *>(_aligned_malloc(N * sizeof(float), 32));
    float *output_avx2_func = static_cast<float *>(_aligned_malloc(N * sizeof(float), 32));
    float *output_avx2_op = static_cast<float *>(_aligned_malloc(N * sizeof(float), 32));

    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10, 10);
    for (size_t i = 0; i < N; ++i) {
        input_data[i] = dist(rng);
    }

    std::cout << "=== SiLU Test Suite ===\n";
    std::cout << "Size: " << N << " elements\n";
    std::cout << "Iterations per test: " << num_iters << "\n\n";

    // ===== Test 1: Naive function (golden output and latency baseline) =====
    std::cout << "=== Test 1: Naive Function (Golden Output & Baseline) ===\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        silu_naive(input_data, output_golden, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_func_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double naive_func_time_avg_ns = static_cast<double>(naive_func_time_ns) / num_iters;
    std::cout << "Naive function avg latency: " << naive_func_time_avg_ns << " ns\n";
    std::cout << "Naive function total time: " << naive_func_time_ns / 1e6 << " ms\n\n";

    // ===== Test 2: Naive Op Class =====
    std::cout << "=== Test 2: Naive Op Class ===\n";
    Tensor input_tensor(static_cast<void *>(input_data), {N}, DataType::F32, false, false);
    Tensor output_tensor_naive_op(static_cast<void *>(output_naive_op), {N}, DataType::F32, false, false);

    SiluOp silu_naive_op(OpBackend::NAIVE);
    silu_naive_op.prepare();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        silu_naive_op.run(input_tensor, output_tensor_naive_op);
    }
    end = std::chrono::high_resolution_clock::now();
    auto naive_op_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double naive_op_time_avg_ns = static_cast<double>(naive_op_time_ns) / num_iters;
    std::cout << "Naive op class avg latency: " << naive_op_time_avg_ns << " ns\n";
    std::cout << "Naive op class total time: " << naive_op_time_ns / 1e6 << " ms\n";

    // Validate against golden
    bool pass = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(output_golden[i] - output_naive_op[i]) > 1e-4f) {
            pass = false;
            break;
        }
    }
    printErrorAnalysis1D(output_golden, output_naive_op, N, 1e-4f);
    if (!pass) {
        std::cerr << "ERROR: Naive op class results don't match golden output!\n";
        _aligned_free(input_data);
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
    for (int i = 0; i < num_iters; ++i) {
        silu_avx2(input_data, output_avx2_func, N);
    }
    end = std::chrono::high_resolution_clock::now();
    auto avx2_func_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avx2_func_time_avg_ns = static_cast<double>(avx2_func_time_ns) / num_iters;
    std::cout << "AVX2 function avg latency: " << avx2_func_time_avg_ns << " ns\n";
    std::cout << "AVX2 function total time: " << avx2_func_time_ns / 1e6 << " ms\n";

    // Validate against golden
    pass = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(output_golden[i] - output_avx2_func[i]) > 1e-4f) {
            pass = false;
            break;
        }
    }
    printErrorAnalysis1D(output_golden, output_avx2_func, N, 1e-4f);
    if (!pass) {
        std::cerr << "ERROR: AVX2 function results don't match golden output!\n";
        _aligned_free(input_data);
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
    Tensor output_tensor_avx2_op(static_cast<void *>(output_avx2_op), {N}, DataType::F32, false, false);

    SiluOp silu_avx2_op(OpBackend::AVX2);
    silu_avx2_op.prepare();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        silu_avx2_op.run(input_tensor, output_tensor_avx2_op);
    }
    end = std::chrono::high_resolution_clock::now();
    auto avx2_op_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avx2_op_time_avg_ns = static_cast<double>(avx2_op_time_ns) / num_iters;
    std::cout << "AVX2 op class avg latency: " << avx2_op_time_avg_ns << " ns\n";
    std::cout << "AVX2 op class total time: " << avx2_op_time_ns / 1e6 << " ms\n";

    // Validate against golden
    pass = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(output_golden[i] - output_avx2_op[i]) > 1e-4f) {
            pass = false;
            break;
        }
    }
    printErrorAnalysis1D(output_golden, output_avx2_op, N, 1e-4f);
    if (!pass) {
        std::cerr << "ERROR: AVX2 op class results don't match golden output!\n";
        _aligned_free(input_data);
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
    _aligned_free(output_golden);
    _aligned_free(output_naive_op);
    _aligned_free(output_avx2_func);
    _aligned_free(output_avx2_op);

    return 0;
}
