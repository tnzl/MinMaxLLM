#include <iostream>
#include <chrono>
#include <random>
#include <malloc.h>
#include <cpu_ops/linear.h>
#include <tensor/tensor.h>
#include "../test_utils.cpp"

int main()
{
    // Test dimensions - typical LLM sizes
    const int M = 16;
    const int K = 2048;
    const int N = 2048;

    // Allocate aligned memory
    float *A = static_cast<float *>(_aligned_malloc(M * K * sizeof(float), 32));
    float *B = static_cast<float *>(_aligned_malloc(N * K * sizeof(float), 32));
    float *C_opt = static_cast<float *>(_aligned_malloc(M * N * sizeof(float), 32));
    float *C_naive = static_cast<float *>(_aligned_malloc(M * N * sizeof(float), 32));
    float *C_linear_owned = static_cast<float *>(_aligned_malloc(M * N * sizeof(float), 32));
    float *C_linear_runtime = static_cast<float *>(_aligned_malloc(M * N * sizeof(float), 32));

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i)
        A[i] = dist(gen);
    for (int i = 0; i < K * N; ++i)
        B[i] = dist(gen);

    // Test 1: Correctness check
    std::cout << "Running correctness test...\n";
    linear_naive(A, B, M, K, N, C_naive);
    int iterations = 100;
    long long naive_total = 0;
    long long opt_total = 0;
    {
        for (int i = 0; i < iterations; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            linear_naive(A, B, M, K, N, C_naive);
            auto end = std::chrono::high_resolution_clock::now();
            naive_total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        std::cout << "\nNaive Linear Latency " << naive_total / iterations << " us.\n";
    }
    {
        for (int i = 0; i < iterations; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            linear_avx2_omp(A, B, M, K, N, C_opt);
            auto end = std::chrono::high_resolution_clock::now();
            opt_total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        std::cout << "\nAVX Linear Latency " << opt_total / iterations << " us.\n";
    }

    {
        Tensor input_tensor(static_cast<void *>(A), {static_cast<size_t>(M), static_cast<size_t>(K)}, DataType::F32, false, false);
        Tensor weight_tensor(static_cast<void *>(B), {static_cast<size_t>(N), static_cast<size_t>(K)}, DataType::F32, false, false);
        Tensor output_tensor(static_cast<void *>(C_linear_owned), {static_cast<size_t>(M), static_cast<size_t>(N)}, DataType::F32, false, false);

        LinearOp linear_with_owned_weight(std::move(weight_tensor));
        linear_with_owned_weight.prepare();

        auto start = std::chrono::high_resolution_clock::now();
        linear_with_owned_weight.run(input_tensor, output_tensor);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "\nLinearOp (stored weight) Latency "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " us.\n";

        Tensor weight_runtime(static_cast<void *>(B), {static_cast<size_t>(N), static_cast<size_t>(K)}, DataType::F32, false, false);
        Tensor output_runtime(static_cast<void *>(C_linear_runtime), {static_cast<size_t>(M), static_cast<size_t>(N)}, DataType::F32, false, false);
        LinearOp linear_runtime{MatmulImplType::AVX2};

        start = std::chrono::high_resolution_clock::now();
        linear_runtime.run(input_tensor, weight_runtime, output_runtime);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "LinearOp (runtime weight) Latency "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " us.\n";
    }

    printErrorAnalysis(C_naive, C_opt, M, N);
    std::cout << "Speedup: " << static_cast<float>(naive_total) / static_cast<float>(opt_total) << "x\n";

    printErrorAnalysis(C_naive, C_linear_owned, M, N);
    printErrorAnalysis(C_naive, C_linear_runtime, M, N);

    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(C_opt);
    _aligned_free(C_naive);
    _aligned_free(C_linear_owned);
    _aligned_free(C_linear_runtime);
}