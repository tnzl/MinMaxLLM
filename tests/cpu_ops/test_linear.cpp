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
    float *C_linear_naive_owned = static_cast<float *>(_aligned_malloc(M * N * sizeof(float), 32));
    float *C_linear_naive_runtime = static_cast<float *>(_aligned_malloc(M * N * sizeof(float), 32));
    float *C_linear_avx_owned = static_cast<float *>(_aligned_malloc(M * N * sizeof(float), 32));
    float *C_linear_avx_runtime = static_cast<float *>(_aligned_malloc(M * N * sizeof(float), 32));

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i)
        A[i] = dist(gen);
    for (int i = 0; i < K * N; ++i)
        B[i] = dist(gen);

    std::cout << "Running LinearOp tests...\n";

    Tensor input_tensor(static_cast<void *>(A), {static_cast<size_t>(M), static_cast<size_t>(K)}, DataType::F32, false, false);

    // Owned-weight path
    {
        Tensor weight_owned_naive(static_cast<void *>(B), {static_cast<size_t>(N), static_cast<size_t>(K)}, DataType::F32, false, false);
        Tensor weight_owned_avx(static_cast<void *>(B), {static_cast<size_t>(N), static_cast<size_t>(K)}, DataType::F32, false, false);
        Tensor output_naive(static_cast<void *>(C_linear_naive_owned), {static_cast<size_t>(M), static_cast<size_t>(N)}, DataType::F32, false, false);
        Tensor output_avx(static_cast<void *>(C_linear_avx_owned), {static_cast<size_t>(M), static_cast<size_t>(N)}, DataType::F32, false, false);

        LinearOp linear_naive_owned(std::move(weight_owned_naive), MatmulImplType::NAIVE);
        LinearOp linear_avx_owned(std::move(weight_owned_avx), MatmulImplType::AVX2);

        linear_naive_owned.prepare();
        linear_avx_owned.prepare();

        auto start = std::chrono::high_resolution_clock::now();
        linear_naive_owned.run(input_tensor, output_naive);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "\nLinearOp (owned weight, NAIVE) Latency "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " us.\n";

        start = std::chrono::high_resolution_clock::now();
        linear_avx_owned.run(input_tensor, output_avx);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "LinearOp (owned weight, AVX2) Latency "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " us.\n";

        printErrorAnalysis(C_linear_naive_owned, C_linear_avx_owned, M, N, "LinearOp owned weight (AVX2 vs NAIVE)");
    }

    // Runtime-weight path
    {
        Tensor weight_runtime(static_cast<void *>(B), {static_cast<size_t>(N), static_cast<size_t>(K)}, DataType::F32, false, false);
        Tensor output_naive(static_cast<void *>(C_linear_naive_runtime), {static_cast<size_t>(M), static_cast<size_t>(N)}, DataType::F32, false, false);
        Tensor output_avx(static_cast<void *>(C_linear_avx_runtime), {static_cast<size_t>(M), static_cast<size_t>(N)}, DataType::F32, false, false);

        LinearOp linear_naive_runtime(MatmulImplType::NAIVE);
        LinearOp linear_avx_runtime(MatmulImplType::AVX2);

        auto start = std::chrono::high_resolution_clock::now();
        linear_naive_runtime.run(input_tensor, weight_runtime, output_naive);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "\nLinearOp (runtime weight, NAIVE) Latency "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " us.\n";

        start = std::chrono::high_resolution_clock::now();
        linear_avx_runtime.run(input_tensor, weight_runtime, output_avx);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "LinearOp (runtime weight, AVX2) Latency "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " us.\n";

        printErrorAnalysis(C_linear_naive_runtime, C_linear_avx_runtime, M, N, "LinearOp runtime weight (AVX2 vs NAIVE)");
    }

    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(C_linear_naive_owned);
    _aligned_free(C_linear_naive_runtime);
    _aligned_free(C_linear_avx_owned);
    _aligned_free(C_linear_avx_runtime);
}