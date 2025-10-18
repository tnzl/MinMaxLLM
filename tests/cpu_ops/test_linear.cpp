#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <malloc.h>
#include <cpu_ops/linear.h>
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
    printErrorAnalysis(C_naive, C_opt, M, N);
    std::cout << "Speedup: " << (float)naive_total / (float)opt_total << "x\n";
}