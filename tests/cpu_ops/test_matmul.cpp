#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <malloc.h>
#include <cpu_ops/matmul.h>
#include "../test_utils.cpp"

// Naive matrix multiplication for validation
void naiveMatMul(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // Test dimensions - typical LLM sizes
    const int M = 2048;
    const int K = 768;
    const int N = 3*768;    
    
    // Allocate aligned memory
    float* A = static_cast<float*>(_aligned_malloc(M * K * sizeof(float), 32));
    float* B = static_cast<float*>(_aligned_malloc(K * N * sizeof(float), 32));
    float* C_opt = static_cast<float*>(_aligned_malloc(M * N * sizeof(float), 32));
    float* C_naive = static_cast<float*>(_aligned_malloc(M * N * sizeof(float), 32));

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < M * K; ++i) A[i] = dist(gen);
    for (int i = 0; i < K * N; ++i) B[i] = dist(gen);

    // Test 1: Correctness check
    std::cout << "Running correctness test...\n";
    naiveMatMul(A, B, C_naive, M, N, K);
    // test for 100 iterations 
    int iterations = 100;
    {
        long long total = 0;
        for(int i=0; i<iterations; i++){
            auto start = std::chrono::high_resolution_clock::now();
            naiveMatMul(A, B, C_naive, M, N, K);
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        std::cout << "\nNaive MatMul Latency " << total/iterations << " us.\n";
    }
    {
        long long total = 0;
        for(int i=0; i<iterations; i++){
            auto start = std::chrono::high_resolution_clock::now();
            hyperOptimizedMatMul(A, B, C_opt, M, N, K);
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        std::cout << "\nAVX MatMul Latency " << total/iterations << " us.\n";
    }

    // print2DVector(C_opt, M, N); std::cout << std::endl;
    // print2DVector(C_naive, M, N); 
    
    if (!validateResults(C_naive, C_opt, M, N, 0.001)) {
        std::cerr << "Error: Results don't match!\n";
        printErrorAnalysis(C_naive, C_opt, M, N);
        return 1;
    }
    std::cout << "Correctness test passed!\n";
    
    // Print detailed error analysis even if test passes
    printErrorAnalysis(C_naive, C_opt, M, N);

    // Rest of the performance benchmark code remains the same...
    // ...
    
    // Cleanup
    free(A);
    free(B);
    free(C_opt);
    free(C_naive);
    
    return 0;
}