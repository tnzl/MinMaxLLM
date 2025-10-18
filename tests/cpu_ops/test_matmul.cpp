#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <malloc.h>
#include <cpu_ops/matmul.h>
#include "../test_utils.cpp"

#ifdef _WIN32
#include <windows.h>
#endif

int main()
{
    // Test dimensions - typical LLM sizes
    const int M = 16;
    const int K = 2048;
    const int N = 2048;

    // Allocate aligned memory
    float *A = static_cast<float *>(_aligned_malloc(M * K * sizeof(float), 32));
    float *B = static_cast<float *>(_aligned_malloc(K * N * sizeof(float), 32));
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
    naiveMatMul(A, B, C_naive, M, N, K);
    int iterations = 100;
    long long naive_total = 0;
    long long opt_total = 0;
    {
        for (int i = 0; i < iterations; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            naiveMatMul(A, B, C_naive, M, N, K);
            auto end = std::chrono::high_resolution_clock::now();
            naive_total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        std::cout << "\nNaive MatMul Latency " << naive_total / iterations << " us.\n";
    }
    {
        for (int i = 0; i < iterations; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            hyperOptimizedMatMul(A, B, C_opt, M, N, K);
            auto end = std::chrono::high_resolution_clock::now();
            opt_total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        std::cout << "\nAVX MatMul Latency " << opt_total / iterations << " us.\n";
    }
    printErrorAnalysis(C_naive, C_opt, M, N);
    std::cout << "Speedup: " << (float)naive_total / (float)opt_total << "x\n";

    // Test 2: AVX2 MatMul with memory-mapped weights
    std::cout << "\nRunning AVX2 MatMul with memory-mapped weights...\n";
    const char *mmap_filename = "temp_weights.bin";
    // Write B to file
    FILE *fp = fopen(mmap_filename, "wb");
    if (!fp)
    {
        std::cerr << "Failed to create temp weights file!\n";
        return 1;
    }
    fwrite(B, sizeof(float), K * N, fp);
    fclose(fp);

    // Memory-map the file
    HANDLE hFile = CreateFileA(mmap_filename, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
    {
        std::cerr << "Failed to open temp weights file!\n";
        return 1;
    }
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap)
    {
        std::cerr << "Failed to create file mapping!\n";
        CloseHandle(hFile);
        return 1;
    }
    float *B_mmap = (float *)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, K * N * sizeof(float));
    if (!B_mmap)
    {
        std::cerr << "Failed to map view of file!\n";
        CloseHandle(hMap);
        CloseHandle(hFile);
        return 1;
    }

    float *C_mmap = static_cast<float *>(_aligned_malloc(M * N * sizeof(float), 32));
    long long mmap_total = 0;
    for (int i = 0; i < iterations; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hyperOptimizedMatMul(A, B_mmap, C_mmap, M, N, K);
        auto end = std::chrono::high_resolution_clock::now();
        mmap_total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "\nAVX MatMul (mmap weights) Latency " << mmap_total / iterations << " us.\n";
    printErrorAnalysis(C_naive, C_mmap, M, N);
    std::cout << "Speedup (mmap weights): " << (float)naive_total / (float)mmap_total << "x\n";

    // Cleanup mmap
    UnmapViewOfFile(B_mmap);
    CloseHandle(hMap);
    CloseHandle(hFile);
    free(C_mmap);
    remove(mmap_filename);

    // Cleanup
    free(A);
    free(B);
    free(C_opt);
    free(C_naive);
    return 0;
}