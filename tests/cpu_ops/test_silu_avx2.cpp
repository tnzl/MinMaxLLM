#include "cpu_ops/silu_avx2.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <immintrin.h>
#include <chrono>
#include "../test_utils.cpp"

// Reference SiLU implementation
static float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

int main() {
    constexpr size_t N = 1024;
    alignas(32) std::vector<float> x(N), out(N), ref(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10, 10);
    for (size_t i = 0; i < N; ++i) x[i] = dist(rng);

    // Reference timing
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) ref[i] = silu_ref(x[i]);
    auto end = std::chrono::high_resolution_clock::now();
    auto ref_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // AVX2 timing
    start = std::chrono::high_resolution_clock::now();
    silu_avx2(x.data(), out.data(), N);
    end = std::chrono::high_resolution_clock::now();
    auto avx_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Validate
    bool pass = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(out[i] - ref[i]) > 1e-4f) {
            pass = false;
            break;
        }
    }
    // Always print error analysis
    printErrorAnalysis1D(ref.data(), out.data(), N);
    std::cout << "Naive SiLU Latency: " << ref_time << " us\n";
    std::cout << "AVX2 SiLU Latency: " << avx_time << " us\n";
    std::cout << "Speedup: " << (float)ref_time / (float)avx_time << "x\n";
    if (pass) {
        std::cout << "SiLU AVX2 test passed!\n";
        return 0;
    } else {
        std::cerr << "SiLU AVX2 test failed!\n";
        return 1;
    }
}
