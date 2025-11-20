#include "ops/softmax.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <immintrin.h>
#include <chrono>
#include "../test_utils.cpp"

int main() {
    constexpr size_t N = 151936;
    alignas(32) std::vector<float> x(N), out(N), ref(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10, 10);
    for (size_t i = 0; i < N; ++i) x[i] = dist(rng);

    // Reference timing
    auto start = std::chrono::high_resolution_clock::now();
    softmax_naive(x.data(), ref.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    auto ref_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // AVX2 timing
    start = std::chrono::high_resolution_clock::now();
    softmax_avx2(x.data(), out.data(), N);
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
    std::cout << "Naive Softmax Latency: " << ref_time << " us\n";
    std::cout << "AVX2 Softmax Latency: " << avx_time << " us\n";
    if (avx_time == 0) {
        std::cout << "Speedup: N/A (AVX2 time too small to measure)\n";
    } else {
        std::cout << "Speedup: " << (float)ref_time / (float)avx_time << "x\n";
    }
    if (pass) {
        std::cout << "Softmax AVX2 test passed!\n";
        return 0;
    } else {
        std::cerr << "Softmax AVX2 test failed!\n";
        return 1;
    }
}
