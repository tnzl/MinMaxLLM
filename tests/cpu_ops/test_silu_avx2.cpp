#include "cpu_ops/silu_avx2.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <immintrin.h>
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

    silu_avx2(x.data(), out.data(), N);
    for (size_t i = 0; i < N; ++i) ref[i] = silu_ref(x[i]);

    // Only print summary error metrics
    printErrorAnalysis1D(ref.data(), out.data(), N);

    // Check pass/fail
    bool pass = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(out[i] - ref[i]) > 1e-4f) {
            pass = false;
            break;
        }
    }
    if (pass) {
        std::cout << "SiLU AVX2 test passed!\n";
        return 0;
    } else {
        std::cerr << "SiLU AVX2 test failed!\n";
        return 1;
    }
}
