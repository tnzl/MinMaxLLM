#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cpu_ops/SkipSimplifiedLayerNormalization_AVX2.h>

// Naive reference for validation
void naive_skip_simplified_layernorm(const float* input, const float* skip, const float* gamma, float* output, size_t H, float epsilon) {
    std::vector<float> tmp(H);
    float sum_sq = 0.0f;
    for (size_t i = 0; i < H; ++i) {
        tmp[i] = input[i] + skip[i];
        sum_sq += tmp[i] * tmp[i];
    }
    float mean_sq = sum_sq / H;
    float denom = 1.0f / std::sqrt(mean_sq + epsilon);
    for (size_t i = 0; i < H; ++i) {
        output[i] = gamma[i] * tmp[i] * denom;
    }
}

int main() {
    constexpr size_t H = 1024;
    constexpr float epsilon = 1e-5f;
    std::vector<float> input(H), skip(H), gamma(H), out_ref(H), out_opt(H);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < H; ++i) {
        input[i] = dist(rng);
        skip[i] = dist(rng);
        gamma[i] = dist(rng);
    }
    naive_skip_simplified_layernorm(input.data(), skip.data(), gamma.data(), out_ref.data(), H, epsilon);
    SkipSimplifiedLayerNormalization_AVX2(input.data(), skip.data(), gamma.data(), out_opt.data(), H, epsilon);
    // Compare
    float max_diff = 0.0f;
    for (size_t i = 0; i < H; ++i) {
        float diff = std::abs(out_ref[i] - out_opt[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max difference: " << max_diff << std::endl;
    if (max_diff < 1e-5f) {
        std::cout << "Test PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Test FAILED!" << std::endl;
        return 1;
    }
}
