#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cpu_ops/SkipSimplifiedLayerNormalization_AVX2.h>
#include <chrono>

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
    constexpr size_t H = 2048;
    constexpr float epsilon = 1e-5f;
    std::vector<float> input(H), skip(H), gamma(H), out_ref(H), out_opt(H);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < H; ++i) {
        input[i] = dist(rng);
        skip[i] = dist(rng);
        gamma[i] = dist(rng);
    }
    // Reference timing
    auto start = std::chrono::high_resolution_clock::now();
    naive_skip_simplified_layernorm(input.data(), skip.data(), gamma.data(), out_ref.data(), H, epsilon);
    auto end = std::chrono::high_resolution_clock::now();
    auto ref_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // AVX2 timing
    std::vector<float> out_skip(H); // dummy, not used in test
    start = std::chrono::high_resolution_clock::now();
    SkipSimplifiedLayerNormalization_AVX2(input.data(), skip.data(), gamma.data(), out_opt.data(), out_skip.data(), H, epsilon);
    end = std::chrono::high_resolution_clock::now();
    auto avx_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // Compare
    float max_diff = 0.0f;
    float l2_error = 0.0f;
    float sum_ref = 0.0f, sum_opt = 0.0f;
    int significant_errors = 0;
    const float threshold = 1e-5f;
    for (size_t i = 0; i < H; ++i) {
        float diff = std::abs(out_ref[i] - out_opt[i]);
        if (diff > max_diff) max_diff = diff;
        l2_error += diff * diff;
        if (diff > threshold) significant_errors++;
        sum_ref += out_ref[i] * out_ref[i];
        sum_opt += out_opt[i] * out_opt[i];
    }
    l2_error = std::sqrt(l2_error / H);
    float norm_ref = std::sqrt(sum_ref);
    float norm_opt = std::sqrt(sum_opt);
    float relative_error = std::fabs(norm_ref - norm_opt) / (norm_ref + 1e-12f);
    std::cout << "\nError Analysis:\n";
    std::cout << "L2 Error: " << l2_error << "\n";
    std::cout << "Max Error: " << max_diff << "\n";
    std::cout << "Relative Error (L2 norm): " << relative_error * 100 << "%\n";
    std::cout << "Elements with error > " << threshold << ": " << significant_errors << " ("
              << (100.0f * significant_errors) / H << "%)\n";
    std::cout << "Naive LayerNorm Latency: " << ref_time << " us\n";
    std::cout << "AVX2 LayerNorm Latency: " << avx_time << " us\n";
    std::cout << "Speedup: ";
    if (avx_time > 0)
        std::cout << (float)ref_time / (float)avx_time << "x\n";
    else
        std::cout << "N/A (AVX2 time is zero)\n";
    if (max_diff < 1e-5f) {
        std::cout << "Test PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Test FAILED!" << std::endl;
        return 1;
    }
}
