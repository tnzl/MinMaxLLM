#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <malloc.h>
#include <cpu_ops/SimplifiedLayerNormalization_AVX2.h>
#include "../test_utils.cpp"

// Reference implementation for validation
void reference_simplified_layernorm(const float* input, const float* scale, float* output, float epsilon, int num_heads, int head_dim) {
    for (int h = 0; h < num_heads; ++h) {
        const float* x = input + h * head_dim;
        float* y = output + h * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) sum += x[d];
        float mean = sum / head_dim;
        float var = 0.0f;
        for (int d = 0; d < head_dim; ++d) var += (x[d] - mean) * (x[d] - mean);
        var /= head_dim;
        float inv_std = 1.0f / std::sqrt(var + epsilon);
        for (int d = 0; d < head_dim; ++d) {
            y[d] = (x[d] - mean) * inv_std * scale[d];
        }
    }
}

int main() {
    // Test dimensions - typical LLM sizes
    const int num_heads = 16;
    const int head_dim = 2048;
    const float epsilon = 1e-5f;
    float* input = static_cast<float*>(_aligned_malloc(num_heads * head_dim * sizeof(float), 32));
    float* scale = static_cast<float*>(_aligned_malloc(head_dim * sizeof(float), 32));
    float* output = static_cast<float*>(_aligned_malloc(num_heads * head_dim * sizeof(float), 32));
    float* ref = static_cast<float*>(_aligned_malloc(num_heads * head_dim * sizeof(float), 32));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < num_heads * head_dim; ++i) input[i] = dist(gen);
    for (int i = 0; i < head_dim; ++i) scale[i] = dist(gen);

    // Correctness check
    std::cout << "Running correctness test...\n";
    reference_simplified_layernorm(input, scale, ref, epsilon, num_heads, head_dim);
    cpu_ops::SimplifiedLayerNormalization_AVX2(input, scale, output, epsilon, num_heads, head_dim);
    printErrorAnalysis(ref, output, num_heads, head_dim);

    // Performance test
    int iterations = 100;
    long long ref_total = 0;
    long long opt_total = 0;
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        reference_simplified_layernorm(input, scale, ref, epsilon, num_heads, head_dim);
        auto end = std::chrono::high_resolution_clock::now();
        ref_total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "\nNaive LayerNorm Latency: " << ref_total / iterations << " us\n";
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cpu_ops::SimplifiedLayerNormalization_AVX2(input, scale, output, epsilon, num_heads, head_dim);
        auto end = std::chrono::high_resolution_clock::now();
        opt_total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "AVX2 LayerNorm Latency: " << opt_total / iterations << " us\n";
    std::cout << "Speedup: ";
    if (opt_total > 0)
        std::cout << (float)ref_total / (float)opt_total << "x\n";
    else
        std::cout << "N/A (AVX2 time is zero)\n";
    _aligned_free(input);
    _aligned_free(scale);
    _aligned_free(output);
    _aligned_free(ref);
    return 0;
}
