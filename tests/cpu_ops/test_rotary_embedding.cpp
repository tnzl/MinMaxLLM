#include <cpu_ops/rotary_embedding.h>
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include "../test_utils.cpp"

// Generate random embeddings
void generate_random_embeddings(float* emb, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        emb[i] = dist(gen);
    }
}

// Reference implementation (scalar)
void rotary_ref(float* emb, int num_heads, int head_size, int pos,
                const float* sin_cache, const float* cos_cache, int rotary_dim) {
    const int rot_dim_half = rotary_dim / 2;
    for (int h = 0; h < num_heads; ++h) {
        float* head = emb + h * head_size;
        for (int i = 0; i < rot_dim_half; ++i) {
            float x1 = head[i];
            float x2 = head[i + rot_dim_half];
            float s = sin_cache[pos * rot_dim_half + i];
            float c = cos_cache[pos * rot_dim_half + i];
            
            head[i] = x1 * c - x2 * s;
            head[i + rot_dim_half] = x1 * s + x2 * c;
        }
    }
}

int main() {
    // Config
    constexpr int num_heads = 16;
    constexpr int head_size = 128;
    constexpr int max_pos = 4096;
    constexpr int rotary_dim = 64; // Test partial rotation
    
    // Initialize caches
    std::vector<float> sin_cache(max_pos * rotary_dim/2);
    std::vector<float> cos_cache(max_pos * rotary_dim/2);
    RotaryEmbeddingAVX2::precompute(sin_cache.data(), cos_cache.data(), 
                                   max_pos, rotary_dim);
    
    // Create test embeddings
    std::vector<float> embeddings(num_heads * head_size);
    generate_random_embeddings(embeddings.data(), embeddings.size());
    auto embeddings_ref = embeddings; // Copy for reference
    
    // Initialize rotary
    RotaryEmbeddingAVX2 rotary(sin_cache.data(), cos_cache.data(),
                              max_pos, head_size, rotary_dim);
    
    // --- Correctness Test ---
    rotary.rotate(embeddings.data(), num_heads, head_size, 42);
    rotary_ref(embeddings_ref.data(), num_heads, head_size, 42,
              sin_cache.data(), cos_cache.data(), rotary_dim);
    // Print error analysis
    printErrorAnalysis(embeddings.data(), embeddings_ref.data(), 1, embeddings.size());
    // --- Benchmark ---
    constexpr int warmup = 100;
    constexpr int trials = 10000;
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        rotary.rotate(embeddings.data(), num_heads, head_size, i % max_pos);
    }
    // AVX2 timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; ++i) {
        rotary.rotate(embeddings.data(), num_heads, head_size, i % max_pos);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto avx2_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    // Reference timing
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; ++i) {
        rotary_ref(embeddings_ref.data(), num_heads, head_size, i % max_pos,
                  sin_cache.data(), cos_cache.data(), rotary_dim);
    }
    end = std::chrono::high_resolution_clock::now();
    auto ref_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    // Results
    std::cout << "\nBenchmark Results:\n";
    std::cout << "AVX2: " << avx2_time / trials << " ns/token\n";
    std::cout << "Scalar: " << ref_time / trials << " ns/token\n";
    std::cout << "Speedup: " << (float)ref_time / avx2_time << "x\n";
    
    return 0;
}