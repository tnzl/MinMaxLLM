#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include <memory>
#include <vector>

class RotaryEmbeddingAVX2 {
private:
    struct alignas(32) RotationCache {
        std::vector<float> sin;
        std::vector<float> cos;
        int max_positions;
        int rotary_dim;
    };
    std::unique_ptr<RotationCache> cache_;
    int rotary_dim_;

public:
    // Constructor with precomputed caches
    RotaryEmbeddingAVX2(const float* sin_cache,
                       const float* cos_cache,
                       int max_positions,
                       int dim,
                       int rotary_dim = 0);

    // Disallow copying
    RotaryEmbeddingAVX2(const RotaryEmbeddingAVX2&) = delete;
    RotaryEmbeddingAVX2& operator=(const RotaryEmbeddingAVX2&) = delete;

    // Main rotation function
    void rotate(float* embeddings, 
                int num_heads,
                int head_size,
                int position_id) const;

    // Helper to precompute caches
    static void precompute(float* sin_cache,
                          float* cos_cache,
                          int max_positions,
                          int dim,
                          float base = 10000.0f);
};

#endif // ROTARY_EMBEDDING_H