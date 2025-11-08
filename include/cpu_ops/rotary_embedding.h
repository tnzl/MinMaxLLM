#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include <memory>
#include <vector>

/**
 * @class RotaryEmbeddingAVX2
 * @brief Efficient rotary positional embedding with AVX2 optimizations.
 *
 * Supports precomputed sine/cosine caches for fast rotary embedding application in transformer models.
 */
class RotaryEmbeddingAVX2 {
private:
    /**
     * @struct RotationCache
     * @brief Stores precomputed sine and cosine values for rotary embedding.
     */
    struct alignas(32) RotationCache {
        std::vector<float> sin; ///< Sine cache
        std::vector<float> cos; ///< Cosine cache
        int max_positions;      ///< Maximum sequence positions cached
        int rotary_dim;         ///< Rotary dimension
    };
    std::unique_ptr<RotationCache> cache_;
    int rotary_dim_;

public:
    /**
     * @brief Constructor with precomputed caches.
     * @param sin_cache Pointer to sine cache array
     * @param cos_cache Pointer to cosine cache array
     * @param max_positions Maximum positions cached
     * @param dim Embedding dimension
     * @param rotary_dim Rotary dimension (default: 0 = use dim)
     */
    RotaryEmbeddingAVX2(const float* sin_cache,
                       const float* cos_cache,
                       int max_positions,
                       int dim,
                       int rotary_dim = 0);

    // Disallow copying
    RotaryEmbeddingAVX2(const RotaryEmbeddingAVX2&) = delete;
    RotaryEmbeddingAVX2& operator=(const RotaryEmbeddingAVX2&) = delete;

    /**
     * @brief Applies rotary embedding to the input embeddings in-place.
     * @param embeddings Pointer to input/output embeddings [num_heads, head_size]
     * @param num_heads Number of attention heads
     * @param head_size Size of each head
     * @param position_id Position index to apply
     */
    void rotate(float* embeddings, 
                int num_heads,
                int head_size,
                int position_id) const;

    /**
     * @brief Precomputes sine and cosine caches for rotary embedding.
     * @param sin_cache Output array for sine values
     * @param cos_cache Output array for cosine values
     * @param max_positions Number of positions to cache
     * @param dim Embedding dimension
     * @param base Base for frequency calculation (default: 10000.0f)
     */
    static void precompute(float* sin_cache,
                          float* cos_cache,
                          int max_positions,
                          int dim,
                          float base = 1000000.0f);
};

#endif // ROTARY_EMBEDDING_H