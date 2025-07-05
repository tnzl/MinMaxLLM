#pragma once
#include <vector>
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cpu_ops/exp_avx2.h>
#include <cpu_ops/softmax_avx2.h>

/**
 * @class GroupQueryAttention
 * @brief Implements Grouped Query Attention (GQA) mechanism with AVX2 optimizations.
 *
 * This class computes attention for transformer models using grouped query attention.
 * It supports AVX2-optimized softmax for efficient computation.
 */
class GroupQueryAttention {
private:
    int num_heads;          // Number of query heads
    int kv_num_heads;       // Number of key/value heads
    int head_dim;           // Dimension of each head
    float scale;            // Scaling factor (1/sqrt(head_dim))
    int seq_len;           // Sequence length

public:
    /**
     * @brief Constructor for GroupQueryAttention.
     * @param num_heads Number of query heads
     * @param kv_num_heads Number of key/value heads
     * @param head_dim Dimension of each head
     * @param scale Scaling factor (default: -1.0f, computed as 1/sqrt(head_dim) if negative)
     */
    GroupQueryAttention(int num_heads, int kv_num_heads, int head_dim, float scale = -1.0f);

    /**
     * @brief Compute attention for a single query position.
     *
     * @param query Pointer to query tensor [num_heads, head_dim]
     * @param key Pointer to key tensor [seq_len, kv_num_heads, head_dim]
     * @param value Pointer to value tensor [seq_len, kv_num_heads, head_dim]
     * @param seq_len Sequence length
     * @return std::vector<float> Output tensor [num_heads, head_dim]
     */
    std::vector<float> forward(const float* query, const float* key, const float* value, int seq_len);
};