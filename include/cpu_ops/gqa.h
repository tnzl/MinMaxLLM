#pragma once
#include <vector>
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cpu_ops/exp_avx2.h>

class GroupQueryAttention {
private:
    int num_heads;          // Number of query heads
    int kv_num_heads;       // Number of key/value heads
    int head_dim;           // Dimension of each head
    float scale;            // Scaling factor (1/sqrt(head_dim))
    int seq_len;           // Sequence length

    // Helper function for AVX2 softmax
    void softmax(float* arr, int size);

public:
    GroupQueryAttention(int num_heads, int kv_num_heads, int head_dim, float scale = -1.0f);

    // Compute attention for a single query position
    // query: [num_heads, head_dim]
    // key: [seq_len, kv_num_heads, head_dim]
    // value: [seq_len, kv_num_heads, head_dim]
    // Output: [num_heads, head_dim]
    std::vector<float> forward(const float* query, const float* key, const float* value, int seq_len);
};