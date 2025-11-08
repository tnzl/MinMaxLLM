#pragma once

#include <tensor/tensor.h>
#include <tensor/kvcache.h>
#include <cpu_ops/matmul.h>
#include <cpu_ops/rotary_embedding.h>
#include <cpu_ops/gqa.h>
#include <cpu_ops/linear.h>
#include <cpu_ops/rmsnorm.h>
#include <vector>

/*
SelfAttention class for Qwen3-style attention that processes one token at a time.

Inputs :
    input : [embed_dim]
    q_proj_weight: [num_heads*head_dim, embed_dim]
    q_norm: [head_dim]
    k_proj_weight: [num_groups*head_dim, embed_dim]
    k_norm: [head_dim]
    v_proj_weight: [num_groups*head_dim, embed_dim]
    o_proj_weight: [embed_dim, num_groups*head_dim]
    kv_cache
    embed_dim: int
    num_heads: int
    num_groups: int
    head_dim: int
    token_idx: int
    output: [embed_dim]
*/

class SelfAttention
{
private:
    Tensor q_proj_wt;
    Tensor k_proj_wt;
    Tensor v_proj_wt;
    Tensor o_proj_wt;
    Tensor q_norm_wt;
    Tensor k_norm_wt;

    // Instance-level buffers to avoid thread safety issues with static members
    std::vector<float> query; // Intermediate buffer for query projections
    std::vector<float> key;    // Intermediate buffer for key projections
    std::vector<float> value;  // Intermediate buffer for value projections
    
    size_t embed_dim = 0;
    size_t num_heads = 0;
    size_t num_groups = 0;
    size_t head_dim = 0;
    size_t layer_idx = 0;
    float scale;

    KVCache *kvcache = nullptr;
    RotaryEmbeddingAVX2 *rope;

public:
    SelfAttention(
        Tensor &_q_proj_wt,
        Tensor &_k_proj_wt,
        Tensor &_v_proj_wt,
        Tensor &_o_proj_wt,
        Tensor &_q_norm_wt,
        Tensor &_k_norm_wt,
        Tensor &sin_cache,
        Tensor &cos_cache,
        size_t _layer_idx,
        KVCache *_kvcache);

    ~SelfAttention();

    // Prepare buffers and prefetch weights
    void prepare();

    // Run attention for a single token
    void run(Tensor &input, size_t token_idx, Tensor &output);
};
