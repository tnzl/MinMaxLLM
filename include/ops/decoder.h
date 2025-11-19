#pragma once

#include <cstddef>
#include <tensor/tensor.h>
#include <tensor/kvcache.h>
#include <ops/self_attention.h>
#include <ops/linear.h>
#include <ops/rmsnorm.h>
#include <ops/elemwise_add.h>
#include <ops/silu_avx2.h>
#include <ops/elemwise_mul.h>
#include <vector>

// TODO : Create a separate MLP kernel and class

class Decoder
{
private:
    // pre-Attention norm
    RMSNormOp input_norm_op;

    // Self-Attention 
    SelfAttention *self_attn;

    // post attention norm
    RMSNormOp post_attn_norm_op;

    size_t mlp_hidden_dim = 0;
    size_t mlp_model_dim = 0;

    // MLP linear ops
    LinearOp mlp_up_proj;
    LinearOp mlp_gate_proj;
    LinearOp mlp_down_proj;

    size_t layer_idx = 0;

public:
    Decoder(
        // pre-Attention norm weights
        Tensor &_input_norm_wt,

        // Attention weights
        Tensor &_q_proj_wt,
        Tensor &_k_proj_wt,
        Tensor &_v_proj_wt,
        Tensor &_o_proj_wt,
        Tensor &_q_norm_wt,
        Tensor &_k_norm_wt,
        Tensor &sin_cache,
        Tensor &cos_cache,
        size_t _layer_idx,
        KVCache *_kvcache,

        // post-Attention norm weights
        Tensor &_post_attn_norm_wt,

        // MLP weights
        Tensor &_mlp_up_proj_wt,
        Tensor &_mlp_gate_proj_wt,
        Tensor &_mlp_down_proj_wt
        );

    ~Decoder();

    // Prepare buffers and prefetch weights
    void prepare();

    // Run attention for a single token
    void run(Tensor &input, size_t token_idx, Tensor &output);
};
