#include <cpu_ops/self_attention.h>

#include <iostream>
SelfAttention::SelfAttention(
    Tensor &_q_proj_wt,
    Tensor &_k_proj_wt,
    Tensor &_v_proj_wt,
    Tensor &_o_proj_wt,
    Tensor &_q_norm_wt,
    Tensor &_k_norm_wt,
    Tensor &sin_cache,
    Tensor &cos_cache,
    size_t _layer_idx,
    KVCache *_kvcache)
{
    q_proj_wt = std::move(_q_proj_wt);
    k_proj_wt = std::move(_k_proj_wt);
    v_proj_wt = std::move(_v_proj_wt);
    o_proj_wt = std::move(_o_proj_wt);
    q_norm_wt = std::move(_q_norm_wt);
    k_norm_wt = std::move(_k_norm_wt);

    embed_dim = k_proj_wt.shape()[1];
    head_dim = k_norm_wt.shape()[0];
    num_heads = q_proj_wt.shape()[0] / head_dim;
    num_groups = k_proj_wt.shape()[0] / head_dim;

    layer_idx = _layer_idx;
    kvcache = _kvcache;

    rope = new RotaryEmbeddingAVX2(sin_cache.data<float>(), cos_cache.data<float>(), sin_cache.shape()[0], head_dim);

    scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
}

SelfAttention::~SelfAttention()
{
    delete rope;
    
    // TODO : Free query, key, value buffers when last self attention is destroyed
}

void SelfAttention::prepare()
{
    if (query.size() < num_heads * head_dim)
        query.resize(num_heads * head_dim);
    if (key.size() < num_groups * head_dim)
        key.resize(num_groups * head_dim);
    if (value.size() < num_groups * head_dim)
        value.resize(num_groups * head_dim);

    q_proj_wt.prefetch_async();
    k_proj_wt.prefetch_async();
    v_proj_wt.prefetch_async();
    o_proj_wt.prefetch_async();
    q_norm_wt.prefetch_async();
    k_norm_wt.prefetch_async();
}

void SelfAttention::run(Tensor &input, size_t token_idx, Tensor &output)
{
    linear_avx2_omp(input.data<float>(), q_proj_wt.data<float>(), 1, embed_dim, num_heads * head_dim, query.data());
    linear_avx2_omp(input.data<float>(), k_proj_wt.data<float>(), 1, embed_dim, num_groups * head_dim, key.data());
    linear_avx2_omp(input.data<float>(), v_proj_wt.data<float>(), 1, embed_dim, num_groups * head_dim, value.data());

    rmsnorm_avx2(query.data(), q_norm_wt.data<float>(), query.data(), num_heads, head_dim, 0.000001);
    rmsnorm_avx2(key.data(), k_norm_wt.data<float>(), key.data(), num_groups, head_dim, 0.000001);

    rope->rotate(query.data(), num_heads, head_dim, token_idx);
    rope->rotate(key.data(), num_groups, head_dim, token_idx);

    kvcache->set_current_key(layer_idx, key.data());
    kvcache->set_current_value(layer_idx, value.data());

    optimized_gqa_forward(
        query.data(),
        kvcache->get_key_memory_ptr(layer_idx),   // Key memory: [G, N_max, h] layout
        kvcache->get_value_memory_ptr(layer_idx), // Value memory: [G, N_max, h] layout
        query.data(),
        num_heads,
        num_groups,
        head_dim,
        token_idx + 1,  // Current sequence length (including current token)
        kvcache->get_max_sequence_length(),
        scale);

    linear_avx2_omp(query.data(), o_proj_wt.data<float>(), 1, num_heads * head_dim, embed_dim, output.data<float>());
}