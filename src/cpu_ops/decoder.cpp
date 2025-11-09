#include <cpu_ops/decoder.h>
#include <cstddef>

Decoder::Decoder(
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
    ) : layer_idx(_layer_idx)
    {
        self_attn = new SelfAttention(_q_proj_wt, _k_proj_wt, _v_proj_wt, _o_proj_wt, _q_norm_wt, _k_norm_wt, sin_cache, cos_cache, _layer_idx, _kvcache);
        input_norm_wt = std::move(_input_norm_wt);
        post_attn_norm_wt = std::move(_post_attn_norm_wt);
        mlp_up_proj_wt = std::move(_mlp_up_proj_wt);
        mlp_gate_proj_wt = std::move(_mlp_gate_proj_wt);
        mlp_down_proj_wt = std::move(_mlp_down_proj_wt);
    };

Decoder::~Decoder(){
    delete self_attn;

    // TODO : Free intermediate tensors and weights
}

void Decoder::prepare(){
    input_norm_wt.prefetch_async();
    
    self_attn->prepare();
    
    post_attn_norm_wt.prefetch_async();

    mlp_gate_proj_wt.prefetch_async();
    mlp_up_proj_wt.prefetch_async();
    mlp_down_proj_wt.prefetch_async();
}

void Decoder::run(Tensor &input, size_t token_idx, Tensor &output){

    // temp tensor for intermediate computation
    Tensor intermediate1(DataType::F32, {input.shape()[0]});
    Tensor intermediate2(DataType::F32, {input.shape()[0]});

    // pre attention norm
    rmsnorm_avx2(input.data<float>(), input_norm_wt.data<float>(), intermediate1.data<float>(), 1, input.shape()[0], 0.000001);

    // self attention
    self_attn->run(intermediate1, token_idx, intermediate2);

    // skip connection self attention
    elemwise_add_avx2_omp(input.data<float>(), intermediate2.data<float>(), intermediate1.data<float>(), 1, input.shape()[0]);

    // post attention norm
    rmsnorm_avx2(intermediate1.data<float>(), post_attn_norm_wt.data<float>(), intermediate2.data<float>(), 1, input.shape()[0], 0.000001);

    // mlp
    size_t up_dim = mlp_up_proj_wt.shape()[0];
    size_t down_dim = mlp_down_proj_wt.shape()[0];

    Tensor intermediate3(DataType::F32, {up_dim});
    Tensor intermediate4(DataType::F32, {up_dim});
    
    linear_avx2_omp(intermediate2.data<float>(), mlp_gate_proj_wt.data<float>(), 1, down_dim, up_dim, intermediate3.data<float>());
    silu_avx2(intermediate3.data<float>(), intermediate3.data<float>(), up_dim);
    linear_avx2_omp(intermediate2.data<float>(), mlp_up_proj_wt.data<float>(), 1, down_dim, up_dim, intermediate4.data<float>());
    elemwise_mul_avx2(intermediate3.data<float>(), intermediate4.data<float>(), intermediate3.data<float>(), 1, up_dim);
    linear_avx2_omp(intermediate3.data<float>(), mlp_down_proj_wt.data<float>(), 1, up_dim, down_dim, intermediate2.data<float>());

    // skip connection mlp
    elemwise_add_avx2_omp(intermediate1.data<float>(), intermediate2.data<float>(), output.data<float>(), 1, input.shape()[0]);    
}