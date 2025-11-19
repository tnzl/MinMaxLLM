#include <cpu_ops/decoder.h>
#include <cstddef>
#include <cassert>

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
        mlp_hidden_dim = _mlp_up_proj_wt.shape()[0];
        mlp_model_dim = _mlp_down_proj_wt.shape()[0];

        mlp_up_proj = std::make_unique<LinearOp>(std::move(_mlp_up_proj_wt));
        mlp_gate_proj = std::make_unique<LinearOp>(std::move(_mlp_gate_proj_wt));
        mlp_down_proj = std::make_unique<LinearOp>(std::move(_mlp_down_proj_wt));
    };

Decoder::~Decoder(){
    //call self_attn destructor
    delete self_attn;

    // TODO : Free intermediate tensors and weights
}

void Decoder::prepare(){
    input_norm_wt.prefetch_async();
    
    self_attn->prepare();
    
    post_attn_norm_wt.prefetch_async();

    if (mlp_gate_proj)
    {
        mlp_gate_proj->prepare();
    }
    if (mlp_up_proj)
    {
        mlp_up_proj->prepare();
    }
    if (mlp_down_proj)
    {
        mlp_down_proj->prepare();
    }
}

void Decoder::run(Tensor &input, size_t token_idx, Tensor &output){

    size_t model_dim = input.shape()[0];
    assert(mlp_gate_proj && mlp_up_proj && mlp_down_proj && "Decoder::run MLP projections are not initialized.");
    assert(model_dim == mlp_model_dim && "Decoder::run input dimension mismatch with MLP down projection.");

    // temp tensor for intermediate computation
    //TODO : find solution for temp tensors 
    Tensor intermediate1(DataType::F32, {model_dim});
    Tensor intermediate2(DataType::F32, {model_dim});

    // pre attention norm
    rmsnorm_avx2(input.data<float>(), input_norm_wt.data<float>(), intermediate1.data<float>(), 1, model_dim, 0.000001);

    // self attention
    self_attn->run(intermediate1, token_idx, intermediate2);

    // skip connection self attention
    elemwise_add_avx2_omp(input.data<float>(), intermediate2.data<float>(), intermediate1.data<float>(), 1, model_dim);

    // post attention norm
    rmsnorm_avx2(intermediate1.data<float>(), post_attn_norm_wt.data<float>(), intermediate2.data<float>(), 1, model_dim, 0.000001);

    // mlp
    Tensor intermediate3(DataType::F32, {mlp_hidden_dim});
    Tensor intermediate4(DataType::F32, {mlp_hidden_dim});

    mlp_gate_proj->run(intermediate2, intermediate3);
    silu_avx2(intermediate3.data<float>(), intermediate3.data<float>(), mlp_hidden_dim);
    mlp_up_proj->run(intermediate2, intermediate4);
    elemwise_mul_avx2(intermediate3.data<float>(), intermediate4.data<float>(), intermediate3.data<float>(), 1, mlp_hidden_dim);
    mlp_down_proj->run(intermediate3, intermediate2);

    // skip connection mlp
    elemwise_add_avx2_omp(intermediate1.data<float>(), intermediate2.data<float>(), output.data<float>(), 1, model_dim);    
}