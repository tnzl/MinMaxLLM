#include <models/qwen3model.h>

#include <cpu_ops/decoder.h>
#include <cpu_ops/linear.h>
#include <cpu_ops/rmsnorm.h>
#include <cpu_ops/rotary_embedding.h>
#include <cpu_ops/softmax_avx2.h>
#include <tensor/kvcache.h>
#include <tensor/safetensors.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
Tensor wrap_tensor(Safetensor &weights, const std::string &key, bool mark_mmapped)
{
    const auto *info = weights.getTensorInfo(key);
    if (!info)
    {
        throw std::runtime_error("Missing tensor in safetensor file: " + key);
    }

    Tensor tensor(weights.tensorDataPtr<float>(key), info->shape, DataType::F32);
    tensor.mark_mmapped(mark_mmapped);
    return tensor;
}
} // namespace

Qwen3Model::Qwen3Model(const Qwen3Config &config)
    : config_(config),
      head_dim_(0),
      tokens_processed_(0),
      hidden_state_(DataType::F32, {static_cast<std::size_t>(config.hidden_size)}),
      decoder_output_(DataType::F32, {static_cast<std::size_t>(config.hidden_size)}),
      norm_output_(DataType::F32, {static_cast<std::size_t>(config.hidden_size)}),
      logits_buffer_(static_cast<std::size_t>(config.vocab_size), 0.0f)
{
    if (config.num_attention_heads <= 0)
    {
        throw std::invalid_argument("num_attention_heads must be positive");
    }

    //TODO : is this check required 
    if (config.hidden_size % config.num_attention_heads != 0)
    {
        throw std::invalid_argument("hidden_size must be divisible by num_attention_heads");
    }
    head_dim_ = config.hidden_size / config.num_attention_heads;
    if (head_dim_ % 2 != 0)
    {
        throw std::invalid_argument("head_dim must be even for rotary embeddings");
    }
    if (config.num_key_value_heads <= 0)
    {
        throw std::invalid_argument("num_key_value_heads must be positive");
    }
    if (config.vocab_size <= 0)
    {
        throw std::invalid_argument("vocab_size must be positive");
    }
}

Qwen3Model::~Qwen3Model() = default;

void Qwen3Model::load_weights(const std::string &safetensor_path, bool use_mmap)
{
    weights_ = std::make_unique<Safetensor>(safetensor_path, use_mmap);

    embedding_weight_ = wrap_tensor(*weights_, "model.embed_tokens.weight", use_mmap);
    final_norm_weight_ = wrap_tensor(*weights_, "model.norm.weight", use_mmap);

    sin_cache_ = Tensor(DataType::F32,
                        {static_cast<std::size_t>(config_.max_position_embeddings),
                         static_cast<std::size_t>(head_dim_ / 2)});
    cos_cache_ = Tensor(DataType::F32,
                        {static_cast<std::size_t>(config_.max_position_embeddings),
                         static_cast<std::size_t>(head_dim_ / 2)});

    RotaryEmbeddingAVX2::precompute(
        sin_cache_.data<float>(),
        cos_cache_.data<float>(),
        config_.max_position_embeddings,
        head_dim_,
        config_.rope_theta);

    kv_cache_ = std::make_unique<KVCache>(
        static_cast<std::size_t>(config_.max_position_embeddings),
        static_cast<std::size_t>(head_dim_),
        static_cast<std::size_t>(config_.num_key_value_heads),
        static_cast<std::size_t>(config_.num_hidden_layers));

    decoders_.clear();
    decoders_.reserve(static_cast<std::size_t>(config_.num_hidden_layers));

    for (int layer = 0; layer < config_.num_hidden_layers; ++layer)
    {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";

        Tensor input_norm = wrap_tensor(*weights_, prefix + "input_layernorm.weight", use_mmap);
        Tensor post_attn_norm = wrap_tensor(*weights_, prefix + "post_attention_layernorm.weight", use_mmap);

        Tensor q_proj = wrap_tensor(*weights_, prefix + "self_attn.q_proj.weight", use_mmap);
        Tensor k_proj = wrap_tensor(*weights_, prefix + "self_attn.k_proj.weight", use_mmap);
        Tensor v_proj = wrap_tensor(*weights_, prefix + "self_attn.v_proj.weight", use_mmap);
        Tensor o_proj = wrap_tensor(*weights_, prefix + "self_attn.o_proj.weight", use_mmap);
        Tensor q_norm = wrap_tensor(*weights_, prefix + "self_attn.q_norm.weight", use_mmap);
        Tensor k_norm = wrap_tensor(*weights_, prefix + "self_attn.k_norm.weight", use_mmap);

        Tensor mlp_up = wrap_tensor(*weights_, prefix + "mlp.up_proj.weight", use_mmap);
        Tensor mlp_gate = wrap_tensor(*weights_, prefix + "mlp.gate_proj.weight", use_mmap);
        Tensor mlp_down = wrap_tensor(*weights_, prefix + "mlp.down_proj.weight", use_mmap);

        auto decoder = std::make_unique<Decoder>(
            input_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            sin_cache_,
            cos_cache_,
            static_cast<std::size_t>(layer),
            kv_cache_.get(),
            post_attn_norm,
            mlp_up,
            mlp_gate,
            mlp_down);

        decoder->prepare();
        decoders_.push_back(std::move(decoder));
    }

    kv_cache_->reset();
    tokens_processed_ = 0;
}

void Qwen3Model::reset_cache()
{
    ensure_weights_loaded();
    ensure_cache_initialized();
    kv_cache_->reset();
    tokens_processed_ = 0;
}

void Qwen3Model::process_prompt_token(int token_id)
{
    ensure_weights_loaded();
    ensure_cache_initialized();
    check_token_valid(token_id);
    ensure_position_capacity();

    embed_token(token_id);

    const std::size_t token_index = kv_cache_->get_current_token_idx();
    run_decoder_stack(token_index);

    kv_cache_->advance();
    ++tokens_processed_;
}

const std::vector<float> &Qwen3Model::predict_next_token(int token_id)
{
    ensure_weights_loaded();
    ensure_cache_initialized();
    check_token_valid(token_id);
    ensure_position_capacity();

    embed_token(token_id);

    const std::size_t token_index = kv_cache_->get_current_token_idx();
    run_decoder_stack(token_index);
    apply_final_norm();
    run_lm_head();

    kv_cache_->advance();
    ++tokens_processed_;

    return logits_buffer_;
}

void Qwen3Model::ensure_weights_loaded() const
{
    if (!weights_)
    {
        throw std::runtime_error("Model weights have not been loaded");
    }
    if (embedding_weight_.raw_data() == nullptr || final_norm_weight_.raw_data() == nullptr)
    {
        throw std::runtime_error("Model weights are not initialized");
    }
}

void Qwen3Model::ensure_cache_initialized()
{
    if (!kv_cache_)
    {
        throw std::runtime_error("KV cache has not been initialized");
    }
}

void Qwen3Model::check_token_valid(int token_id) const
{
    if (token_id < 0 || token_id >= config_.vocab_size)
    {
        throw std::out_of_range("Token id out of vocabulary range");
    }
}

void Qwen3Model::ensure_position_capacity() const
{
    if (!kv_cache_)
    {
        throw std::runtime_error("KV cache unavailable");
    }
    if (kv_cache_->get_current_token_idx() >= kv_cache_->get_max_sequence_length())
    {
        throw std::runtime_error("Exceeded maximum position embeddings");
    }
}

void Qwen3Model::embed_token(int token_id)
{
    const std::size_t hidden = static_cast<std::size_t>(config_.hidden_size);
    float *dst = hidden_state_.data<float>();
    const float *src = embedding_weight_.data<float>() + hidden * static_cast<std::size_t>(token_id);
    std::memcpy(dst, src, hidden * sizeof(float));
}

void Qwen3Model::run_decoder_stack(std::size_t token_index)
{
    Tensor *current_input = &hidden_state_;
    Tensor *current_output = &decoder_output_;

    for (auto &decoder : decoders_)
    {
        decoder->run(*current_input, token_index, *current_output);
        std::swap(current_input, current_output);
    }

    if (current_input == &decoder_output_)
    {
        std::swap(hidden_state_, decoder_output_);
    }
}

void Qwen3Model::apply_final_norm()
{
    rmsnorm_avx2(
        hidden_state_.data<float>(),
        final_norm_weight_.data<float>(),
        norm_output_.data<float>(),
        1,
        config_.hidden_size,
        config_.rms_norm_eps);
}

void Qwen3Model::run_lm_head()
{
    linear_avx2_omp(
        norm_output_.data<float>(),
        embedding_weight_.data<float>(),
        1,
        config_.hidden_size,
        config_.vocab_size,
        logits_buffer_.data());
    
        //TODO : give option to set temperature
    softmax_avx2(logits_buffer_.data(), config_.vocab_size);
}

