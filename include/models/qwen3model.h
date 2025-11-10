#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

class Safetensor;
class KVCache;
class Decoder;

struct Qwen3Config
{
    int hidden_size = 2048;
    int intermediate_size = 6144;
    int max_position_embeddings = 40960;
    int max_window_layers = 28;
    int num_attention_heads = 16;
    int num_hidden_layers = 28;
    int num_key_value_heads = 8;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    int vocab_size = 151936;
    int bos_token_id = 151643;
    int eos_token_id = 151645;
};

enum class TokenPhase
{
    Prompt,
    Generation
};

class Qwen3Model
{
public:
    explicit Qwen3Model(const Qwen3Config &config = Qwen3Config());
    ~Qwen3Model();

    void load_weights(const std::string &safetensor_path, bool use_mmap = false);

    void reset_cache();

    void process_prompt_token(int token_id);
    const std::vector<float> &predict_next_token(int token_id);

    const Qwen3Config &config() const noexcept { return config_; }
    std::size_t tokens_processed() const noexcept { return tokens_processed_; }

private:
    void ensure_weights_loaded() const;
    void ensure_cache_initialized();
    void check_token_valid(int token_id) const;
    void ensure_position_capacity() const;

    void embed_token(int token_id);
    void run_decoder_stack(std::size_t token_index);
    void apply_final_norm();
    void run_lm_head();

    Qwen3Config config_;
    int head_dim_;
    std::size_t tokens_processed_;

    std::unique_ptr<Safetensor> weights_;
    std::unique_ptr<KVCache> kv_cache_;
    std::vector<std::unique_ptr<Decoder>> decoders_;

    Tensor embedding_weight_;
    Tensor final_norm_weight_;
    Tensor sin_cache_;
    Tensor cos_cache_;

    Tensor hidden_state_;
    Tensor decoder_output_;
    Tensor norm_output_;

    std::vector<float> logits_buffer_;
};

