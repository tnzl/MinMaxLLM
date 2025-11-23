#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../tensor/tensor.h"
#include "../ops/rmsnorm.h"
#include "model_base.h"

class Safetensor;
class KVCache;
class Decoder;

/**
 * @brief Configuration class for Qwen3 model architecture.
 * 
 * Each model architecture should have its own config class.
 * Different model sizes (e.g., Qwen3-1.5B, Qwen3-7B) use
 * different instances of this config with different values.
 */
class Qwen3Config
{
public:
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

class Qwen3Model : public ModelBase
{
public:
    explicit Qwen3Model(const Qwen3Config &config = Qwen3Config());
    ~Qwen3Model() override;

    // ModelBase interface
    void load_weights(const std::string &safetensor_path, bool use_mmap = false) override;

    void reset_cache() override;

    void process_prompt_token(int token_id) override;
    const std::vector<float> &predict_next_token(int token_id) override;

    std::size_t tokens_processed() const noexcept override { return tokens_processed_; }

    // Qwen3Model-specific methods
    const Qwen3Config &config() const noexcept { return config_; }

protected:
    // ModelBase protected interface
    void ensure_weights_loaded() const override;
    void ensure_cache_initialized() override;
    void check_token_valid(int token_id) const override;
    void ensure_position_capacity() const override;

private:

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
    RMSNormOp final_norm_op_;
    Tensor sin_cache_;
    Tensor cos_cache_;

    Tensor hidden_state_;
    Tensor decoder_output_;
    Tensor norm_output_;

    std::vector<float> logits_buffer_;
};

