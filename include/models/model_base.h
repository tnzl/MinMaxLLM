#pragma once

#include <cstddef>
#include <string>
#include <vector>

/**
 * @brief Base class for all language model implementations.
 * 
 * This class defines the common interface that all model architectures
 * (Qwen, LLaMA, etc.) must implement. Different model sizes are handled
 * through configuration parameters rather than separate classes.
 */
class ModelBase
{
public:
    virtual ~ModelBase() = default;

    /**
     * @brief Load model weights from a safetensors file.
     * @param safetensor_path Path to the safetensors file
     * @param use_mmap Whether to use memory-mapped I/O
     */
    virtual void load_weights(const std::string &safetensor_path, bool use_mmap = false) = 0;

    /**
     * @brief Reset the KV cache and token counter.
     */
    virtual void reset_cache() = 0;

    /**
     * @brief Process a single token during prompt processing.
     * @param token_id The token ID to process
     */
    virtual void process_prompt_token(int token_id) = 0;

    /**
     * @brief Predict the next token given the current token.
     * @param token_id The current token ID
     * @return Reference to the logits vector (vocab_size elements)
     */
    virtual const std::vector<float> &predict_next_token(int token_id) = 0;

    /**
     * @brief Get the number of tokens processed so far.
     * @return Number of tokens processed
     */
    virtual std::size_t tokens_processed() const noexcept = 0;

protected:
    /**
     * @brief Ensure that model weights have been loaded.
     * @throws std::runtime_error if weights are not loaded
     */
    virtual void ensure_weights_loaded() const = 0;

    /**
     * @brief Ensure that the KV cache has been initialized.
     * @throws std::runtime_error if cache is not initialized
     */
    virtual void ensure_cache_initialized() = 0;

    /**
     * @brief Check if a token ID is valid for this model.
     * @param token_id The token ID to validate
     * @throws std::out_of_range if token ID is invalid
     */
    virtual void check_token_valid(int token_id) const = 0;

    /**
     * @brief Ensure that the current position is within capacity.
     * @throws std::runtime_error if position exceeds maximum
     */
    virtual void ensure_position_capacity() const = 0;
};

