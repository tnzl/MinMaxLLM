#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "model_base.h"

/**
 * @brief Generic inference engine that manages model instances polymorphically.
 * 
 * The InferenceEngine acts as a factory and wrapper for model implementations.
 * It automatically selects the correct model type and configuration based on
 * the model name provided.
 */
class InferenceEngine
{
public:
    /**
     * @brief Construct an InferenceEngine for a specific model type.
     * @param model_name Name of the model (e.g., "Qwen3-1.7B")
     * @throws std::invalid_argument if model_name is not supported
     */
    explicit InferenceEngine(const std::string &model_name);

    /**
     * @brief Destructor.
     */
    ~InferenceEngine();

    // Non-copyable, movable
    InferenceEngine(const InferenceEngine &) = delete;
    InferenceEngine &operator=(const InferenceEngine &) = delete;
    InferenceEngine(InferenceEngine &&) noexcept = default;
    InferenceEngine &operator=(InferenceEngine &&) noexcept = default;

    /**
     * @brief Load model weights from a safetensors file.
     * @param safetensor_path Path to the safetensors file
     * @param use_mmap Whether to use memory-mapped I/O
     */
    void load_weights(const std::string &safetensor_path, bool use_mmap = false);

    /**
     * @brief Reset the KV cache and token counter.
     */
    void reset_cache();

    /**
     * @brief Process a single token during prompt processing.
     * @param token_id The token ID to process
     */
    void process_prompt_token(int token_id);

    /**
     * @brief Predict the next token given the current token.
     * @param token_id The current token ID
     * @return Reference to the logits vector (vocab_size elements)
     */
    const std::vector<float> &predict_next_token(int token_id);

    /**
     * @brief Get the number of tokens processed so far.
     * @return Number of tokens processed
     */
    std::size_t tokens_processed() const noexcept;

    /**
     * @brief Get the model name.
     * @return The model name
     */
    const std::string &model_name() const noexcept { return model_name_; }

private:
    /**
     * @brief Extract model family from model name (e.g., "Qwen3-1.7B" -> "qwen3").
     * @param model_name Full model name
     * @return Model family name in lowercase
     */
    static std::string extract_model_family(const std::string &model_name);

    /**
     * @brief Factory method to create a model instance based on model name.
     * @param model_name Name of the model
     * @return Unique pointer to the created model
     * @throws std::invalid_argument if model_name is not supported
     */
    std::unique_ptr<ModelBase> create_model(const std::string &model_name);

    std::string model_name_;
    std::unique_ptr<ModelBase> model_;
};

