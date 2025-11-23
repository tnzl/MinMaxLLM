#include <models/inference_engine.h>
#include <models/qwen3model.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

InferenceEngine::InferenceEngine(const std::string &model_name)
    : model_name_(model_name), model_(create_model(model_name))
{
}

InferenceEngine::~InferenceEngine() = default;

std::string InferenceEngine::extract_model_family(const std::string &model_name)
{
    // Extract model family from model name
    // Examples: "Qwen3-1.7B" -> "qwen3", "qwen3-1.7b" -> "qwen3", "Qwen3" -> "qwen3"
    
    std::string family = model_name;
    
    // Find the first dash or space (if any) to separate family from size
    size_t dash_pos = family.find('-');
    size_t space_pos = family.find(' ');
    size_t separator_pos = (dash_pos != std::string::npos) ? dash_pos : space_pos;
    
    if (separator_pos != std::string::npos)
    {
        family = family.substr(0, separator_pos);
    }
    
    // Convert to lowercase
    std::transform(family.begin(), family.end(), family.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    return family;
}

std::unique_ptr<ModelBase> InferenceEngine::create_model(const std::string &model_name)
{
    // Extract model family from the full model name
    std::string model_family = extract_model_family(model_name);
    
    // Factory logic: create model instance based on model family
    // Config is automatically selected based on model family
    if (model_family == "qwen3")
    {
        // Create Qwen3Model with default config
        // The config values can be customized later if needed based on model size
        Qwen3Config config;
        return std::make_unique<Qwen3Model>(config);
    }
    
    // Add more model families here as they are implemented
    // Example:
    // if (model_family == "llama")
    // {
    //     LlamaConfig config;
    //     return std::make_unique<LlamaModel>(config);
    // }
    
    throw std::invalid_argument("Unsupported model name: " + model_name + " (family: " + model_family + ")");
}

void InferenceEngine::load_weights(const std::string &safetensor_path, bool use_mmap)
{
    if (!model_)
    {
        throw std::runtime_error("Model instance is null");
    }
    model_->load_weights(safetensor_path, use_mmap);
}

void InferenceEngine::reset_cache()
{
    if (!model_)
    {
        throw std::runtime_error("Model instance is null");
    }
    model_->reset_cache();
}

void InferenceEngine::process_prompt_token(int token_id)
{
    if (!model_)
    {
        throw std::runtime_error("Model instance is null");
    }
    model_->process_prompt_token(token_id);
}

const std::vector<float> &InferenceEngine::predict_next_token(int token_id)
{
    if (!model_)
    {
        throw std::runtime_error("Model instance is null");
    }
    return model_->predict_next_token(token_id);
}

std::size_t InferenceEngine::tokens_processed() const noexcept
{
    if (!model_)
    {
        return 0;
    }
    return model_->tokens_processed();
}

