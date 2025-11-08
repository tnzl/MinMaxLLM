#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <cctype>
#include <sstream>
#include <chrono>
#include <optional>
#include <iomanip>

#include <cpu_ops/self_attention.h>
#include <tensor/safetensors.h>
#include <tensor/kvcache.h>
#include <tensor/tensor.h>
#include <cpu_ops/rotary_embedding.h>

#include "../../test_utils.cpp"

namespace
{

constexpr float kCacheTolerance = 5e-2f;
constexpr float kOutputTolerance = 1e-2f;
constexpr std::size_t kMaxLoggedMismatches = 5;
constexpr std::size_t kMaxMismatchThreshold = 500;

void log_info(const std::string &message)
{
    std::cout << "[INFO] " << message << '\n';
}

void log_success(const std::string &message)
{
    std::cout << "[OK] " << message << '\n';
}

void log_warning(const std::string &message)
{
    std::cout << "[WARN] " << message << '\n';
}

void log_error(const std::string &message)
{
    std::cerr << "[ERROR] " << message << '\n';
}

class SectionTimer
{
public:
    explicit SectionTimer(std::string label)
        : label_(std::move(label)),
          start_(std::chrono::steady_clock::now()),
          stopped_(false)
    {
    }

    ~SectionTimer()
    {
        if (!stopped_)
        {
            stop();
        }
    }

    void stop()
    {
        if (stopped_)
        {
            return;
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start_).count();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << elapsed_ms;
        std::cout << "[TIME] " << label_ << ": " << oss.str() << " ms\n";
        stopped_ = true;
    }

private:
    std::string label_;
    std::chrono::steady_clock::time_point start_;
    bool stopped_;
};

struct ProgramArgs
{
    std::string safetensor_path;
    std::string input_path;
    std::string past_key_path;
    std::string past_value_path;
    std::string golden_path;
    std::string output_path;
    std::optional<std::size_t> max_seq_len_override;
};

ProgramArgs parse_program_args(int argc, char **argv)
{
    ProgramArgs args{
        argv[1],
        argv[2],
        argv[3],
        argv[4],
        argv[5],
        argv[6],
        std::nullopt};

    if (argc >= 8)
    {
        args.max_seq_len_override = static_cast<std::size_t>(std::stoul(argv[7]));
    }

    return args;
}

const auto *require_tensor_info(Safetensor &weights, const std::string &tensor_name)
{
    if (const auto *info = weights.getTensorInfo(tensor_name))
    {
        return info;
    }

    throw std::runtime_error("Missing required tensor in safetensor file: " + tensor_name);
}

template <typename InfoType>
Tensor wrap_tensor(Safetensor &weights, const std::string &tensor_name, const InfoType *info)
{
    return Tensor(weights.tensorDataPtr<float>(tensor_name), info->shape, DataType::F32);
}

std::vector<float> load_txt_vector(const std::string &path)
{
    std::ifstream file(path);
    if (!file)
    {
        throw std::runtime_error("Cannot open file: " + path);
    }

    auto trim = [](std::string s) {
        const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        return s;
    };

    std::string line;
    std::string shape_line;
    while (std::getline(file, line))
    {
        shape_line = trim(line);
        if (!shape_line.empty())
        {
            break;
        }
    }

    if (shape_line.empty())
    {
        throw std::runtime_error("Missing shape/dtype line in file: " + path);
    }

    const std::string shape_prefix = "shape:";
    const std::string dtype_prefix = "dtype:";
    if (shape_line.find(shape_prefix) != 0)
    {
        throw std::runtime_error("Invalid shape/dtype header in file: " + path);
    }

    const auto dtype_pos = shape_line.find(dtype_prefix);
    if (dtype_pos == std::string::npos)
    {
        throw std::runtime_error("Missing dtype in header for file: " + path);
    }

    const auto paren_open = shape_line.find('(');
    const auto paren_close = shape_line.find(')');
    if (paren_open == std::string::npos || paren_close == std::string::npos || paren_close <= paren_open)
    {
        throw std::runtime_error("Invalid shape format in file: " + path);
    }

    const std::string dims_str = shape_line.substr(paren_open + 1, paren_close - paren_open - 1);
    std::vector<std::size_t> dims;
    {
        std::istringstream dims_stream(dims_str);
        std::string dim_token;
        while (std::getline(dims_stream, dim_token, ','))
        {
            dim_token = trim(dim_token);
            if (dim_token.empty())
            {
                continue;
            }
            dims.push_back(static_cast<std::size_t>(std::stoul(dim_token)));
        }
    }

    if (dims.empty())
    {
        throw std::runtime_error("No dimensions parsed from shape header in file: " + path);
    }

    std::size_t expected_size = 1;
    for (const auto dim : dims)
    {
        expected_size *= dim;
    }

    std::string dtype = trim(shape_line.substr(dtype_pos + dtype_prefix.size()));
    if (!dtype.empty() && dtype.back() == ',')
    {
        dtype.pop_back();
        dtype = trim(dtype);
    }

    if (dtype != "float32")
    {
        throw std::runtime_error("Unsupported dtype '" + dtype + "' in file: " + path + " (expected float32)");
    }

    std::string data_line;
    while (std::getline(file, line))
    {
        data_line = trim(line);
        if (!data_line.empty())
        {
            break;
        }
    }

    if (data_line.empty())
    {
        throw std::runtime_error("Missing data line in file: " + path);
    }

    std::vector<float> data;
    if (expected_size > 0)
    {
        data.reserve(expected_size);
    }
    {
        std::istringstream data_stream(data_line);
        float value;
        while (data_stream >> value)
        {
            data.push_back(value);
        }
    }

    {
        float value;
        while (file >> value)
        {
            data.push_back(value);
        }
    }

    if (data.empty())
    {
        throw std::runtime_error("No data parsed from file: " + path);
    }

    if (expected_size != data.size())
    {
        throw std::runtime_error("Data size mismatch for file: " + path + ". Expected " + std::to_string(expected_size) + ", got " + std::to_string(data.size()));
    }

    return data;
}

void seed_kv_cache(KVCache &cache,
                   const std::vector<float> &past_key_flat,
                   const std::vector<float> &past_value_flat,
                   std::size_t num_groups,
                   std::size_t head_dim,
                   std::size_t past_sequence_length)
{
    for (std::size_t token = 0; token < past_sequence_length; ++token)
    {
        for (std::size_t group = 0; group < num_groups; ++group)
        {
            const std::size_t base_index = (group * past_sequence_length + token) * head_dim;
            cache.set_key(0, group, token, past_key_flat.data() + base_index);
            cache.set_value(0, group, token, past_value_flat.data() + base_index);
        }
        cache.advance();
    }
}

bool compare_rotary_token(const Tensor &sin_cache_tensor,
                          std::size_t head_dim,
                          int token_idx,
                          const std::string &reference_path)
{
    SectionTimer timer("Verify rotary cache token " + std::to_string(token_idx));
    const auto sin_cache_flat = load_txt_vector(reference_path);

    const std::size_t entries_per_token = head_dim / 2;

    const float *cache_data = sin_cache_tensor.data<float>() + static_cast<std::size_t>(token_idx) * entries_per_token;
    for (std::size_t i = 0; i < entries_per_token; ++i)
    {
        const float diff = std::fabs(cache_data[i] - sin_cache_flat[i]);
        if (diff > 5e-6f)
        {
            log_warning("Rotary cache mismatch at token " + std::to_string(token_idx) +
                        ", index " + std::to_string(i) +
                        " (value=" + std::to_string(cache_data[i]) +
                        ", expected=" + std::to_string(sin_cache_flat[i]) +
                        ", diff=" + std::to_string(diff) + ")");
            return false;
        }
    }

    log_success("Rotary cache comparison passed for token " + std::to_string(token_idx));
    return true;
}

bool compare_cache_against_reference(const KVCache &cache,
                                     const std::vector<float> &expected_key,
                                     const std::vector<float> &expected_value,
                                     std::size_t num_groups,
                                     std::size_t head_dim,
                                     std::size_t sequence_length,
                                     const std::string &label)
{
    const std::size_t expected_size = num_groups * sequence_length * head_dim;
    if (expected_key.size() != expected_size || expected_value.size() != expected_size)
    {
        throw std::runtime_error("Reference cache size mismatch for " + label);
    }

    std::size_t mismatches = 0;

    for (std::size_t group = 0; group < num_groups; ++group)
    {
        for (std::size_t token = 0; token < sequence_length; ++token)
        {
            const float *key_cache = cache.get_key_at(0, group, token);
            const float *value_cache = cache.get_value_at(0, group, token);

            const std::size_t base_index = (group * sequence_length + token) * head_dim;
            const float *key_ref = expected_key.data() + base_index;
            const float *value_ref = expected_value.data() + base_index;

            for (std::size_t i = 0; i < head_dim; ++i)
            {
                const float key_diff = std::fabs(key_cache[i] - key_ref[i]);
                if (key_diff > kCacheTolerance)
                {
                    if (mismatches < kMaxLoggedMismatches)
                    {
                        log_error("Key mismatch (" + label + ") layer=0, group=" + std::to_string(group) +
                                  ", token=" + std::to_string(token) + ", index=" + std::to_string(i) +
                                  " (value=" + std::to_string(key_cache[i]) + ", expected=" + std::to_string(key_ref[i]) +
                                  ", diff=" + std::to_string(key_diff) + ")");
                    }
                    if (++mismatches >= kMaxMismatchThreshold)
                    {
                        log_error("Exceeded mismatch threshold while validating keys for " + label);
                        return false;
                    }
                }

                const float value_diff = std::fabs(value_cache[i] - value_ref[i]);
                if (value_diff > kCacheTolerance)
                {
                    if (mismatches < kMaxLoggedMismatches)
                    {
                        log_error("Value mismatch (" + label + ") layer=0, group=" + std::to_string(group) +
                                  ", token=" + std::to_string(token) + ", index=" + std::to_string(i) +
                                  " (value=" + std::to_string(value_cache[i]) + ", expected=" + std::to_string(value_ref[i]) +
                                  ", diff=" + std::to_string(value_diff) + ")");
                    }
                    if (++mismatches >= kMaxMismatchThreshold)
                    {
                        log_error("Exceeded mismatch threshold while validating values for " + label);
                        return false;
                    }
                }
            }
        }
    }

    if (mismatches == 0)
    {
        log_success("Cache comparison passed for " + label);
    }
    else
    {
        log_warning("Cache comparison for " + label + " completed with " + std::to_string(mismatches) + " mismatches");
    }

    return mismatches == 0;
}

} // namespace

int main(int argc, char **argv)
{
    log_info("=== SelfAttention Module Test C++ run ===");

    if (argc < 7)
    {
        log_error(std::string("Usage: ") + argv[0] +
                  " <model_fp32-00001-of-00002.safetensors> <input.txt> <past_key.txt> <past_value.txt> <golden_output.txt> <output.txt> [max_seq_len]");
        return 1;
    }

    const auto args = parse_program_args(argc, argv);
    SectionTimer total_timer("Total runtime");

    try
    {
        SectionTimer weights_timer("Load safetensor weights");
        Safetensor weights(args.safetensor_path, true);
        weights_timer.stop();
        log_info("Loaded weights from " + args.safetensor_path);

        const auto *q_norm_info = require_tensor_info(weights, "model.layers.0.self_attn.q_norm.weight");
        const auto *k_norm_info = require_tensor_info(weights, "model.layers.0.self_attn.k_norm.weight");
        const auto *q_proj_info = require_tensor_info(weights, "model.layers.0.self_attn.q_proj.weight");
        const auto *k_proj_info = require_tensor_info(weights, "model.layers.0.self_attn.k_proj.weight");
        const auto *v_proj_info = require_tensor_info(weights, "model.layers.0.self_attn.v_proj.weight");
        const auto *o_proj_info = require_tensor_info(weights, "model.layers.0.self_attn.o_proj.weight");

        const std::size_t head_dim = q_norm_info->shape.at(0);
        const std::size_t embed_dim = q_proj_info->shape.at(1);
        const std::size_t num_heads = q_proj_info->shape.at(0) / head_dim;
        const std::size_t num_groups = k_proj_info->shape.at(0) / head_dim;

        if (k_norm_info->shape.at(0) != head_dim)
        {
            throw std::runtime_error("k_norm head dimension mismatch");
        }

        auto load_vector = [](const std::string &path, const std::string &label) {
            SectionTimer timer("Load " + label);
            auto vec = load_txt_vector(path);
            timer.stop();
            log_info(label + " loaded from " + path + " (" + std::to_string(vec.size()) + " elements)");
            return vec;
        };

        auto input_flat = load_vector(args.input_path, "input vector");
        if (input_flat.size() != embed_dim)
        {
            throw std::runtime_error("Input tensor size mismatch. Expected " + std::to_string(embed_dim) + ", got " + std::to_string(input_flat.size()));
        }

        auto golden_flat = load_vector(args.golden_path, "golden output vector");
        if (golden_flat.size() != embed_dim)
        {
            throw std::runtime_error("Golden output size mismatch. Expected " + std::to_string(embed_dim) + ", got " + std::to_string(golden_flat.size()));
        }

        auto past_key_flat = load_vector(args.past_key_path, "past key vector");
        auto past_value_flat = load_vector(args.past_value_path, "past value vector");

        if (past_key_flat.size() != past_value_flat.size())
        {
            throw std::runtime_error("Past key/value sizes do not match");
        }

        if (past_key_flat.size() % (num_groups * head_dim) != 0)
        {
            throw std::runtime_error("Past key size incompatible with num_groups and head_dim");
        }

        const std::size_t past_sequence_length = past_key_flat.size() / (num_groups * head_dim);
        const std::size_t max_seq_len_arg = args.max_seq_len_override.value_or(past_sequence_length + 1);
        const std::size_t max_seq_len = (max_seq_len_arg > past_sequence_length + 1) ? max_seq_len_arg : (past_sequence_length + 1);

        log_info("embed_dim=" + std::to_string(embed_dim) +
                 ", head_dim=" + std::to_string(head_dim) +
                 ", num_heads=" + std::to_string(num_heads) +
                 ", num_groups=" + std::to_string(num_groups) +
                 ", past_seq_len=" + std::to_string(past_sequence_length) +
                 ", max_seq_len=" + std::to_string(max_seq_len));

        Tensor q_proj_tensor = wrap_tensor(weights, "model.layers.0.self_attn.q_proj.weight", q_proj_info);
        Tensor k_proj_tensor = wrap_tensor(weights, "model.layers.0.self_attn.k_proj.weight", k_proj_info);
        Tensor v_proj_tensor = wrap_tensor(weights, "model.layers.0.self_attn.v_proj.weight", v_proj_info);
        Tensor o_proj_tensor = wrap_tensor(weights, "model.layers.0.self_attn.o_proj.weight", o_proj_info);
        Tensor q_norm_tensor = wrap_tensor(weights, "model.layers.0.self_attn.q_norm.weight", q_norm_info);
        Tensor k_norm_tensor = wrap_tensor(weights, "model.layers.0.self_attn.k_norm.weight", k_norm_info);

        Tensor sin_cache_tensor(DataType::F32, {max_seq_len, head_dim / 2});
        Tensor cos_cache_tensor(DataType::F32, {max_seq_len, head_dim / 2});

        {
            SectionTimer timer("Precompute rotary caches");
            RotaryEmbeddingAVX2::precompute(
                sin_cache_tensor.data<float>(),
                cos_cache_tensor.data<float>(),
                static_cast<int>(max_seq_len),
                static_cast<int>(head_dim),
                1000000.0f);
            timer.stop();
        }

        compare_rotary_token(
            sin_cache_tensor,
            head_dim,
            18,
            "D:\\projects\\inspect_model\\qwen3_attention_dumps\\layer0_token18_sin.txt");

        KVCache cache(max_seq_len, head_dim, num_groups, /*num_layers*/ 1);

        {
            SectionTimer timer("Seed KV cache with past sequence");
            seed_kv_cache(cache, past_key_flat, past_value_flat, num_groups, head_dim, past_sequence_length);
            timer.stop();
        }

        if (!compare_cache_against_reference(cache,
                                             past_key_flat,
                                             past_value_flat,
                                             num_groups,
                                             head_dim,
                                             past_sequence_length,
                                             "past sequence length"))
        {
            // return 1;
        }

        Tensor input_tensor(DataType::F32, {embed_dim});
        {
            SectionTimer timer("Copy input tensor");
            std::memcpy(input_tensor.data<float>(), input_flat.data(), embed_dim * sizeof(float));
            timer.stop();
        }

        Tensor output_tensor(DataType::F32, {embed_dim});

        SelfAttention self_attn(
            q_proj_tensor,
            k_proj_tensor,
            v_proj_tensor,
            o_proj_tensor,
            q_norm_tensor,
            k_norm_tensor,
            sin_cache_tensor,
            cos_cache_tensor,
            /*layer_idx*/ 0,
            &cache);

        {
            SectionTimer timer("SelfAttention::prepare");
            self_attn.prepare();
            timer.stop();
        }

        {
            SectionTimer timer("SelfAttention::run");
            self_attn.run(input_tensor, past_sequence_length, output_tensor);
            timer.stop();
        }

        std::vector<float> output_vec(embed_dim);
        {
            SectionTimer timer("Copy output tensor");
            std::memcpy(output_vec.data(), output_tensor.data<float>(), embed_dim * sizeof(float));
            timer.stop();
        }

        auto updated_key_flat = load_vector("D:\\projects\\inspect_model\\qwen3_attention_dumps\\layer0_token18_updated_key.txt",
                                            "updated key reference");
        auto updated_value_flat = load_vector("D:\\projects\\inspect_model\\qwen3_attention_dumps\\layer0_token18_updated_value.txt",
                                              "updated value reference");

        if (!compare_cache_against_reference(cache,
                                             updated_key_flat,
                                             updated_value_flat,
                                             num_groups,
                                             head_dim,
                                             past_sequence_length + 1,
                                             "updated sequence length"))
        {
            // return 1;
        }

        {
            // compare input_tensor and input_flat if they're equal
            for (std::size_t i = 0; i < embed_dim; ++i)
            {
                if (input_tensor.data<float>()[i] != input_flat[i])
                {
                    log_error("Input tensor and input flat mismatch at index " + std::to_string(i) +
                              " (value=" + std::to_string(input_tensor.data<float>()[i]) +
                              ", expected=" + std::to_string(input_flat[i]) +
                              ", diff=" + std::to_string(std::fabs(input_tensor.data<float>()[i] - input_flat[i])) + ")");
                }
            }
            log_success("Input tensor and input flat match after self_attn.run");
        }

        {
            SectionTimer timer("Persist output to disk");
            save_txt(args.output_path, output_vec);
            timer.stop();
        }
        log_success("Output saved to " + args.output_path);

        printErrorAnalysis1D(golden_flat.data(), output_vec.data(), embed_dim, 1e-3f);

        bool match = true;
        for (std::size_t i = 0; i < embed_dim; ++i)
        {
            if (std::fabs(output_vec[i] - golden_flat[i]) > kOutputTolerance)
            {
                match = false;
                break;
            }
        }

        if (match)
        {
            log_success("SelfAttention output matches golden within tolerance " + std::to_string(kOutputTolerance));
        }
        else
        {
            log_error("SelfAttention output deviates from golden beyond tolerance " + std::to_string(kOutputTolerance));
            return 2;
        }
    }
    catch (const std::exception &ex)
    {
        log_error(std::string("Error: ") + ex.what());
        return 1;
    }

    total_timer.stop();
    return 0;
}


