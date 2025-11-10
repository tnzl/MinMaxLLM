#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

#include <models/qwen3model.h>

namespace
{
void print_usage(const char *program)
{
    std::cerr << "Usage: " << program << " <model.safetensors> <prompt_tokens.txt> <max_new_tokens>\n";
}

std::string read_file_to_string(const std::string &path)
{
    std::ifstream file(path);
    if (!file)
    {
        throw std::runtime_error("Failed to open prompt file: " + path);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<int> load_prompt_tokens(const std::string &path)
{
    const std::string content = read_file_to_string(path);

    std::vector<int> tokens;
    std::string token_str;
    std::stringstream ss(content);

    while (std::getline(ss, token_str, ','))
    {
        if (token_str.empty())
        {
            continue;
        }

        std::stringstream token_stream(token_str);
        int token = 0;
        token_stream >> token;
        if (token_stream.fail())
        {
            throw std::runtime_error("Invalid token entry in prompt file: '" + token_str + "'");
        }
        tokens.push_back(token);
    }

    return tokens;
}

int greedy_argmax(const std::vector<float> &probabilities)
{
    const auto it = std::max_element(probabilities.begin(), probabilities.end());
    return static_cast<int>(std::distance(probabilities.begin(), it));
}

std::size_t current_memory_usage()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters;
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters), sizeof(counters)))
    {
        return static_cast<std::size_t>(counters.WorkingSetSize);
    }
    return 0;
#else
    return 0;
#endif
}

double bytes_to_megabytes(std::size_t bytes)
{
    constexpr double one_mb = 1024.0 * 1024.0;
    return static_cast<double>(bytes) / one_mb;
}
} // namespace

int main(int argc, char **argv)
{
    using Clock = std::chrono::steady_clock;

    if (argc != 4)
    {
        print_usage(argv[0]);
        return 1;
    }

    const std::string safetensor_path = argv[1];
    const std::string prompt_tokens_path = argv[2];
    const std::size_t max_new_tokens = static_cast<std::size_t>(std::stoul(argv[3]));

    try
    {
        const auto prompt_tokens = load_prompt_tokens(prompt_tokens_path);

        const auto load_start = Clock::now();
        const auto memory_before_load = current_memory_usage();
        Qwen3Config config;
        Qwen3Model model(config);
        model.load_weights(safetensor_path, true);
        const auto load_end = Clock::now();
        const auto load_duration = load_end - load_start;
        const auto memory_after_load = current_memory_usage();

        std::vector<int> generated_tokens;

        int current_token = config.bos_token_id;
        const bool do_generation = max_new_tokens > 0;
        std::size_t prompt_tokens_to_process = prompt_tokens.size();

        if (do_generation && prompt_tokens_to_process > 0)
        {
            --prompt_tokens_to_process;
            current_token = prompt_tokens.back();
        }

        const auto prompt_start = Clock::now();
        for (std::size_t i = 0; i < prompt_tokens_to_process; ++i)
        {
            model.process_prompt_token(prompt_tokens[i]);
        }
        const auto prompt_end = Clock::now();
        const auto prompt_duration = prompt_end - prompt_start;
        const auto memory_after_prompt = current_memory_usage();

        std::cout << "Processed " << prompt_tokens_to_process << " prompt "
                  << (prompt_tokens_to_process == 1 ? "token" : "tokens") << "\n";

        auto generation_duration = Clock::duration::zero();
        auto memory_after_generation = memory_after_prompt;
        if (do_generation)
        {
            if (!prompt_tokens.empty() && prompt_tokens_to_process < prompt_tokens.size())
            {
                current_token = prompt_tokens.back();
            }

            const auto generation_start = Clock::now();
            std::cout << "Generated tokens:";
            for (std::size_t step = 0; step < max_new_tokens; ++step)
            {
                const auto &probabilities = model.predict_next_token(current_token);
                const int next_token = greedy_argmax(probabilities);
                generated_tokens.push_back(next_token);
                std::cout << next_token << " ";

                if (next_token == config.eos_token_id)
                {
                    break;
                }

                current_token = next_token;
            }
            std::cout << "\n";
            const auto generation_end = Clock::now();
            generation_duration = generation_end - generation_start;
            memory_after_generation = current_memory_usage();
        }

        const double load_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(load_duration).count();
        const double prompt_total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(prompt_duration).count();
        const double prompt_ms_per_token = prompt_tokens_to_process > 0
                                               ? prompt_total_ms / static_cast<double>(prompt_tokens_to_process)
                                               : 0.0;
        const std::size_t generation_steps = generated_tokens.size();
        const double generation_total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(generation_duration).count();
        const double generation_ms_per_token = generation_steps > 0
                                                   ? generation_total_ms / static_cast<double>(generation_steps)
                                                   : 0.0;
#if defined(_WIN32)
        const auto load_memory_delta = memory_after_load > memory_before_load ? memory_after_load - memory_before_load : 0;
        const auto prompt_memory_delta = memory_after_prompt > memory_after_load ? memory_after_prompt - memory_after_load : 0;
        const auto generation_memory_delta = memory_after_generation > memory_after_prompt ? memory_after_generation - memory_after_prompt : 0;
#endif

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Metrics:\n";
        std::cout << "  Model load time: " << load_ms << " ms\n";
#if defined(_WIN32)
        std::cout << "  Memory after load: " << bytes_to_megabytes(memory_after_load) << " MB";
        if (load_memory_delta > 0)
        {
            std::cout << " (+" << bytes_to_megabytes(load_memory_delta) << " MB)";
        }
        std::cout << "\n";
#else
        std::cout << "  Memory metrics available on Windows only\n";
#endif
        std::cout << "  Prompt processing total time: " << prompt_total_ms << " ms";
        if (prompt_tokens_to_process > 0)
        {
            std::cout << " (" << prompt_ms_per_token << " ms/token)";
        }
        std::cout << "\n";
#if defined(_WIN32)
        std::cout << "  Memory after prompt: " << bytes_to_megabytes(memory_after_prompt) << " MB";
        if (prompt_memory_delta > 0)
        {
            std::cout << " (+" << bytes_to_megabytes(prompt_memory_delta) << " MB)";
        }
        std::cout << "\n";
#endif
        if (do_generation)
        {
            std::cout << "  Generation total time: " << generation_total_ms << " ms";
            if (generation_steps > 0)
            {
                std::cout << " (" << generation_ms_per_token << " ms/token)";
            }
            std::cout << "\n";
            std::cout << "  Tokens generated: " << generation_steps << "\n";
#if defined(_WIN32)
            std::cout << "  Memory after generation: " << bytes_to_megabytes(memory_after_generation) << " MB";
            if (generation_memory_delta > 0)
            {
                std::cout << " (+" << bytes_to_megabytes(generation_memory_delta) << " MB)";
            }
            std::cout << "\n";
#endif
        }
        else
        {
            std::cout << "  Generation skipped (max_new_tokens == 0)\n";
        }

        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

