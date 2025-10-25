#include <tensor/safetensors.h>
#include <cpu_ops/linear.h>
#include <cpu_ops/silu_avx2.h>
#include <cpu_ops/elemwise_mul.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include "../../test_utils.cpp"
#include <windows.h>
#include <psapi.h>

void printPeakMemoryUsage()
{
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    {
        std::cout << "Peak Working Set Size: "
                  << pmc.PeakWorkingSetSize / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    }
}

class Qwen3MLPTester
{
private:
    std::string safetensors_path;
    std::string input_path;
    std::string output_path;
    int N;
    int input_dim;
    int up_dim;
    int output_dim;
    int layer_idx;
    bool use_mmap;
    bool use_advise;

    std::vector<float> input;
    std::vector<float> output;
    Safetensor *st;

    const float *gate_wt_ptr;
    const float *up_wt_ptr;
    const float *down_wt_ptr;

public:
    Qwen3MLPTester(int argc, char **argv)
    {
        if (argc < 11)
        {
            throw std::invalid_argument("Usage: " + std::string(argv[0]) +
                                        " <safetensors> <input.txt> <output.txt> <N> <K> <M> <layer_idx> <use_mmap(1/0)> <use_advise(1/0)>");
        }

        safetensors_path = argv[1];
        input_path = argv[2];
        output_path = argv[3];
        N = std::stoi(argv[4]);
        input_dim = std::stoi(argv[5]);
        up_dim = std::stoi(argv[6]);
        output_dim = std::stoi(argv[7]);
        layer_idx = std::stoi(argv[8]);
        use_mmap = std::stoi(argv[9]) != 0;
        use_advise = std::stoi(argv[10]) != 0;

        input.resize(N * input_dim);
        output.resize(N * output_dim);

        st = nullptr;
        gate_wt_ptr = nullptr;
        up_wt_ptr = nullptr;
        down_wt_ptr = nullptr;
    }

    ~Qwen3MLPTester()
    {
        if (st)
        {
            delete st;
        }
    }

    void loadInput()
    {
        load_txt(input_path, input.data());
    }

    void loadWeights()
    {
        auto start_st = std::chrono::high_resolution_clock::now();
        st = new Safetensor(safetensors_path, use_mmap);
        auto end_st = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed_st = end_st - start_st;
        std::cout << "✅ Safetensors" << (use_mmap ? "(mmap)" : "") << " loaded in " << elapsed_st.count() << " us\n";

        std::string gate_wt_key = "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight";
        std::string up_wt_key = "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight";
        std::string down_wt_key = "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight";

        if (st->getTensorInfo(gate_wt_key) == nullptr ||
            st->getTensorInfo(up_wt_key) == nullptr ||
            st->getTensorInfo(down_wt_key) == nullptr)
        {
            throw std::runtime_error("Error: One or more tensor keys not found in safetensors file.");
        }

        gate_wt_ptr = st->tensorDataPtr<float>(gate_wt_key);
        up_wt_ptr = st->tensorDataPtr<float>(up_wt_key);
        down_wt_ptr = st->tensorDataPtr<float>(down_wt_key);
    }

    void optimized_qwen3_mlp(const float *input, const float *gate_weight, const float *up_weight,
                             const float *down_weight, int N, int input_dim, int up_dim,
                             int output_dim, float *output, bool use_advise)
    {
        float *intermediate_silu = static_cast<float *>(_aligned_malloc(N * up_dim * sizeof(float), 32));
        float *intermediate_up = static_cast<float *>(_aligned_malloc(N * up_dim * sizeof(float), 32));

        // Prefetch ALL weights at the start
        if (use_advise)
        {
            size_t gate_size = input_dim * up_dim * sizeof(float);
            size_t up_size = input_dim * up_dim * sizeof(float);
            size_t down_size = up_dim * output_dim * sizeof(float);

            Safetensor::windows_advise((void *)gate_weight, gate_size);
            Safetensor::windows_advise((void *)up_weight, up_size);
            Safetensor::windows_advise((void *)down_weight, down_size);
        }

        // Gate projection
        linear_avx2_omp(input, gate_weight, N, input_dim, up_dim, intermediate_silu);

        // SiLU activation
        silu_avx2(intermediate_silu, intermediate_silu, N * up_dim);

        // Up projection
        linear_avx2_omp(input, up_weight, N, input_dim, up_dim, intermediate_up);

        // Element-wise multiplication
        elemwise_mul_avx2(intermediate_silu, intermediate_up, intermediate_up, N, up_dim);

        // Down projection
        linear_avx2_omp(intermediate_up, down_weight, N, up_dim, output_dim, output);

        _aligned_free(intermediate_silu);
        _aligned_free(intermediate_up);
    }

    void runBenchmark()
    {
        const int iterations = 10;
        double total_time = 0.0;

        // Warmup run for mmap+advise case to populate page cache
        if (use_mmap)
        {
            optimized_qwen3_mlp(input.data(), gate_wt_ptr, up_wt_ptr, down_wt_ptr,
                                N, input_dim, up_dim, output_dim, output.data(), true);
            // Clear output for actual timed runs
            std::fill(output.begin(), output.end(), 0.0f);
        }

        for (int iter = 0; iter < iterations; ++iter)
        {
            auto start = std::chrono::high_resolution_clock::now();

            optimized_qwen3_mlp(input.data(), gate_wt_ptr, up_wt_ptr, down_wt_ptr,
                                N, input_dim, up_dim, output_dim, output.data(),
                                use_mmap && use_advise);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> elapsed = end - start;
            total_time += elapsed.count();
        }

        double avg_time = total_time / iterations;
        std::cout << "C++ MLP execution time: " << avg_time << " us\n";
    }

    void saveOutput()
    {
        save_txt(output_path, output);
        std::cout << "✅ Output saved to " << output_path << std::endl;
    }

    void printConfig()
    {
        std::cout << "Qwen3 MLP Configuration:\n";
        std::cout << "  Safetensors file: " << safetensors_path << "\n";
        std::cout << "  Input file: " << input_path << "\n";
        std::cout << "  Output file: " << output_path << "\n";
        std::cout << "  Batch size (N): " << N << "\n";
        std::cout << "  Input dimension: " << input_dim << "\n";
        std::cout << "  Up projection dimension: " << up_dim << "\n";
        std::cout << "  Output dimension: " << output_dim << "\n";
        std::cout << "  Layer index: " << layer_idx << "\n";
        std::cout << "  Use mmap: " << (use_mmap ? "true" : "false") << "\n";
        std::cout << "  Use advise: " << (use_advise ? "true" : "false") << "\n";
        std::cout << "----------------------------------------\n";
    }
};

int main(int argc, char **argv)
{
    try
    {
        Qwen3MLPTester tester(argc, argv);

        // Print configuration
        tester.printConfig();

        // Load input data
        tester.loadInput();

        // Load weights from safetensors
        tester.loadWeights();

        // Run benchmark
        tester.runBenchmark();

        // Save results
        tester.saveOutput();

        // Print memory usage
        printPeakMemoryUsage();

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}