#include <safetensors/safetensors.h>
#include <cpu_ops/linear.h>
#include <cpu_ops/silu_avx2.h>
#include <cpu_ops/elemwise_mul.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include "../../test_utils.cpp"

void qwen3_mlp(const float *input, const float *gate_weight, const float *up_weight, const float *down_weight, int N, int input_dim, int up_dim, int output_dim, float *output)
{
    float *intermediate_silu = static_cast<float *>(_aligned_malloc(N * up_dim * sizeof(float), 32));
    float *intermediate_up = static_cast<float *>(_aligned_malloc(N * up_dim * sizeof(float), 32));

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

    // Free intermediate buffers
    _aligned_free(intermediate_silu);
    _aligned_free(intermediate_up);
}

int main(int argc, char **argv)
{
    std::cout << "=== Qwen3 MLP Module Test C++ run ===\n";
    if (argc < 8)
    {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <weight.txt> <output.txt> <N> <K> <M>\n";
        std::cerr << "Example shape: 2,256\n";
        return 1;
    }
    std::string safetensors_path = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    int N = std::stoi(argv[4]);
    int input_dim = std::stoi(argv[5]);
    int up_dim = std::stoi(argv[6]);
    int output_dim = std::stoi(argv[7]);
    int layer_idx = std::stoi(argv[8]);

    // Print configuration
    std::cout << "Qwen3 MLP Configuration:\n";
    std::cout << "  Input file: " << input_path << "\n";
    std::cout << "  Safetensors file: " << safetensors_path << "\n";
    std::cout << "  Output file: " << output_path << "\n";
    std::cout << "  Batch size (N): " << N << "\n";
    std::cout << "  Input dimension (K): " << input_dim << "\n";
    std::cout << "  Up projection dimension: " << up_dim << "\n";
    std::cout << "  Output dimension (M): " << output_dim << "\n";
    std::cout << "  Layer index: " << layer_idx << "\n";
    std::cout << "----------------------------------------\n";

    std::vector<float> input(N * input_dim);
    load_txt(input_path, input.data());

    SafeTensor st(safetensors_path);

    std::string gate_wt_key = "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight";
    std::string up_wt_key = "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight";
    std::string down_wt_key = "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight";

    // check keys exist
    if (st.getTensorInfo(gate_wt_key) == nullptr ||
        st.getTensorInfo(up_wt_key) == nullptr ||
        st.getTensorInfo(down_wt_key) == nullptr)
    {
        std::cerr << "Error: One or more tensor keys not found in safetensors file.\n";
        return 1;
    }

    const float *gate_wt_ptr = reinterpret_cast<const float *>(st.tensorDataPtr(gate_wt_key));
    const float *up_wt_ptr = reinterpret_cast<const float *>(st.tensorDataPtr(up_wt_key));
    const float *down_wt_ptr = reinterpret_cast<const float *>(st.tensorDataPtr(down_wt_key));

    // float *output = static_cast<float *>(_aligned_malloc(N * output_dim * sizeof(float), 32));
    std::vector<float> output(N * output_dim);

    try
    {
        qwen3_mlp(input.data(), gate_wt_ptr, up_wt_ptr, down_wt_ptr, N, input_dim, up_dim, output_dim, output.data());
        std::cout << "✅ Qwen3 MLP computation completed.\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during Qwen3 MLP computation: " << e.what() << "\n";
        return 1;
    }

    // Save output
    save_txt(output_path, output);
    std::cout << "✅ Output saved to " << output_path << std::endl;

    return 0;
}
