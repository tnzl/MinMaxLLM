#include <cpu_ops/matmul.h>
#include <cpu_ops/linear.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include "../../test_utils.cpp"

int main(int argc, char **argv)
{
    std::cout << "=== Qwen3 MLP Gate Module Test C++ run ===\n";

    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <weight.txt> <output.txt> <N> <K> <M>\n";
        std::cerr << "Example shape: 2,256\n";
        return 1;
    }
    std::string input_path = argv[1];
    std::string weight_path = argv[2];
    std::string output_path = argv[3];
    int N = std::stoi(argv[4]);
    int K = std::stoi(argv[5]);
    int M = std::stoi(argv[6]);

    std::vector<float> input(N * K);
    load_txt(input_path, input.data());

    std::vector<float> weight(K * M);
    load_bin(weight_path, weight.data(), K * M);

    // Print statements
    // print first 10 and last 10 elements of input and weight for verification
    std::cout << "Input (first 10 elements): ";
    for (int i = 0; i < std::min(10, N * K); ++i)
        std::cout << input[i] << " ";
    std::cout << "\n";
    std::cout << "Input (last 10 elements): ";
    for (int i = std::max(0, N * K - 10); i < N * K; ++i)
        std::cout << input[i] << " ";
    std::cout << "\n";

    std::cout << "\nWeight (first 10 elements): ";
    for (int i = 0; i < std::min(10, K * M); ++i)
        std::cout << weight[i] << " ";
    std::cout << "\n";
    std::cout << "Weight (last 10 elements): ";
    for (int i = std::max(0, K * M - 10); i < K * M; ++i)
        std::cout << weight[i] << " ";
    std::cout << "\n";

    // Allocate aligned memory for output
    std::vector<float> output(M * N);

    // Perform matrix multiplication
    std::cout << "\nPerforming matrix multiplication: (" << N << "x" << K << ") * (" << K << "x" << M << ") = (" << N << "x" << M << ")\n";

    // // transpose weight matrix input is (M, K), output is (K, M)
    // std::vector<float> weight_t(M*K);
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < K; ++j) {
    //         weight_t[j*M + i] = weight[i*K + j];
    //     }
    // }

    // hyperOptimizedMatMul(input.data(), weight_t.data(), output.data(), N, M, K);

    linear_avx2_omp(input.data(), weight.data(), N, K, M, output.data());

    std::cout << "Matrix multiplication completed.\n";

    std::cout << "Output (first 10 elements): ";
    for (int i = 0; i < std::min(10, N * M); ++i)
        std::cout << output[i] << " ";
    std::cout << "\n";
    std::cout << "Output (last 10 elements): ";
    for (int i = std::max(0, N * M - 10); i < N * M; ++i)
        std::cout << output[i] << " ";
    std::cout << "\n";

    save_txt(output_path, output);
    std::cout << "✅ Output saved to " << output_path << std::endl;
    return 0;
}