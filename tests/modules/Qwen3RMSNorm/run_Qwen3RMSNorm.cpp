#include <cpu_ops/rmsnorm.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include "../../test_utils.cpp"

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <weight.txt> <output.txt> <shape> <eps>\n";
        std::cerr << "Example shape: 2,256\n";
        return 1;
    }
    std::string input_path = argv[1];
    std::string weight_path = argv[2];
    std::string output_path = argv[3];
    std::string shape_str = argv[4];
    float eps = std::stof(argv[5]);

    std::vector<int> shape = parse_shape(shape_str);
    int batch_size, hidden_size;
    if (shape.size() == 1) {
        batch_size = 1;
        hidden_size = shape[0];
    } else if (shape.size() == 2) {
        batch_size = shape[0];
        hidden_size = shape[1];
    } else if (shape.size() > 2) {
        batch_size = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) batch_size *= shape[i];
        hidden_size = shape.back();
    } else {
        std::cerr << "Invalid shape: must have at least 1 dimension\n";
        return 1;
    }

    std::vector<float> input = load_txt(input_path);
    std::vector<float> weight = load_txt(weight_path);
    if ((int)input.size() != batch_size * hidden_size) {
        std::cerr << "Input size mismatch: " << input.size() << " vs " << batch_size << "*" << hidden_size << "\n";
        return 1;
    }
    if ((int)weight.size() != hidden_size) {
        std::cerr << "Weight size mismatch: " << weight.size() << " vs " << hidden_size << "\n";
        return 1;
    }
    std::vector<float> output(batch_size * hidden_size);
    rmsnorm_avx2(input.data(), weight.data(), output.data(), batch_size, hidden_size, eps);
    save_txt(output_path, output);
    std::cout << "✅ Output saved to " << output_path << std::endl;
    return 0;
}