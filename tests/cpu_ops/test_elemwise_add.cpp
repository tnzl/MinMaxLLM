#include <chrono>
#include <cpu_ops/elemwise_add.h>
#include <iostream>
#include <random>
#include <vector>
#include "../test_utils.cpp"

std::vector<float> naive_elemwise_add(const float* a, const float* b, int batch_size, int hidden_size) {
    std::vector<float> out(batch_size * hidden_size);
    for (int bidx = 0; bidx < batch_size; ++bidx) {
        for (int d = 0; d < hidden_size; ++d) {
            out[bidx * hidden_size + d] = a[bidx * hidden_size + d] + b[bidx * hidden_size + d];
        }
    }
    return out;
}

int main() {
    const int batch_size = 8;
    const int hidden_size = 2506;
    std::vector<float> a(batch_size * hidden_size);
    std::vector<float> b(batch_size * hidden_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : a) x = dist(gen);
    for (auto& x : b) x = dist(gen);

    constexpr int num_iters = 100;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> ref;
    for (int i = 0; i < num_iters; ++i) {
        ref = naive_elemwise_add(a.data(), b.data(), batch_size, hidden_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double naive_time_avg_ns = static_cast<double>(naive_time_ns) / num_iters;

    std::vector<float> out(batch_size * hidden_size);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        elemwise_add_avx2_omp(a.data(), b.data(), out.data(), batch_size, hidden_size);
    }
    end = std::chrono::high_resolution_clock::now();
    auto avx_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avx_time_avg_ns = static_cast<double>(avx_time_ns) / num_iters;

    bool pass = validateResults(ref.data(), out.data(), batch_size, hidden_size, 0.00001f);
    printErrorAnalysis(ref.data(), out.data(), batch_size, hidden_size);
    if (!pass) {
        std::cerr << "Error: Elementwise add results don't match!\n";
        return 1;
    }

    std::cout << "Elementwise add correctness test passed!\n";
    std::cout << "Naive add Avg Latency: " << naive_time_avg_ns << " ns\n";
    std::cout << "AVX+OMP add Avg Latency: " << avx_time_avg_ns << " ns\n";
    std::cout << "Speedup: " << naive_time_avg_ns / avx_time_avg_ns << "x\n";
    return 0;
}


