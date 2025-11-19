#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <cpu_ops/gqa.h>
#include <tensor/tensor.h>
#include "../test_utils.cpp"
#include <chrono>
#include <algorithm>
#include <iomanip>

int main()
{
    // Typical LLM GQA sizes
    const int num_heads = 32;
    const int kv_num_heads = 8;
    const int head_dim = 128;
    const int seq_len = 1048;
    const int max_seq_len = 1048;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Allocate input data
    std::vector<float> query(num_heads * head_dim);
    std::vector<float> key(kv_num_heads * max_seq_len * head_dim);
    std::vector<float> value(kv_num_heads * max_seq_len * head_dim);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : query)
        x = dist(gen);
    for (auto &x : key)
        x = dist(gen);
    for (auto &x : value)
        x = dist(gen);

    // Output buffers for each test
    std::vector<float> output_naive_func(num_heads * head_dim);
    std::vector<float> output_naive_op(num_heads * head_dim);
    std::vector<float> output_avx2_func(num_heads * head_dim);
    std::vector<float> output_avx2_op(num_heads * head_dim);

    std::cout << "=== GQA Test Suite ===\n";
    std::cout << "Configuration: A=" << num_heads << ", G=" << kv_num_heads 
              << ", h=" << head_dim << ", N=" << seq_len << ", N_max=" << max_seq_len 
              << ", scale=" << scale << "\n\n";

    // ============================================================================
    // Test 1: Naive function (golden reference)
    // ============================================================================
    std::cout << "Test 1: Naive Function (Golden Reference)\n";
    std::cout << "----------------------------------------\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    gqa_naive(
        query.data(), 
        key.data(), 
        value.data(), 
        output_naive_func.data(), 
        seq_len, 
        max_seq_len, 
        kv_num_heads, 
        num_heads, 
        head_dim, 
        scale
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_func_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Latency: " << naive_func_time << " us\n";
    std::cout << "Status: PASSED (golden reference)\n\n";

    // ============================================================================
    // Test 2: Naive Op Class
    // ============================================================================
    std::cout << "Test 2: Naive Op Class\n";
    std::cout << "----------------------------------------\n";
    
    // Create tensors
    Tensor query_tensor(DataType::F32, {static_cast<size_t>(num_heads), static_cast<size_t>(head_dim)});
    Tensor key_tensor(DataType::F32, {static_cast<size_t>(kv_num_heads), static_cast<size_t>(max_seq_len), static_cast<size_t>(head_dim)});
    Tensor value_tensor(DataType::F32, {static_cast<size_t>(kv_num_heads), static_cast<size_t>(max_seq_len), static_cast<size_t>(head_dim)});
    Tensor output_tensor(DataType::F32, {static_cast<size_t>(num_heads), static_cast<size_t>(head_dim)});
    
    // Copy data to tensors
    std::memcpy(query_tensor.data<float>(), query.data(), query.size() * sizeof(float));
    std::memcpy(key_tensor.data<float>(), key.data(), key.size() * sizeof(float));
    std::memcpy(value_tensor.data<float>(), value.data(), value.size() * sizeof(float));
    
    // Create and run naive op
    GQAOp naive_op(OpBackend::NAIVE, scale);
    naive_op.prepare();
    
    start = std::chrono::high_resolution_clock::now();
    naive_op.run(query_tensor, key_tensor, value_tensor, output_tensor, seq_len);
    end = std::chrono::high_resolution_clock::now();
    auto naive_op_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Copy output back
    std::memcpy(output_naive_op.data(), output_tensor.data<float>(), output_naive_op.size() * sizeof(float));
    
    std::cout << "Latency: " << naive_op_time << " us\n";
    printErrorAnalysis(output_naive_func.data(), output_naive_op.data(), num_heads, head_dim, "Naive Op vs Naive Function");
    std::cout << "\n";

    // ============================================================================
    // Test 3: AVX2 Function
    // ============================================================================
    std::cout << "Test 3: AVX2 Function\n";
    std::cout << "----------------------------------------\n";
    
    start = std::chrono::high_resolution_clock::now();
    optimized_gqa_forward(
        query.data(), 
        key.data(), 
        value.data(), 
        output_avx2_func.data(), 
        num_heads, 
        kv_num_heads, 
        head_dim, 
        seq_len, 
        max_seq_len, 
        scale
    );
    end = std::chrono::high_resolution_clock::now();
    auto avx2_func_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Latency: " << avx2_func_time << " us\n";
    printErrorAnalysis(output_naive_func.data(), output_avx2_func.data(), num_heads, head_dim, "AVX2 Function vs Naive Function");
    std::cout << "\n";

    // ============================================================================
    // Test 4: AVX2 Op Class
    // ============================================================================
    std::cout << "Test 4: AVX2 Op Class\n";
    std::cout << "----------------------------------------\n";
    
    // Reuse tensors (data already copied)
    Tensor output_tensor_avx2(DataType::F32, {static_cast<size_t>(num_heads), static_cast<size_t>(head_dim)});
    
    // Create and run AVX2 op
    GQAOp avx2_op(OpBackend::AVX2, scale);
    avx2_op.prepare();
    
    start = std::chrono::high_resolution_clock::now();
    avx2_op.run(query_tensor, key_tensor, value_tensor, output_tensor_avx2, seq_len);
    end = std::chrono::high_resolution_clock::now();
    auto avx2_op_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Copy output back
    std::memcpy(output_avx2_op.data(), output_tensor_avx2.data<float>(), output_avx2_op.size() * sizeof(float));
    
    std::cout << "Latency: " << avx2_op_time << " us\n";
    printErrorAnalysis(output_naive_func.data(), output_avx2_op.data(), num_heads, head_dim, "AVX2 Op vs Naive Function");
    std::cout << "\n";

    // ============================================================================
    // Summary
    // ============================================================================
    std::cout << "=== Performance Summary ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "1. Naive Function:     " << std::setw(8) << naive_func_time << " us (baseline)\n";
    std::cout << "2. Naive Op Class:     " << std::setw(8) << naive_op_time << " us (speedup: " 
              << (float)naive_func_time / naive_op_time << "x)\n";
    std::cout << "3. AVX2 Function:      " << std::setw(8) << avx2_func_time << " us (speedup: " 
              << (float)naive_func_time / avx2_func_time << "x)\n";
    std::cout << "4. AVX2 Op Class:      " << std::setw(8) << avx2_op_time << " us (speedup: " 
              << (float)naive_func_time / avx2_op_time << "x)\n";
    
    return 0;
}
