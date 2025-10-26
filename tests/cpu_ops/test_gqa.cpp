#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cpu_ops/gqa.h>
#include "../test_utils.cpp"
#include <chrono>
#include <algorithm>

void naive_gqa_forward(
    const float *query, // [A, h]
    const float *key,   // [G, N_max, h]
    const float *value, // [G, N_max, h]
    float *output,      // [A, h]
    int N,              // actual sequence length
    int N_max,          // max sequence length
    int G,              // number of KV groups
    int A,              // number of attention heads
    int h,              // head dimension
    float scale         // scaling factor (typically 1/sqrt(h))
)
{
    // Calculate how many attention heads per KV group
    int heads_per_group = A / G;

    // Temporary buffers for attention scores and weights
    std::vector<float> attn_scores(N);
    std::vector<float> attn_weights(N);

    // Iterate over each attention head
    for (int a = 0; a < A; a++)
    {
        // Determine which KV group this attention head belongs to
        int g = a / heads_per_group;

        // Pointer to current query head: [h]
        const float *q = query + a * h;

        // Pointer to KV group: [N_max, h]
        const float *k_group = key + g * N_max * h;
        const float *v_group = value + g * N_max * h;

        // Step 1: Compute attention scores Q @ K^T
        // scores[n] = sum_d(q[d] * k[n, d]) * scale
        for (int n = 0; n < N; n++)
        {
            float score = 0.0f;
            const float *k_n = k_group + n * h;

            for (int d = 0; d < h; d++)
            {
                score += q[d] * k_n[d];
            }
            attn_scores[n] = score * scale;
        }

        // Step 2: Softmax over sequence dimension
        // Find max for numerical stability
        float max_score = attn_scores[0];
        for (int n = 1; n < N; n++)
        {
            max_score = std::max(max_score, attn_scores[n]);
        }

        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int n = 0; n < N; n++)
        {
            attn_weights[n] = std::exp(attn_scores[n] - max_score);
            sum_exp += attn_weights[n];
        }

        // Normalize
        for (int n = 0; n < N; n++)
        {
            attn_weights[n] /= sum_exp;
        }

        // Step 3: Compute weighted sum of values
        // output[a, d] = sum_n(attn_weights[n] * v[n, d])
        float *out = output + a * h;

        for (int d = 0; d < h; d++)
        {
            float sum = 0.0f;
            for (int n = 0; n < N; n++)
            {
                const float *v_n = v_group + n * h;
                sum += attn_weights[n] * v_n[d];
            }
            out[d] = sum;
        }
    }
}

int main()
{
    // Typical LLM GQA sizes
    const int num_heads = 32;
    const int kv_num_heads = 8;
    const int head_dim = 128;
    const int seq_len = 1048;
    const int max_seq_len = 1048;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> query(num_heads * head_dim);
    std::vector<float> key(kv_num_heads * max_seq_len * head_dim);
    std::vector<float> value(kv_num_heads * max_seq_len * head_dim);
    std::vector<float> output_ref(num_heads * head_dim);
    std::vector<float> output(num_heads * head_dim);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : query)
        x = dist(gen);
    for (auto &x : key)
        x = dist(gen);
    for (auto &x : value)
        x = dist(gen);

    // Naive reference timing
    auto start = std::chrono::high_resolution_clock::now();
    naive_gqa_forward(query.data(), key.data(), value.data(), output_ref.data(), seq_len, max_seq_len, kv_num_heads, num_heads, head_dim, scale);
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Optimized GQA timing
    start = std::chrono::high_resolution_clock::now();
    optimized_gqa_forward(query.data(), key.data(), value.data(), output.data(), num_heads, kv_num_heads, head_dim, seq_len, max_seq_len, scale);
    end = std::chrono::high_resolution_clock::now();
    auto avx_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // // Validate
    // bool pass = validateResults(output.data(), output_ref.data(), num_heads, head_dim, 0.001);
    // Always print error analysis
    printErrorAnalysis(output.data(), output_ref.data(), num_heads, head_dim);
    std::cout << "Naive GQA Latency: " << naive_time << " us\n";
    std::cout << "AVX GQA Latency: " << avx_time << " us\n";
    std::cout << "Speedup: " << (float)naive_time / (float)avx_time << "x\n";
    return 0;
}
