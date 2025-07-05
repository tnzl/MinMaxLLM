#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cpu_ops/gqa.h>
#include "../test_utils.cpp"

// Naive GQA implementation for validation
std::vector<float> naive_gqa_forward(const float* query, const float* key, const float* value, int num_heads, int kv_num_heads, int head_dim, int seq_len, float scale) {
    int group_size = num_heads / kv_num_heads;
    std::vector<float> output(num_heads * head_dim, 0.0f);
    std::vector<float> attention_scores(seq_len);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head_idx = h / group_size;
        const float* curr_query = query + h * head_dim;
        // Compute attention scores
        for (int pos = 0; pos < seq_len; ++pos) {
            const float* curr_key = key + pos * kv_num_heads * head_dim + kv_head_idx * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += curr_query[d] * curr_key[d];
            }
            attention_scores[pos] = dot * scale;
        }
        // Softmax
        float max_score = *std::max_element(attention_scores.begin(), attention_scores.end());
        float sum = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            attention_scores[pos] = std::exp(attention_scores[pos] - max_score);
            sum += attention_scores[pos];
        }
        for (int pos = 0; pos < seq_len; ++pos) {
            attention_scores[pos] /= sum;
        }
        // Weighted sum of values
        float* curr_output = output.data() + h * head_dim;
        for (int pos = 0; pos < seq_len; ++pos) {
            const float* curr_value = value + pos * kv_num_heads * head_dim + kv_head_idx * head_dim;
            float weight = attention_scores[pos];
            for (int d = 0; d < head_dim; ++d) {
                curr_output[d] += weight * curr_value[d];
            }
        }
    }
    return output;
}

int main() {
    // Typical LLM GQA sizes
    const int num_heads = 16;
    const int kv_num_heads = 4;
    const int head_dim = 32;
    const int seq_len = 128;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> query(num_heads * head_dim);
    std::vector<float> key(seq_len * kv_num_heads * head_dim);
    std::vector<float> value(seq_len * kv_num_heads * head_dim);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : query) x = dist(gen);
    for (auto& x : key) x = dist(gen);
    for (auto& x : value) x = dist(gen);

    // Naive reference
    auto ref = naive_gqa_forward(query.data(), key.data(), value.data(), num_heads, kv_num_heads, head_dim, seq_len, scale);

    // Optimized GQA
    GroupQueryAttention gqa(num_heads, kv_num_heads, head_dim, scale);
    auto out = gqa.forward(query.data(), key.data(), value.data(), seq_len);

    // Validate
    if (!validateResults(ref.data(), out.data(), num_heads, head_dim, 0.001)) {
        std::cerr << "Error: GQA results don't match!\n";
        printErrorAnalysis(ref.data(), out.data(), num_heads, head_dim);
        return 1;
    }
    std::cout << "GQA correctness test passed!\n";
    printErrorAnalysis(ref.data(), out.data(), num_heads, head_dim);
    return 0;
}
