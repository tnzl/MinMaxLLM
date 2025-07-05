#include <cpu_ops/exp_avx2.h>
#include <cpu_ops/gqa.h>
#include <stdexcept>

void GroupQueryAttention::softmax(float* arr, int size) {
    __m256 max_val = _mm256_set1_ps(-INFINITY);
    int i;
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(arr + i);
        max_val = _mm256_max_ps(max_val, vec);
    }
    alignas(32) float max_arr[8];
    _mm256_store_ps(max_arr, max_val);
    float max_scalar = std::max({max_arr[0], max_arr[1], max_arr[2], max_arr[3],
                                max_arr[4], max_arr[5], max_arr[6], max_arr[7]});
    for (; i < size; ++i) {
        if (arr[i] > max_scalar) {
            max_scalar = arr[i];
        }
    }
    __m256 max_vec = _mm256_set1_ps(max_scalar);
    __m256 sum_vec = _mm256_setzero_ps();
    float sum = 0.0f;
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(arr + i);
        vec = _mm256_sub_ps(vec, max_vec);
        vec = exp256_ps(vec);
        _mm256_storeu_ps(arr + i, vec);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    alignas(32) float sum_arr[8];
    _mm256_store_ps(sum_arr, sum_vec);
    sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
          sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
    for (; i < size; ++i) {
        arr[i] = std::exp(arr[i] - max_scalar);
        sum += arr[i];
    }
    __m256 sum_vec_inv = _mm256_set1_ps(1.0f / sum);
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(arr + i);
        vec = _mm256_mul_ps(vec, sum_vec_inv);
        _mm256_storeu_ps(arr + i, vec);
    }
    for (; i < size; ++i) {
        arr[i] /= sum;
    }
}

GroupQueryAttention::GroupQueryAttention(int num_heads, int kv_num_heads, int head_dim, float scale)
    : num_heads(num_heads), kv_num_heads(kv_num_heads), head_dim(head_dim) {
    this->scale = (scale > 0) ? scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
}

std::vector<float> GroupQueryAttention::forward(const float* query, const float* key, const float* value, int seq_len) {
    this->seq_len = seq_len;
    if (num_heads % kv_num_heads != 0) {
        throw std::invalid_argument("num_heads must be divisible by kv_num_heads");
    }
    const int group_size = num_heads / kv_num_heads;
    std::vector<float> output(num_heads * head_dim, 0.0f);
    std::vector<float> attention_scores(seq_len);
    for (int h = 0; h < num_heads; ++h) {
        const int kv_head_idx = h / group_size;
        const float* curr_query = query + h * head_dim;
        #pragma omp simd
        for (int pos = 0; pos < seq_len; ++pos) {
            const float* curr_key = key + pos * kv_num_heads * head_dim + kv_head_idx * head_dim;
            __m256 dot_vec = _mm256_setzero_ps();
            int d;
            for (d = 0; d + 8 <= head_dim; d += 8) {
                __m256 q_vec = _mm256_loadu_ps(curr_query + d);
                __m256 k_vec = _mm256_loadu_ps(curr_key + d);
                dot_vec = _mm256_fmadd_ps(q_vec, k_vec, dot_vec);
            }
            alignas(32) float dot_arr[8];
            _mm256_store_ps(dot_arr, dot_vec);
            float dot = dot_arr[0] + dot_arr[1] + dot_arr[2] + dot_arr[3] +
                        dot_arr[4] + dot_arr[5] + dot_arr[6] + dot_arr[7];
            for (; d < head_dim; ++d) {
                dot += curr_query[d] * curr_key[d];
            }
            attention_scores[pos] = dot * scale;
        }
        softmax(attention_scores.data(), seq_len);
        float* curr_output = output.data() + h * head_dim;
        int pos = 0;
        const float* first_value = value + pos * kv_num_heads * head_dim + kv_head_idx * head_dim;
        float weight = attention_scores[pos];
        #pragma omp simd
        for (int d = 0; d < head_dim; ++d) {
            curr_output[d] = weight * first_value[d];
        }
        for (pos = 1; pos < seq_len; ++pos) {
            const float* curr_value = value + pos * kv_num_heads * head_dim + kv_head_idx * head_dim;
            weight = attention_scores[pos];
            #pragma omp simd
            for (int d = 0; d < head_dim; ++d) {
                curr_output[d] += weight * curr_value[d];
            }
        }
    }
    return output;
}
