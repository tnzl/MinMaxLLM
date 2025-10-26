#include <cpu_ops/gqa.h>
#include <cpu_ops/softmax_avx2.h>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

#include <vector>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

// Helper function for horizontal sum of AVX2 register
inline float horizontal_sum_avx(__m256 vec)
{
    __m128 low = _mm256_castps256_ps128(vec);
    __m128 high = _mm256_extractf128_ps(vec, 1);
    low = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(low);
    __m128 sums = _mm_add_ps(low, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void optimized_gqa_forward(
    const float *query, // [A, h] - single token query for all attention heads
    const float *key,   // [G, N_max, h] - keys for all KV groups and positions
    const float *value, // [G, N_max, h] - values for all KV groups and positions
    float *output,      // [A, h] - output for all attention heads
    int A,              // number of attention heads
    int G,              // number of KV groups
    int h,              // head dimension
    int N,              // actual sequence length (N <= N_max)
    int N_max,          // max sequence length
    float scale         // scaling factor
)
{
    // Calculate query heads per KV group
    int heads_per_group = A / G;

    // Precompute KV group mapping for each attention head - FIXED BUG
    std::vector<int> head_to_group(A);
    for (int a = 0; a < A; a++)
    {
        head_to_group[a] = a / heads_per_group; // CORRECT: each KV group serves multiple query heads
    }

// Parallelize over attention heads
#pragma omp parallel for schedule(static)
    for (int a = 0; a < A; a++)
    {
        int g = head_to_group[a];

        // Get pointers to current head's data
        const float *curr_query = query + a * h;
        const float *curr_key_base = key + g * N_max * h;
        const float *curr_value_base = value + g * N_max * h;
        float *curr_output = output + a * h;

        // Compute attention scores using AVX2
        std::vector<float> attention_scores(N);

        // Phase 1: Compute Q•K^T dot products
        for (int pos = 0; pos < N; pos++)
        {
            const float *curr_key = curr_key_base + pos * h;

            __m256 dot_sum = _mm256_setzero_ps();
            int dim = 0;

            // Process 8 elements at a time with AVX2
            for (; dim <= h - 8; dim += 8)
            {
                __m256 q_vec = _mm256_loadu_ps(curr_query + dim);
                __m256 k_vec = _mm256_loadu_ps(curr_key + dim);
                __m256 mul = _mm256_mul_ps(q_vec, k_vec);
                dot_sum = _mm256_add_ps(dot_sum, mul);
            }

            // Horizontal sum of AVX2 register
            float dot_product = horizontal_sum_avx(dot_sum);

            // Handle remaining elements
            for (; dim < h; dim++)
            {
                dot_product += curr_query[dim] * curr_key[dim];
            }

            attention_scores[pos] = dot_product * scale;
        }

        // Phase 2: Apply softmax (using your optimized version)
        softmax_avx2(attention_scores.data(), N);

        // Phase 3: Compute weighted sum of values
        // Initialize output to zero
        for (int dim = 0; dim < h; dim++)
        {
            curr_output[dim] = 0.0f;
        }

        // Accumulate weighted values
        for (int pos = 0; pos < N; pos++)
        {
            const float *curr_value = curr_value_base + pos * h;
            float weight = attention_scores[pos];
            __m256 weight_vec = _mm256_set1_ps(weight);

            int dim = 0;
            // Process 8 elements at a time with AVX2
            for (; dim <= h - 8; dim += 8)
            {
                __m256 out_vec = _mm256_loadu_ps(curr_output + dim);
                __m256 val_vec = _mm256_loadu_ps(curr_value + dim);
                __m256 weighted = _mm256_mul_ps(weight_vec, val_vec);
                out_vec = _mm256_add_ps(out_vec, weighted);
                _mm256_storeu_ps(curr_output + dim, out_vec);
            }

            // Handle remaining elements
            for (; dim < h; dim++)
            {
                curr_output[dim] += weight * curr_value[dim];
            }
        }
    }
}