#include <cpu_ops/gqa.h>
#include <cpu_ops/softmax_avx2.h>
#include <tensor/tensor.h>
#include <tensor/kvcache.h>

#include <vector>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <cassert>
#include <stdexcept>
#include <algorithm>

namespace
{
struct GQADims
{
    int A;      // number of attention heads
    int G;      // number of KV groups
    int h;      // head dimension
    int N_max;  // max sequence length
};

GQADims compute_gqa_dims(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N)
{
    const auto &query_shape = query.shape();
    assert(query_shape.size() == 2 && "GQAOp expects query tensor of rank 2 [A, h].");
    int A = static_cast<int>(query_shape[0]);
    int h = static_cast<int>(query_shape[1]);

    const auto &key_shape = key.shape();
    assert(key_shape.size() == 3 && "GQAOp expects key tensor of rank 3 [G, N_max, h].");
    int G = static_cast<int>(key_shape[0]);
    int N_max = static_cast<int>(key_shape[1]);
    assert(static_cast<int>(key_shape[2]) == h && "GQAOp key head dimension must match query head dimension.");

    const auto &value_shape = value.shape();
    assert(value_shape.size() == 3 && "GQAOp expects value tensor of rank 3 [G, N_max, h].");
    assert(static_cast<int>(value_shape[0]) == G && "GQAOp value KV groups must match key KV groups.");
    assert(static_cast<int>(value_shape[1]) == N_max && "GQAOp value max sequence length must match key max sequence length.");
    assert(static_cast<int>(value_shape[2]) == h && "GQAOp value head dimension must match query head dimension.");

    const auto &output_shape = output.shape();
    assert(output_shape.size() == 2 && "GQAOp expects output tensor of rank 2 [A, h].");
    assert(static_cast<int>(output_shape[0]) == A && "GQAOp output attention heads must match query attention heads.");
    assert(static_cast<int>(output_shape[1]) == h && "GQAOp output head dimension must match query head dimension.");

    assert(N > 0 && N <= N_max && "GQAOp actual sequence length must be positive and <= max sequence length.");
    assert(A > 0 && G > 0 && h > 0 && N_max > 0 && "GQAOp dimensions must be positive.");
    assert(A % G == 0 && "GQAOp number of attention heads must be divisible by number of KV groups.");

    return {A, G, h, N_max};
}
} // namespace

void gqa_naive(
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
            max_score = (std::max)(max_score, attn_scores[n]);
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

GQAOp::GQAOp(OpBackend backend, float scale) 
    : BaseOp(backend), scale_(scale), kvcache_(nullptr), layer_idx_(0), num_heads_(0), num_groups_(0), head_dim_(0)
{
}

GQAOp::GQAOp(KVCache *kvcache, size_t layer_idx, size_t num_heads, size_t num_groups, size_t head_dim, OpBackend backend, float scale)
    : BaseOp(backend), scale_(scale), kvcache_(kvcache), layer_idx_(layer_idx), num_heads_(num_heads), num_groups_(num_groups), head_dim_(head_dim)
{
}

GQAOp::~GQAOp() = default;

void GQAOp::prepare()
{
    // No weights to prefetch for GQA
}

void GQAOp::run(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N)
{
    if (scale_ == 0.0f)
    {
        throw std::runtime_error("GQAOp::run called without a scale value. Either set scale in constructor or use run() with scale parameter.");
    }

    run_internal(query, key, value, output, N, scale_);
}

void GQAOp::run(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N, float scale)
{
    run_internal(query, key, value, output, N, scale);
}

void GQAOp::run(Tensor &query, Tensor &output, int N)
{
    if (!kvcache_)
    {
        throw std::runtime_error("GQAOp::run called with KV cache mode but no KV cache was provided in constructor.");
    }
    if (scale_ == 0.0f)
    {
        throw std::runtime_error("GQAOp::run called without a scale value. Either set scale in constructor or use run() with scale parameter.");
    }

    // Create tensors from KV cache
    Tensor key_tensor(static_cast<void *>(const_cast<float *>(kvcache_->get_key_memory_ptr(layer_idx_))), 
                      {num_groups_, kvcache_->get_max_sequence_length(), head_dim_}, 
                      DataType::F32, false, false);
    Tensor value_tensor(static_cast<void *>(const_cast<float *>(kvcache_->get_value_memory_ptr(layer_idx_))), 
                        {num_groups_, kvcache_->get_max_sequence_length(), head_dim_}, 
                        DataType::F32, false, false);

    run_internal(query, key_tensor, value_tensor, output, N, scale_);
}

void GQAOp::run_internal(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N, float scale)
{
    GQAImplType selected = resolve_impl(query, key, value);
    switch (selected)
    {
    case GQAImplType::NAIVE:
        naive_impl(query, key, value, output, N, scale);
        break;
    case GQAImplType::AVX2:
        avx2_impl(query, key, value, output, N, scale);
        break;
    default:
        throw std::runtime_error("No implementation registered for selected GQAImplType.");
    }
}

GQAImplType GQAOp::resolve_impl(Tensor &, Tensor &, Tensor &) const
{
    switch (backend_)
    {
    case OpBackend::NAIVE:
        return GQAImplType::NAIVE;
    case OpBackend::AVX2:
        return GQAImplType::AVX2;
    default:
        return GQAImplType::NAIVE;
    }
}

void GQAOp::naive_impl(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N, float scale)
{
    assert(query.dtype() == DataType::F32 && "GQAOp::naive_impl supports only float32 query tensors.");
    assert(key.dtype() == DataType::F32 && "GQAOp::naive_impl supports only float32 key tensors.");
    assert(value.dtype() == DataType::F32 && "GQAOp::naive_impl supports only float32 value tensors.");
    assert(output.dtype() == DataType::F32 && "GQAOp::naive_impl supports only float32 output tensors.");

    const GQADims dims = compute_gqa_dims(query, key, value, output, N);
    gqa_naive(
        query.data<float>(),
        key.data<float>(),
        value.data<float>(),
        output.data<float>(),
        N,
        dims.N_max,
        dims.G,
        dims.A,
        dims.h,
        scale);
}

void GQAOp::avx2_impl(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N, float scale)
{
    assert(query.dtype() == DataType::F32 && "GQAOp::avx2_impl supports only float32 query tensors.");
    assert(key.dtype() == DataType::F32 && "GQAOp::avx2_impl supports only float32 key tensors.");
    assert(value.dtype() == DataType::F32 && "GQAOp::avx2_impl supports only float32 value tensors.");
    assert(output.dtype() == DataType::F32 && "GQAOp::avx2_impl supports only float32 output tensors.");

    const GQADims dims = compute_gqa_dims(query, key, value, output, N);
    optimized_gqa_forward(
        query.data<float>(),
        key.data<float>(),
        value.data<float>(),
        output.data<float>(),
        dims.A,
        dims.G,
        dims.h,
        N,
        dims.N_max,
        scale);
}