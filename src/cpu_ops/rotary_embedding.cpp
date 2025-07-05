#include <cpu_ops/rotary_embedding.h>
#include <immintrin.h> // AVX2 intrinsics
#include <cmath>
#include <cstring>

RotaryEmbeddingAVX2::RotaryEmbeddingAVX2(const float* sin_cache,
                                       const float* cos_cache,
                                       int max_positions,
                                       int dim,
                                       int rotary_dim)
    : rotary_dim_(rotary_dim == 0 ? dim : rotary_dim)
{
    cache_ = std::make_unique<RotationCache>();
    const int rot_dim_half = rotary_dim_ / 2;
    
    // Allocate with padding for SIMD alignment
    cache_->sin.resize(max_positions * rot_dim_half + 8);
    cache_->cos.resize(max_positions * rot_dim_half + 8);
    
    std::memcpy(cache_->sin.data(), sin_cache, max_positions * rot_dim_half * sizeof(float));
    std::memcpy(cache_->cos.data(), cos_cache, max_positions * rot_dim_half * sizeof(float));
    
    cache_->max_positions = max_positions;
    cache_->rotary_dim = rotary_dim_;
}

void RotaryEmbeddingAVX2::rotate(float* embeddings,
                               int num_heads,
                               int head_size,
                               int position_id) const
{
    const int rot_dim = rotary_dim_;
    const int rot_dim_half = rot_dim / 2;
    const float* sin_ptr = &cache_->sin[position_id * rot_dim_half];
    const float* cos_ptr = &cache_->cos[position_id * rot_dim_half];

    #pragma omp parallel for
    for (int h = 0; h < num_heads; ++h) {
        float* head = embeddings + h * head_size;
        
        // AVX2 processing (8 floats per register)
        int i = 0;
        for (; i + 8 <= rot_dim_half; i += 8) {
            __m256 x1 = _mm256_loadu_ps(head + i);
            __m256 x2 = _mm256_loadu_ps(head + i + rot_dim_half);
            __m256 sin = _mm256_loadu_ps(sin_ptr + i);
            __m256 cos = _mm256_loadu_ps(cos_ptr + i);
            
            __m256 x1_new = _mm256_sub_ps(
                _mm256_mul_ps(x1, cos),
                _mm256_mul_ps(x2, sin)
            );
            __m256 x2_new = _mm256_add_ps(
                _mm256_mul_ps(x1, sin),
                _mm256_mul_ps(x2, cos)
            );
            
            _mm256_storeu_ps(head + i, x1_new);
            _mm256_storeu_ps(head + i + rot_dim_half, x2_new);
        }
        
        // Remainder processing
        for (; i < rot_dim_half; ++i) {
            float x1 = head[i];
            float x2 = head[i + rot_dim_half];
            float s = sin_ptr[i];
            float c = cos_ptr[i];
            head[i] = x1 * c - x2 * s;
            head[i + rot_dim_half] = x1 * s + x2 * c;
        }
    }
}

void RotaryEmbeddingAVX2::precompute(float* sin_cache,
                                   float* cos_cache,
                                   int max_positions,
                                   int dim,
                                   float base)
{
    const int rot_dim_half = dim / 2;
    for (int pos = 0; pos < max_positions; ++pos) {
        for (int i = 0; i < rot_dim_half; ++i) {
            float inv_freq = std::pow(base, -2.0f * i / dim);
            float angle = pos * inv_freq;
            sin_cache[pos * rot_dim_half + i] = std::sin(angle);
            cos_cache[pos * rot_dim_half + i] = std::cos(angle);
        }
    }
}