#include "cpu_ops/SkipSimplifiedLayerNormalization_AVX2.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

// Highly optimized AVX2 implementation for Skip + RMS LayerNorm (batch size = 1)
void SkipSimplifiedLayerNormalization_AVX2(const float* input, const float* skip, const float* gamma, float* output, float* out_skip, size_t H, float epsilon) {
    constexpr int VEC = 8; // AVX2 processes 8 floats at a time
    size_t i = 0;
    __m256 sum_sq = _mm256_setzero_ps();

    // 1. Compute (input + skip), store in out_skip, and accumulate sum of squares
    for (; i + VEC <= H; i += VEC) {
        __m256 vin = _mm256_loadu_ps(input + i);
        __m256 vskip = _mm256_loadu_ps(skip + i);
        __m256 vadd = _mm256_add_ps(vin, vskip);
        _mm256_storeu_ps(out_skip + i, vadd); // Store (input+skip) in out_skip
        sum_sq = _mm256_add_ps(sum_sq, _mm256_mul_ps(vadd, vadd));
    }
    float sum_sq_scalar = 0.0f;
    alignas(32) float tmp[VEC];
    _mm256_store_ps(tmp, sum_sq);
    for (int j = 0; j < VEC; ++j) sum_sq_scalar += tmp[j];
    // Handle tail
    for (; i < H; ++i) {
        float v = input[i] + skip[i];
        out_skip[i] = v;
        sum_sq_scalar += v * v;
    }
    float mean_sq = sum_sq_scalar / H;
    float denom = 1.0f / std::sqrt(mean_sq + epsilon);

    // 2. Normalize and scale by gamma, store in output
    i = 0;
    __m256 vdenom = _mm256_set1_ps(denom);
    for (; i + VEC <= H; i += VEC) {
        __m256 vadd = _mm256_loadu_ps(out_skip + i); // (input+skip)
        __m256 vgamma = _mm256_loadu_ps(gamma + i);
        __m256 vnorm = _mm256_mul_ps(_mm256_mul_ps(vadd, vdenom), vgamma);
        _mm256_storeu_ps(output + i, vnorm);
    }
    for (; i < H; ++i) {
        output[i] = gamma[i] * out_skip[i] * denom;
    }
}
