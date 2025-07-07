#include <cpu_ops/SimplifiedLayerNormalization_AVX2.h>
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace cpu_ops {

void SimplifiedLayerNormalization_AVX2(const float* input, const float* scale, float* output, float epsilon, int num_heads, int head_dim) {
    constexpr int kSimdWidth = 8; // AVX2 processes 8 floats at a time
    for (int h = 0; h < num_heads; ++h) {
        const float* x = input + h * head_dim;
        float* y = output + h * head_dim;

        // 1. Compute sum and sum of squares in one pass
        __m256 sum_vec = _mm256_setzero_ps();
        __m256 sumsq_vec = _mm256_setzero_ps();
        int d = 0;
        for (; d + kSimdWidth <= head_dim; d += kSimdWidth) {
            __m256 vx = _mm256_loadu_ps(x + d);
            sum_vec = _mm256_add_ps(sum_vec, vx);
            sumsq_vec = _mm256_add_ps(sumsq_vec, _mm256_mul_ps(vx, vx));
        }
        float sum = 0.0f, sumsq = 0.0f;
        alignas(32) float tmp[kSimdWidth];
        _mm256_store_ps(tmp, sum_vec);
        for (int i = 0; i < kSimdWidth; ++i) sum += tmp[i];
        _mm256_store_ps(tmp, sumsq_vec);
        for (int i = 0; i < kSimdWidth; ++i) sumsq += tmp[i];
        for (; d < head_dim; ++d) {
            sum += x[d];
            sumsq += x[d] * x[d];
        }
        float mean = sum / head_dim;
        float variance = (sumsq - 2 * mean * sum + head_dim * mean * mean) / head_dim;
        float inv_std_var = 1.0f / std::sqrt(variance + epsilon);

        // 2. Normalize and scale
        d = 0;
        for (; d + kSimdWidth <= head_dim; d += kSimdWidth) {
            __m256 vx = _mm256_loadu_ps(x + d);
            __m256 vmean = _mm256_set1_ps(mean);
            __m256 vscale = _mm256_loadu_ps(scale + d);
            __m256 vnorm = _mm256_mul_ps(_mm256_sub_ps(vx, vmean), _mm256_set1_ps(inv_std_var));
            __m256 vy = _mm256_mul_ps(vnorm, vscale);
            _mm256_storeu_ps(y + d, vy);
        }
        for (; d < head_dim; ++d) {
            y[d] = (x[d] - mean) * inv_std_var * scale[d];
        }
    }
}

} // namespace cpu_ops
