#include "cpu_ops/silu_avx2.h"
#include <immintrin.h>
#include <cmath>

// Fast vectorized exp approximation for AVX2 (for sigmoid)
// Reference: https://stackoverflow.com/a/412988/404271
static inline __m256 exp256_ps(__m256 x) {
    __m256 a = _mm256_set1_ps(12102203.0f); // 2^23 / ln(2)
    __m256i ipart = _mm256_cvttps_epi32(_mm256_fmadd_ps(x, a, _mm256_set1_ps(1065353216.0f - 0.5f)));
    __m256 fpart = _mm256_sub_ps(x, _mm256_mul_ps(_mm256_cvtepi32_ps(ipart), _mm256_set1_ps(0.69314718f)));
    __m256 poly = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(fpart, _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(fpart, fpart))));
    __m256 expipart = _mm256_castsi256_ps(ipart);
    return _mm256_mul_ps(expipart, poly);
}

void silu_avx2(const float* x, float* out, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;
    for (; i + simd_width <= n; i += simd_width) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vnegx = _mm256_sub_ps(_mm256_setzero_ps(), vx);
        __m256 vexp = exp256_ps(vnegx);
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vsigmoid = _mm256_div_ps(vone, _mm256_add_ps(vone, vexp));
        __m256 vsilu = _mm256_mul_ps(vx, vsigmoid);
        _mm256_store_ps(out + i, vsilu);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float xi = x[i];
        float sig = 1.0f / (1.0f + std::exp(-xi));
        out[i] = xi * sig;
    }
}
