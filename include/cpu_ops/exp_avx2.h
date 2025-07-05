#pragma once
#include <immintrin.h>

// Fast exponential approximation for AVX2 (__m256)
inline __m256 exp256_ps(__m256 x) {
    const __m256 a0 = _mm256_set1_ps(1.0f);
    const __m256 a1 = _mm256_set1_ps(1.0f);
    const __m256 a2 = _mm256_set1_ps(0.5f);
    const __m256 a3 = _mm256_set1_ps(0.166666667f);
    const __m256 a4 = _mm256_set1_ps(0.0416666667f);
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 x4 = _mm256_mul_ps(x3, x);
    __m256 res = _mm256_add_ps(a0, _mm256_mul_ps(a1, x));
    res = _mm256_add_ps(res, _mm256_mul_ps(a2, x2));
    res = _mm256_add_ps(res, _mm256_mul_ps(a3, x3));
    res = _mm256_add_ps(res, _mm256_mul_ps(a4, x4));
    return res;
}
