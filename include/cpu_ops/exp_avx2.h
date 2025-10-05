#pragma once
#include <immintrin.h>

/**
 * @brief Fast exponential approximation for AVX2 (__m256)
 *
 * Approximates exp(x) for each element in the AVX2 vector using a 4th order Taylor expansion.
 * Input values are clamped to the range [-88.3762626647949, 88.3762626647949] to avoid overflow.
 *
 * @param x Input AVX2 vector (__m256)
 * @return __m256 Vector with exp(x) applied elementwise
 */
#ifndef M_LN2
#define M_LN2 0.69314718055994530941723212145818f
#endif

inline __m256 exp256_ps(__m256 x) {
    // Clamp input to prevent overflow
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    
    // Range reduction: x = m * ln2 + r, where |r| <= ln(2)/2
    const __m256 ln2 = _mm256_set1_ps(M_LN2);
    const __m256 inv_ln2 = _mm256_set1_ps(1.44269504088896340736f);
    
    // m = round(x / ln2)
    __m256 m = _mm256_floor_ps(_mm256_fmadd_ps(x, inv_ln2, _mm256_set1_ps(0.5f)));
    
    // r = x - m * ln2
    __m256 r = _mm256_fnmadd_ps(m, ln2, x);
    
    // Taylor approximation for exp(r) on reduced range
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.166666666666666f);  // 1/6
    const __m256 c4 = _mm256_set1_ps(0.041666666666666f);  // 1/24
    const __m256 c5 = _mm256_set1_ps(0.008333333333333f);  // 1/120
    
    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 r3 = _mm256_mul_ps(r2, r);
    __m256 r4 = _mm256_mul_ps(r3, r);
    
    __m256 result = c1;
    result = _mm256_fmadd_ps(r, result, c1);      // 1 + r
    result = _mm256_fmadd_ps(r2, c2, result);     // + r²/2
    result = _mm256_fmadd_ps(r3, c3, result);     // + r³/6
    result = _mm256_fmadd_ps(r4, c4, result);     // + r⁴/24
    
    // Scale by 2^m
    __m256i exponent = _mm256_cvtps_epi32(m);
    exponent = _mm256_add_epi32(exponent, _mm256_set1_epi32(127));
    exponent = _mm256_slli_epi32(exponent, 23);
    __m256 pow2 = _mm256_castsi256_ps(exponent);
    
    return _mm256_mul_ps(result, pow2);
}
