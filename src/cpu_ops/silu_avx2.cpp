#include "cpu_ops/exp_avx2.h"
#include "cpu_ops/silu_avx2.h"
#include <immintrin.h>
#include <cmath>

void silu_avx2(const float* x, float* out, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;
    
    // Handle misaligned start
    for (; i < n && (uintptr_t(x + i) % 32 != 0); ++i) {
        float xi = x[i];
        out[i] = xi / (1.0f + std::exp(-xi));
    }
    
    // Main SIMD loop
    for (; i + simd_width <= n; i += simd_width) {
        __m256 vx = _mm256_load_ps(x + i);
        
        // Fast sigmoid approximation for better performance
        __m256 vnegx = _mm256_sub_ps(_mm256_setzero_ps(), vx);
        __m256 vexp = exp256_ps(vnegx);  // Keep accurate or use approximation
        
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vsigmoid = _mm256_div_ps(vone, _mm256_add_ps(vone, vexp));
        
        // Use regular multiply (FMA not necessarily better here)
        __m256 vsilu = _mm256_mul_ps(vx, vsigmoid);
        _mm256_store_ps(out + i, vsilu);
    }
    
    // Handle remaining elements
    for (; i < n; ++i) {
        float xi = x[i];
        out[i] = xi / (1.0f + std::exp(-xi));
    }
}
