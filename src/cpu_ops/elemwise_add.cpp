#include <cpu_ops/elemwise_add.h>
#include <immintrin.h>
#include <omp.h>

void elemwise_add_avx2_omp(const float* a, const float* b, float* out, int batch_size, int hidden_size) {
    int total = batch_size * hidden_size;
    int simd_length = total / 8;
    int simd_end = simd_length * 8;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int block = 0; block < simd_length; ++block) {
            int idx = block * 8;
            __m256 va = _mm256_loadu_ps(a + idx);
            __m256 vb = _mm256_loadu_ps(b + idx);
            __m256 vout = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out + idx, vout);
        }

#pragma omp for schedule(static)
        for (int idx = simd_end; idx < total; ++idx) {
            out[idx] = a[idx] + b[idx];
        }
    }
}

