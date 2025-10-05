#include <cpu_ops/elemwise_mul.h>
#include <immintrin.h>

void elemwise_mul_avx2(const float* a, const float* b, float* out, int batch_size, int hidden_size) {
    int total = batch_size * hidden_size;
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }
    for (; i < total; ++i) {
        out[i] = a[i] * b[i];
    }
}
