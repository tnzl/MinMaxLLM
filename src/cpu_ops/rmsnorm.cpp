#include <cpu_ops/rmsnorm.h>
#include <immintrin.h>
#include <cmath>

void rmsnorm_avx2(const float* input, const float* weight, float* output, int batch_size, int hidden_size, float eps) {
    for (int b = 0; b < batch_size; ++b) {
        const float* in = input + b * hidden_size;
        float mean_sq = 0.0f;
        int d = 0;
        __m256 sum_vec = _mm256_setzero_ps();
        for (; d + 8 <= hidden_size; d += 8) {
            __m256 x = _mm256_loadu_ps(in + d);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x, x));
        }
        alignas(32) float sum_arr[8];
        _mm256_store_ps(sum_arr, sum_vec);
        for (int i = 0; i < 8; ++i) mean_sq += sum_arr[i];
        for (; d < hidden_size; ++d) mean_sq += in[d] * in[d];
        mean_sq /= hidden_size;
        float denom = 1.0f / std::sqrt(mean_sq + eps);
        d = 0;
        for (; d + 8 <= hidden_size; d += 8) {
            __m256 x = _mm256_loadu_ps(in + d);
            __m256 w = _mm256_loadu_ps(weight + d);
            __m256 norm = _mm256_mul_ps(x, _mm256_set1_ps(denom));
            __m256 out = _mm256_mul_ps(w, norm);
            _mm256_storeu_ps(output + b * hidden_size + d, out);
        }
        for (; d < hidden_size; ++d) {
            output[b * hidden_size + d] = weight[d] * in[d] * denom;
        }
    }
}
