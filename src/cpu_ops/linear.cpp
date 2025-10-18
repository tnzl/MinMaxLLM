#include <cpu_ops/linear.h>

// Naive reference version
void linear_naive(const float *input, const float *weight, int M, int K, int N, float *output)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += input[i * K + k] * weight[j * K + k];
            }
            output[i * N + j] = sum;
        }
    }
}

void linear_avx2_omp(const float *input, const float *weight, int M, int K, int N, float *output)
{
#pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        const float *in_row = input + i * K;
        for (int j = 0; j < N; ++j)
        {
            const float *w_row = weight + j * K;

            __m256 vsum = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8)
            {
                __m256 va = _mm256_loadu_ps(in_row + k);
                __m256 vb = _mm256_loadu_ps(w_row + k);
                vsum = _mm256_fmadd_ps(va, vb, vsum); // fused multiply-add
            }

            // horizontal add
            __m128 low = _mm256_castps256_ps128(vsum);
            __m128 high = _mm256_extractf128_ps(vsum, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);

            // Remainder
            for (; k < K; ++k)
                sum += in_row[k] * w_row[k];

            output[i * N + j] = sum;
        }
    }
}