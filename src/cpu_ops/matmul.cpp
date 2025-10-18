#include <immintrin.h>
#include <omp.h>
#include <string.h>

// Naive matrix multiplication
void naiveMatMul(float *A, float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Block sizes tuned for L1/L2 cache
#define BLOCK_M 128
#define BLOCK_N 256
#define BLOCK_K 512
#define MICRO_M 6
#define MICRO_N 16

// Micro-kernel: computes MICRO_M x MICRO_N block
static inline void microKernel(float *A, float *B, float *C, int K, int lda, int ldb, int ldc)
{
    __m256 c[MICRO_M][2];

    // Initialize accumulators
    for (int i = 0; i < MICRO_M; i++)
    {
        c[i][0] = _mm256_setzero_ps();
        c[i][1] = _mm256_setzero_ps();
    }

    // Main computation loop
    for (int k = 0; k < K; k++)
    {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b1 = _mm256_loadu_ps(&B[k * ldb + 8]);

        for (int i = 0; i < MICRO_M; i++)
        {
            __m256 a = _mm256_broadcast_ss(&A[i * lda + k]);
            c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
        }
    }

    // Store results
    for (int i = 0; i < MICRO_M; i++)
    {
        _mm256_storeu_ps(&C[i * ldc], _mm256_add_ps(c[i][0], _mm256_loadu_ps(&C[i * ldc])));
        _mm256_storeu_ps(&C[i * ldc + 8], _mm256_add_ps(c[i][1], _mm256_loadu_ps(&C[i * ldc + 8])));
    }
}

void hyperOptimizedMatMul(float *A, float *B, float *C, int M, int N, int K)
{
    // Initialize output matrix
    memset(C, 0, M * N * sizeof(float));

// Outer parallelization over M blocks
#pragma omp parallel for schedule(dynamic)
    for (int mm = 0; mm < M; mm += BLOCK_M)
    {
        int m_end = (mm + BLOCK_M < M) ? mm + BLOCK_M : M;

        for (int kk = 0; kk < K; kk += BLOCK_K)
        {
            int k_end = (kk + BLOCK_K < K) ? kk + BLOCK_K : K;
            int k_block = k_end - kk;

            for (int nn = 0; nn < N; nn += BLOCK_N)
            {
                int n_end = (nn + BLOCK_N < N) ? nn + BLOCK_N : N;

                // Process MICRO_M x MICRO_N micro-kernels
                for (int m = mm; m < m_end; m += MICRO_M)
                {
                    int m_micro = (m + MICRO_M < m_end) ? MICRO_M : m_end - m;

                    for (int n = nn; n < n_end; n += MICRO_N)
                    {
                        int n_micro = (n + MICRO_N < n_end) ? MICRO_N : n_end - n;

                        if (m_micro == MICRO_M && n_micro == MICRO_N)
                        {
                            // Fast path: full micro-kernel
                            microKernel(&A[m * K + kk], &B[kk * N + n],
                                        &C[m * N + n], k_block, K, N, N);
                        }
                        else
                        {
                            // Boundary case: scalar fallback
                            for (int i = 0; i < m_micro; i++)
                            {
                                for (int k = 0; k < k_block; k++)
                                {
                                    float a_val = A[(m + i) * K + kk + k];

                                    // Vectorized inner loop
                                    int j = 0;
                                    for (; j + 8 <= n_micro; j += 8)
                                    {
                                        __m256 b_vec = _mm256_loadu_ps(&B[(kk + k) * N + n + j]);
                                        __m256 c_vec = _mm256_loadu_ps(&C[(m + i) * N + n + j]);
                                        __m256 a_vec = _mm256_set1_ps(a_val);
                                        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                                        _mm256_storeu_ps(&C[(m + i) * N + n + j], c_vec);
                                    }

                                    // Scalar cleanup
                                    for (; j < n_micro; j++)
                                    {
                                        C[(m + i) * N + n + j] += a_val * B[(kk + k) * N + n + j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}