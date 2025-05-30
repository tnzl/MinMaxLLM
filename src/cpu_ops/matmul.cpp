#include <cpu_ops/matmul.h>

// Hyperoptimized for typical LLM shapes (e.g., batch_size x seq_len x hidden_dim)
// Assumes:
// 1. M is typically small (batch_size * seq_len)
// 2. K is large (hidden_dim, typically 4096+)
// 3. N is medium-sized (hidden_dim or intermediate size)
void hyperOptimizedMatMul(float* A, float* B, float* C, int M, int N, int K) {
    constexpr int BLOCK_SIZE = 64; // Optimal for L1 cache
    constexpr int UNROLL_FACTOR = 4;
    constexpr int VEC_SIZE = 8; // AVX2

    // Zero output matrix
    memset(C, 0, M * N * sizeof(float));

    for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
        int k_end = kk + BLOCK_SIZE < K ? kk + BLOCK_SIZE : K;
        
        for (int mm = 0; mm < M; mm += BLOCK_SIZE) {
            int m_end = mm + BLOCK_SIZE < M ? mm + BLOCK_SIZE : M;
            
            for (int nn = 0; nn < N; nn += BLOCK_SIZE) {
                int n_end = nn + BLOCK_SIZE < N ? nn + BLOCK_SIZE : N;
                
                // Block processing
                for (int m = mm; m < m_end; ++m) {
                    for (int n = nn; n < n_end; n += VEC_SIZE * UNROLL_FACTOR) {
                        // Prefetch next B block
                        if (kk + BLOCK_SIZE < K) {
                            _mm_prefetch((const char*)&B[(kk + BLOCK_SIZE) * N + n], _MM_HINT_T0);
                        }
                        
                        // Initialize accumulators
                        __m256 c[UNROLL_FACTOR];
                        for (int u = 0; u < UNROLL_FACTOR; ++u) {
                            c[u] = _mm256_setzero_ps();
                        }
                        
                        // Inner kernel
                        for (int k = kk; k < k_end; ++k) {
                            __m256 a = _mm256_set1_ps(A[m * K + k]);
                            
                            for (int u = 0; u < UNROLL_FACTOR; ++u) {
                                int current_n = n + u * VEC_SIZE;
                                if (current_n >= n_end) break;
                                
                                __m256 b = _mm256_loadu_ps(&B[k * N + current_n]);
                                c[u] = _mm256_fmadd_ps(a, b, c[u]);
                            }
                        }
                        
                        // Store results
                        for (int u = 0; u < UNROLL_FACTOR; ++u) {
                            int current_n = n + u * VEC_SIZE;
                            if (current_n >= n_end) break;
                            
                            if (current_n + VEC_SIZE <= n_end) {
                                __m256 prev = _mm256_loadu_ps(&C[m * N + current_n]);
                                _mm256_storeu_ps(&C[m * N + current_n], _mm256_add_ps(prev, c[u]));
                            } else {
                                // Handle tail elements
                                float tmp[VEC_SIZE];
                                _mm256_storeu_ps(tmp, c[u]);
                                for (int v = 0; v < n_end - current_n; ++v) {
                                    C[m * N + current_n + v] += tmp[v];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}