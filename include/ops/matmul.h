#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

// Naive matrix multiplication
void naiveMatMul(float *A, float *B, float *C, int M, int N, int K);

/**
 * @brief Performs a highly optimized matrix multiplication (C = A x B) for float matrices.
 *
 * Uses AVX2 and memory alignment for maximum performance. All matrices are in row-major order.
 *
 * @param A Pointer to matrix A (MxK)
 * @param B Pointer to matrix B (KxN)
 * @param C Pointer to output matrix C (MxN)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
void hyperOptimizedMatMul(float *A, float *B, float *C, int M, int N, int K);