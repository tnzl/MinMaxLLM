#include <immintrin.h>
#include <omp.h>
#include <cstdio>

void linear_naive(const float *input, const float *weight, int M, int K, int N, float *output);
void linear_avx2_omp(const float *input, const float *weight, int M, int K, int N, float *output);