#pragma once
#include <cstddef>

// AVX2- and OpenMP-optimized elementwise addition function
void elemwise_add_avx2_omp(const float* a, const float* b, float* out, int batch_size, int hidden_size);

