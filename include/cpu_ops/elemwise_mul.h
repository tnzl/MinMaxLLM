#pragma once
#include <cstddef>

// AVX2-optimized elementwise multiplication function
void elemwise_mul_avx2(const float* a, const float* b, float* out, int batch_size, int hidden_size);