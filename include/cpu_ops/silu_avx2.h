#pragma once
#include <cstddef>

// Computes out[i] = sigmoid(x[i]) * x[i] for i in [0, n)
// Uses AVX2 for maximum performance. Input and output arrays must be 32-byte aligned.
void silu_avx2(const float* x, float* out, size_t n);
