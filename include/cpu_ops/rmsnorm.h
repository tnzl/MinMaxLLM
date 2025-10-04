#pragma once
#include <cstddef>

// AVX2-optimized RMSNorm function
void rmsnorm_avx2(const float* input, const float* weight, float* output, int batch_size, int hidden_size, float eps);
