#pragma once
#include <immintrin.h>
#include <cstddef>

// Fused Skip + RMS LayerNorm (batch size = 1, AVX2)
// input, skip: [H]
// gamma: [H]
// output: [H]
// epsilon: scalar
void SkipSimplifiedLayerNormalization_AVX2(const float* input, const float* skip, const float* gamma, float* output, size_t H, float epsilon);
