#pragma once
#include <immintrin.h>
#include <cstddef>

namespace cpu_ops {
// Applies simplified layer normalization over each head (row) of the input.
// input: [num_heads, head_dim]
// scale: [head_dim]
// output: [num_heads, head_dim]
// epsilon: scalar
void SimplifiedLayerNormalization_AVX2(const float* input, const float* scale, float* output, float epsilon, int num_heads, int head_dim);
}
