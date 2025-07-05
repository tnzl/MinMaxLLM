#pragma once
#include <immintrin.h>
#include <cstddef>

/**
 * @brief Fused Skip + RMS Layer Normalization for batch size 1 using AVX2.
 *
 * Performs layer normalization with skip connection and RMS normalization in a single fused operation.
 * All arrays are expected to be of size H.
 *
 * @param input Pointer to input array [H]
 * @param skip Pointer to skip connection array [H]
 * @param gamma Pointer to scale parameter array [H]
 * @param output Pointer to output array [H]
 * @param H Number of elements in each array
 * @param epsilon Small constant for numerical stability
 */
void SkipSimplifiedLayerNormalization_AVX2(const float* input, const float* skip, const float* gamma, float* output, size_t H, float epsilon);
