#pragma once
#include <cstddef>

/**
 * @brief Computes the SiLU (Sigmoid Linear Unit) activation using AVX2.
 *
 * Calculates out[i] = sigmoid(x[i]) * x[i] for i in [0, n), using AVX2 for maximum performance.
 * Input and output arrays must be 32-byte aligned.
 *
 * @param x Pointer to input array
 * @param out Pointer to output array
 * @param n Number of elements
 */
void silu_avx2(const float* x, float* out, size_t n);

/**
 * @brief Naive reference implementation of SiLU (Sigmoid Linear Unit) activation.
 *
 * Calculates out[i] = sigmoid(x[i]) * x[i] for i in [0, n).
 * This is a simple reference implementation for testing and comparison.
 *
 * @param x Pointer to input array
 * @param out Pointer to output array
 * @param n Number of elements
 */
void silu(const float* x, float* out, size_t n);

/**
 * @brief Naive reference implementation of SiLU for a single value.
 *
 * @param x Input value
 * @return SiLU(x) = x / (1.0f + exp(-x))
 */
float silu(float x);

