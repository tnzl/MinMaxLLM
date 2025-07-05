#pragma once
#include <immintrin.h>
#include <cstddef>

/**
 * @brief Computes the softmax of a float array using AVX2 vectorization.
 *
 * This function normalizes the input array in-place so that the output values
 * sum to 1, using numerically stable softmax (subtracting max before exp).
 *
 * @param arr Pointer to the input array (modified in-place)
 * @param size Number of elements in the array
 */
void softmax_avx2(float* arr, int size);
