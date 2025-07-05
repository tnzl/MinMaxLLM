#include <cpu_ops/exp_avx2.h>
#include <cpu_ops/softmax_avx2.h>
#include <algorithm>
#include <cmath>

// exp256_ps should be available from exp_avx2.h
extern __m256 exp256_ps(__m256 x);

/**
 * @brief AVX2-optimized softmax implementation for float arrays.
 *
 * This function computes the softmax of the input array in-place using AVX2 intrinsics.
 * Steps:
 *   1. Find the maximum value for numerical stability.
 *   2. Subtract max, exponentiate, and sum all values (vectorized).
 *   3. Normalize by dividing by the sum (vectorized).
 *
 * @param arr Pointer to the input array (modified in-place)
 * @param size Number of elements in the array
 */
void softmax_avx2(float* arr, int size) {
    // Step 1: Find max value in the array for numerical stability
    __m256 max_val = _mm256_set1_ps(-INFINITY);
    int i;
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(arr + i);
        max_val = _mm256_max_ps(max_val, vec);
    }
    alignas(32) float max_arr[8];
    _mm256_store_ps(max_arr, max_val);
    float max_scalar = std::max({max_arr[0], max_arr[1], max_arr[2], max_arr[3],
                                max_arr[4], max_arr[5], max_arr[6], max_arr[7]});
    for (; i < size; ++i) {
        if (arr[i] > max_scalar) {
            max_scalar = arr[i];
        }
    }
    // Step 2: Subtract max, exponentiate, and sum
    __m256 max_vec = _mm256_set1_ps(max_scalar);
    __m256 sum_vec = _mm256_setzero_ps();
    float sum = 0.0f;
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(arr + i);
        vec = _mm256_sub_ps(vec, max_vec);
        vec = exp256_ps(vec);
        _mm256_storeu_ps(arr + i, vec);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    alignas(32) float sum_arr[8];
    _mm256_store_ps(sum_arr, sum_vec);
    sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
          sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
    for (; i < size; ++i) {
        arr[i] = std::exp(arr[i] - max_scalar);
        sum += arr[i];
    }
    // Step 3: Normalize by dividing by the sum
    __m256 sum_vec_inv = _mm256_set1_ps(1.0f / sum);
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(arr + i);
        vec = _mm256_mul_ps(vec, sum_vec_inv);
        _mm256_storeu_ps(arr + i, vec);
    }
    for (; i < size; ++i) {
        arr[i] /= sum;
    }
}
