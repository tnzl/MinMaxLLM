#include <ops/exp_avx2.h>
#include <ops/softmax.h>
#include <tensor/tensor.h>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <stdexcept>

// exp256_ps should be available from exp_avx2.h
extern __m256 exp256_ps(__m256 x);

/**
 * @brief Naive reference implementation of softmax.
 *
 * This function computes the softmax of the input array, writing the result to output.
 * Uses a numerically stable softmax (subtracting max before exp).
 *
 * @param input Pointer to the input array
 * @param output Pointer to the output array
 * @param size Number of elements in the array
 */
void softmax_naive(const float* input, float* output, size_t size) {
    float max_val = input[0];
    for (size_t i = 1; i < size; ++i) max_val = (std::max)(max_val, input[i]);
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (size_t i = 0; i < size; ++i) output[i] /= sum;
}

/**
 * @brief AVX2-optimized softmax implementation for float arrays.
 *
 * This function computes the softmax of the input array using AVX2 intrinsics.
 * Steps:
 *   1. Find the maximum value for numerical stability.
 *   2. Subtract max, exponentiate, and sum all values (vectorized).
 *   3. Normalize by dividing by the sum (vectorized).
 *
 * @param input Pointer to the input array
 * @param output Pointer to the output array
 * @param size Number of elements in the array
 */
void softmax_avx2(const float* input, float* output, int size) {
    // Step 1: Find max value in the array for numerical stability
    __m256 max_val = _mm256_set1_ps(-INFINITY);
    int i;
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(input + i);
        max_val = _mm256_max_ps(max_val, vec);
    }
    alignas(32) float max_arr[8];
    _mm256_store_ps(max_arr, max_val);
    float max_scalar = max_arr[0];
    for (int j = 1; j < 8; ++j) {
        if (max_arr[j] > max_scalar) {
            max_scalar = max_arr[j];
        }
    }
    for (; i < size; ++i) {
        if (input[i] > max_scalar) {
            max_scalar = input[i];
        }
    }
    // Step 2: Subtract max, exponentiate, and sum
    __m256 max_vec = _mm256_set1_ps(max_scalar);
    __m256 sum_vec = _mm256_setzero_ps();
    float sum = 0.0f;
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(input + i);
        vec = _mm256_sub_ps(vec, max_vec);
        vec = exp256_ps(vec);
        _mm256_storeu_ps(output + i, vec);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    alignas(32) float sum_arr[8];
    _mm256_store_ps(sum_arr, sum_vec);
    sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
          sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
    for (; i < size; ++i) {
        output[i] = std::exp(input[i] - max_scalar);
        sum += output[i];
    }
    // Step 3: Normalize by dividing by the sum
    __m256 sum_vec_inv = _mm256_set1_ps(1.0f / sum);
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(output + i);
        vec = _mm256_mul_ps(vec, sum_vec_inv);
        _mm256_storeu_ps(output + i, vec);
    }
    for (; i < size; ++i) {
        output[i] /= sum;
    }
}

namespace
{
struct SoftmaxDims
{
    size_t size;
};

SoftmaxDims compute_softmax_dims(Tensor &input, Tensor &output)
{
    const auto &input_shape = input.shape();
    assert(!input_shape.empty() && "SoftmaxOp expects input tensor with at least one dimension.");

    size_t input_size = input.size();
    size_t output_size = output.size();

    assert(input_size == output_size && "SoftmaxOp input and output tensors must have the same size.");
    assert(input_size > 0 && "SoftmaxOp input tensor must have at least one element.");

    return {input_size};
}
} // namespace

SoftmaxOp::SoftmaxOp(OpBackend backend) : BaseOp(backend)
{
}

SoftmaxOp::~SoftmaxOp() = default;

void SoftmaxOp::prepare()
{
    // No preparation needed for softmax (no weights to prefetch)
}

void SoftmaxOp::run(Tensor &input, Tensor &output)
{
    run_internal(input, output);
}

void SoftmaxOp::run_internal(Tensor &input, Tensor &output)
{
    SoftmaxImplType selected = resolve_impl(input);
    switch (selected)
    {
    case SoftmaxImplType::NAIVE:
        naive_impl(input, output);
        break;
    case SoftmaxImplType::AVX2:
        avx2_impl(input, output);
        break;
    default:
        throw std::runtime_error("No implementation registered for selected SoftmaxImplType.");
    }
}

SoftmaxImplType SoftmaxOp::resolve_impl(Tensor &) const
{
    switch (backend_)
    {
    case OpBackend::NAIVE:
        return SoftmaxImplType::NAIVE;
    case OpBackend::AVX2:
        return SoftmaxImplType::AVX2;
    default:
        return SoftmaxImplType::NAIVE;
    }
}

void SoftmaxOp::naive_impl(Tensor &input, Tensor &output)
{
    assert(input.dtype() == DataType::F32 && "SoftmaxOp::naive_impl supports only float32 input tensors.");
    assert(output.dtype() == DataType::F32 && "SoftmaxOp::naive_impl supports only float32 output tensors.");

    const SoftmaxDims dims = compute_softmax_dims(input, output);
    softmax_naive(input.data<float>(), output.data<float>(), dims.size);
}

void SoftmaxOp::avx2_impl(Tensor &input, Tensor &output)
{
    assert(input.dtype() == DataType::F32 && "SoftmaxOp::avx2_impl supports only float32 input tensors.");
    assert(output.dtype() == DataType::F32 && "SoftmaxOp::avx2_impl supports only float32 output tensors.");

    const SoftmaxDims dims = compute_softmax_dims(input, output);
    softmax_avx2(input.data<float>(), output.data<float>(), static_cast<int>(dims.size));
}

