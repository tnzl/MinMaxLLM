#include "ops/exp_avx2.h"
#include "ops/silu.h"
#include <tensor/tensor.h>
#include <immintrin.h>
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace
{
size_t compute_silu_size(Tensor &input, Tensor &output)
{
    const auto &input_shape = input.shape();
    assert(!input_shape.empty() && "SiluOp expects input tensor with at least one dimension.");

    const auto &output_shape = output.shape();
    assert(input_shape == output_shape && "SiluOp expects input and output tensors to have the same shape.");

    size_t total_size = 1;
    for (size_t dim : input_shape)
    {
        total_size *= dim;
    }

    return total_size;
}
} // namespace

void silu_avx2(const float* x, float* out, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;
    
    // Handle misaligned start
    for (; i < n && (uintptr_t(x + i) % 32 != 0); ++i) {
        float xi = x[i];
        out[i] = xi / (1.0f + std::exp(-xi));
    }
    
    // Main SIMD loop
    for (; i + simd_width <= n; i += simd_width) {
        __m256 vx = _mm256_load_ps(x + i);
        
        // Fast sigmoid approximation for better performance
        __m256 vnegx = _mm256_sub_ps(_mm256_setzero_ps(), vx);
        __m256 vexp = exp256_ps(vnegx);  // Keep accurate or use approximation
        
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vsigmoid = _mm256_div_ps(vone, _mm256_add_ps(vone, vexp));
        
        // Use regular multiply (FMA not necessarily better here)
        __m256 vsilu = _mm256_mul_ps(vx, vsigmoid);
        _mm256_store_ps(out + i, vsilu);
    }
    
    // Handle remaining elements
    for (; i < n; ++i) {
        float xi = x[i];
        out[i] = xi / (1.0f + std::exp(-xi));
    }
}

// Naive reference SiLU implementation
float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void silu_naive(const float* x, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = silu(x[i]);
    }
}

SiluOp::SiluOp(OpBackend backend) : BaseOp(backend)
{
}

SiluOp::~SiluOp() = default;

void SiluOp::prepare()
{
    // SiLU doesn't have weights, so nothing to prepare
}

void SiluOp::run(Tensor &input, Tensor &output)
{
    run_internal(input, output);
}

void SiluOp::run_internal(Tensor &input, Tensor &output)
{
    SiluImplType selected = resolve_impl(input);
    switch (selected)
    {
    case SiluImplType::NAIVE:
        naive_impl(input, output);
        break;
    case SiluImplType::AVX2:
        avx2_impl(input, output);
        break;
    default:
        throw std::runtime_error("No implementation registered for selected SiluImplType.");
    }
}

SiluImplType SiluOp::resolve_impl(Tensor &) const
{
    switch (backend_)
    {
    case OpBackend::NAIVE:
        return SiluImplType::NAIVE;
    case OpBackend::AVX2:
        return SiluImplType::AVX2;
    default:
        return SiluImplType::NAIVE;
    }
}

void SiluOp::naive_impl(Tensor &input, Tensor &output)
{
    assert(input.dtype() == DataType::F32 && "SiluOp::naive_impl supports only float32 input tensors.");
    assert(output.dtype() == DataType::F32 && "SiluOp::naive_impl supports only float32 output tensors.");

    size_t n = compute_silu_size(input, output);
    silu_naive(input.data<float>(), output.data<float>(), n);
}

void SiluOp::avx2_impl(Tensor &input, Tensor &output)
{
    assert(input.dtype() == DataType::F32 && "SiluOp::avx2_impl supports only float32 input tensors.");
    assert(output.dtype() == DataType::F32 && "SiluOp::avx2_impl supports only float32 output tensors.");

    size_t n = compute_silu_size(input, output);
    silu_avx2(input.data<float>(), output.data<float>(), n);
}

