#include <cpu_ops/rmsnorm.h>
#include <tensor/tensor.h>

#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <stdexcept>

namespace
{
struct RMSNormDims
{
    int batch;
    int hidden;
};

RMSNormDims compute_rmsnorm_dims(Tensor &input, Tensor &weight, Tensor &output)
{
    const auto &input_shape = input.shape();
    assert(!input_shape.empty() && input_shape.size() <= 2 && "RMSNormOp expects input tensor of rank 1 or 2.");

    int hidden = static_cast<int>(input_shape.back());
    int batch = 1;
    if (input_shape.size() == 2)
    {
        batch = static_cast<int>(input_shape.front());
    }

    const auto &weight_shape = weight.shape();
    assert(weight_shape.size() == 1 && static_cast<int>(weight_shape[0]) == hidden && "RMSNormOp expects weight tensor with shape [hidden_dim].");

    const auto &output_shape = output.shape();
    if (input_shape.size() == 1)
    {
        assert(output_shape.size() == 1 && static_cast<int>(output_shape[0]) == hidden && "RMSNormOp output tensor shape mismatch for vector input.");
    }
    else
    {
        assert(output_shape.size() == 2 && static_cast<int>(output_shape[0]) == batch && static_cast<int>(output_shape[1]) == hidden && "RMSNormOp output tensor shape mismatch.");
    }

    assert(hidden > 0 && "RMSNormOp hidden dimension must be positive.");

    return {batch, hidden};
}
} // namespace

void rmsnorm_naive(const float *input, const float *weight, float *output, int batch_size, int hidden_size, float eps)
{
    for (int b = 0; b < batch_size; ++b)
    {
        const float *in_row = input + static_cast<size_t>(b) * hidden_size;
        const float *w_row = weight;
        float *out_row = output + static_cast<size_t>(b) * hidden_size;

        float sum_sq = 0.0f;
        for (int d = 0; d < hidden_size; ++d)
        {
            float v = in_row[d];
            sum_sq += v * v;
        }

        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        float denom = 1.0f / std::sqrt(mean_sq + eps);

        for (int d = 0; d < hidden_size; ++d)
        {
            out_row[d] = in_row[d] * denom * w_row[d];
        }
    }
}

void rmsnorm_avx2(const float *input, const float *weight, float *output, int batch_size, int hidden_size, float eps)
{
    for (int b = 0; b < batch_size; ++b)
    {
        const float *in_row = input + static_cast<size_t>(b) * hidden_size;
        const float *w_row = weight;
        float *out_row = output + static_cast<size_t>(b) * hidden_size;

        int d = 0;
        __m256 sum_vec = _mm256_setzero_ps();
        for (; d + 8 <= hidden_size; d += 8)
        {
            __m256 x = _mm256_loadu_ps(in_row + d);
            sum_vec = _mm256_fmadd_ps(x, x, sum_vec);
        }

        alignas(32) float sum_arr[8];
        _mm256_store_ps(sum_arr, sum_vec);

        float sum_sq = 0.0f;
        for (int i = 0; i < 8; ++i)
        {
            sum_sq += sum_arr[i];
        }
        for (; d < hidden_size; ++d)
        {
            float v = in_row[d];
            sum_sq += v * v;
        }

        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        float denom = 1.0f / std::sqrt(mean_sq + eps);
        __m256 denom_vec = _mm256_set1_ps(denom);

        d = 0;
        for (; d + 8 <= hidden_size; d += 8)
        {
            __m256 x = _mm256_loadu_ps(in_row + d);
            __m256 w = _mm256_loadu_ps(w_row + d);
            __m256 norm = _mm256_mul_ps(x, denom_vec);
            __m256 out = _mm256_mul_ps(w, norm);
            _mm256_storeu_ps(out_row + d, out);
        }
        for (; d < hidden_size; ++d)
        {
            out_row[d] = in_row[d] * denom * w_row[d];
        }
    }
}

RMSNormOp::RMSNormOp(OpBackend backend, float epsilon) : BaseOp(backend), eps_(epsilon)
{
}

RMSNormOp::RMSNormOp(Tensor &&weight, OpBackend backend, float epsilon) : BaseOp(backend), eps_(epsilon)
{
    owned_weight_ = std::move(weight);
}

RMSNormOp::~RMSNormOp() = default;

void RMSNormOp::prepare()
{
    if (owned_weight_.has_value())
    {
        owned_weight_->prefetch_async();
    }
}

void RMSNormOp::run(Tensor &input, Tensor &output)
{
    if (!owned_weight_.has_value())
    {
        throw std::runtime_error("RMSNormOp::run called without a stored weight tensor.");
    }

    run_internal(input, *owned_weight_, output, eps_);
}

void RMSNormOp::run(Tensor &input, Tensor &output, float eps)
{
    if (!owned_weight_.has_value())
    {
        throw std::runtime_error("RMSNormOp::run called without a stored weight tensor.");
    }

    run_internal(input, *owned_weight_, output, eps);
}

void RMSNormOp::run(Tensor &input, Tensor &weight, Tensor &output)
{
    run_internal(input, weight, output, eps_);
}

void RMSNormOp::run(Tensor &input, Tensor &weight, Tensor &output, float eps)
{
    run_internal(input, weight, output, eps);
}

void RMSNormOp::run_internal(Tensor &input, Tensor &weight, Tensor &output, float eps)
{
    RMSNormImplType selected = resolve_impl(input, weight);
    switch (selected)
    {
    case RMSNormImplType::NAIVE:
        naive_impl(input, weight, output, eps);
        break;
    case RMSNormImplType::AVX2:
        avx2_impl(input, weight, output, eps);
        break;
    default:
        throw std::runtime_error("No implementation registered for selected RMSNormImplType.");
    }
}

RMSNormImplType RMSNormOp::resolve_impl(Tensor &, Tensor &) const
{
    //TODO: move all the compute_rmsnorm_dims calls here so that it helps resolve the implementation
    
    switch (backend_)
    {
    case OpBackend::NAIVE:
        return RMSNormImplType::NAIVE;
    case OpBackend::AVX2:
        return RMSNormImplType::AVX2;
    default:
        return RMSNormImplType::NAIVE;
    }
}

void RMSNormOp::naive_impl(Tensor &input, Tensor &weight, Tensor &output, float eps)
{
    assert(input.dtype() == DataType::F32 && "RMSNormOp::naive_impl supports only float32 input tensors.");
    assert(weight.dtype() == DataType::F32 && "RMSNormOp::naive_impl supports only float32 weight tensors.");
    assert(output.dtype() == DataType::F32 && "RMSNormOp::naive_impl supports only float32 output tensors.");

    const RMSNormDims dims = compute_rmsnorm_dims(input, weight, output);
    rmsnorm_naive(input.data<float>(), weight.data<float>(), output.data<float>(), dims.batch, dims.hidden, eps);
}

void RMSNormOp::avx2_impl(Tensor &input, Tensor &weight, Tensor &output, float eps)
{
    assert(input.dtype() == DataType::F32 && "RMSNormOp::avx2_impl supports only float32 input tensors.");
    assert(weight.dtype() == DataType::F32 && "RMSNormOp::avx2_impl supports only float32 weight tensors.");
    assert(output.dtype() == DataType::F32 && "RMSNormOp::avx2_impl supports only float32 output tensors.");

    const RMSNormDims dims = compute_rmsnorm_dims(input, weight, output);
    rmsnorm_avx2(input.data<float>(), weight.data<float>(), output.data<float>(), dims.batch, dims.hidden, eps);
}
