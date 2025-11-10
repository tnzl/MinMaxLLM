#include <cpu_ops/linear.h>
#include <tensor/tensor.h>

#include <cassert>
#include <stdexcept>

namespace
{
struct LinearDims
{
    int M;
    int K;
    int N;
};

LinearDims compute_linear_dims(Tensor &input, Tensor &weight)
{
    const auto &input_shape = input.shape();
    assert(!input_shape.empty() && "LinearOp expects input tensor with at least one dimension.");

    int M = 1;
    int K = static_cast<int>(input_shape.back());
    if (input_shape.size() > 1)
    {
        M = static_cast<int>(input_shape.front());
    }

    const auto &weight_shape = weight.shape();
    assert(weight_shape.size() >= 2 && "LinearOp expects weight tensor with shape [out_features, in_features].");

    int N = static_cast<int>(weight_shape.front());
    int weight_k = static_cast<int>(weight_shape.back());

    assert(weight_k == K && "LinearOp weight and input feature dimensions must match.");

    return {M, K, N};
}

void ensure_output_shape(Tensor &output, int M, int N)
{
    const auto &output_shape = output.shape();
    assert(!output_shape.empty() && "LinearOp expects output tensor with shape.");

    if (output_shape.size() == 1)
    {
        assert(M == 1 && output_shape[0] == static_cast<size_t>(N) && "LinearOp output tensor shape mismatch for vector output.");
        return;
    }

    assert(output_shape.size() == 2 && output_shape[0] == static_cast<size_t>(M) && output_shape[1] == static_cast<size_t>(N) && "LinearOp output tensor shape mismatch.");
}


void validate_dtype(Tensor &tensor)
{
    assert(tensor.dtype() == DataType::F32 && "LinearOp currently supports only float32 tensors.");
}
} // namespace

// Naive reference version
void linear_naive(const float *input, const float *weight, int M, int K, int N, float *output)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += input[i * K + k] * weight[j * K + k];
            }
            output[i * N + j] = sum;
        }
    }
}

void linear_avx2_omp(const float *input, const float *weight, int M, int K, int N, float *output)
{
#pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        const float *in_row = input + i * K;
        for (int j = 0; j < N; ++j)
        {
            const float *w_row = weight + j * K;

            __m256 vsum = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8)
            {
                __m256 va = _mm256_loadu_ps(in_row + k);
                __m256 vb = _mm256_loadu_ps(w_row + k);
                vsum = _mm256_fmadd_ps(va, vb, vsum); // fused multiply-add
            }

            // horizontal add
            __m128 low = _mm256_castps256_ps128(vsum);
            __m128 high = _mm256_extractf128_ps(vsum, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);

            // Remainder
            for (; k < K; ++k)
                sum += in_row[k] * w_row[k];

            output[i * N + j] = sum;
        }
    }
}

std::unordered_map<MatmulImplType, LinearOp::ImplFunction> LinearOp::impl_registry_ = {
    {MatmulImplType::NAIVE, &LinearOp::naive_impl},
    {MatmulImplType::AVX2, &LinearOp::avx2_impl}};

LinearOp::LinearOp(MatmulImplType impl_type) : impl_type_(impl_type)
{
}

// callers can hand over temporaries or moved lvalues, while we still move once into owned_weight_
LinearOp::LinearOp(Tensor &&weight, MatmulImplType impl_type) : impl_type_(impl_type)
{
    owned_weight_ = std::make_unique<Tensor>(std::move(weight));
}

LinearOp::~LinearOp() = default;

void LinearOp::prepare()
{
    if (owned_weight_)
    {
        owned_weight_->prefetch_async();
    }
}

void LinearOp::run(Tensor &input, Tensor &output)
{
    if (!owned_weight_)
    {
        throw std::runtime_error("LinearOp::run called without a stored weight tensor.");
    }

    run_internal(input, *owned_weight_, output);
}

void LinearOp::run(Tensor &input, Tensor &weight, Tensor &output)
{
    run_internal(input, weight, output);
}

void LinearOp::run_internal(Tensor &input, Tensor &weight, Tensor &output)
{
    MatmulImplType selected = resolve_impl(input, weight);
    auto it = impl_registry_.find(selected);
    if (it == impl_registry_.end())
    {
        throw std::runtime_error("No implementation registered for selected MatmulImplType.");
    }

    it->second(input, weight, output);
}

MatmulImplType LinearOp::resolve_impl(Tensor &input, Tensor &weight) const
{
    const LinearDims dims = compute_linear_dims(input, weight);

    if (impl_registry_.count(impl_type_))
    {
        return impl_type_;
    }

    return MatmulImplType::NAIVE;
}

void LinearOp::naive_impl(Tensor &input, Tensor &weight, Tensor &output)
{
    validate_dtype(input);
    validate_dtype(weight);
    validate_dtype(output);

    const LinearDims dims = compute_linear_dims(input, weight);
    linear_naive(input.data<float>(), weight.data<float>(), dims.M, dims.K, dims.N, output.data<float>());
}

void LinearOp::avx2_impl(Tensor &input, Tensor &weight, Tensor &output)
{
    validate_dtype(input);
    validate_dtype(weight);
    validate_dtype(output);

    const LinearDims dims = compute_linear_dims(input, weight);
    linear_avx2_omp(input.data<float>(), weight.data<float>(), dims.M, dims.K, dims.N, output.data<float>());
}
