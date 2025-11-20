#pragma once

#include <ops/base_op.h>
#include <immintrin.h>
#include <cstddef>
#include <tensor/tensor.h>

class Tensor;

// Reference kernels
void softmax_naive(const float* input, float* output, size_t size);
void softmax_avx2(const float* input, float* output, int size);

enum class SoftmaxImplType
{
    NAIVE,
    AVX2
};

class SoftmaxOp : public BaseOp
{
public:
    SoftmaxOp(OpBackend backend = OpBackend::AVX2);
    ~SoftmaxOp();

    void prepare() override;

    void run(Tensor &input, Tensor &output);

private:
    static void naive_impl(Tensor &input, Tensor &output);
    static void avx2_impl(Tensor &input, Tensor &output);

    void run_internal(Tensor &input, Tensor &output);
    SoftmaxImplType resolve_impl(Tensor &input) const;
};

