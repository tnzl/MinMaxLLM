#pragma once

#include <cpu_ops/base_op.h>
#include <immintrin.h>
#include <omp.h>
#include <cstdio>
#include <optional>
#include <tensor/tensor.h>

class Tensor;

void linear_naive(const float *input, const float *weight, int M, int K, int N, float *output);
void linear_avx2_omp(const float *input, const float *weight, int M, int K, int N, float *output);

enum class LinearImplType
{
    NAIVE,
    AVX2
};

class LinearOp : public BaseOp
{
public:
    using ImplFunction = void (*)(Tensor &input, Tensor &weight, Tensor &output);

    LinearOp(OpBackend backend = OpBackend::AVX2);
    LinearOp(Tensor &&weight, OpBackend backend = OpBackend::AVX2);
    ~LinearOp();

    void prepare() override;

    void run(Tensor &input, Tensor &output);
    void run(Tensor &input, Tensor &weight, Tensor &output);

private:
    // all the validations specific to impl and kernel call will be done in the impl functions
    static void naive_impl(Tensor &input, Tensor &weight, Tensor &output);
    static void avx2_impl(Tensor &input, Tensor &weight, Tensor &output);

    void run_internal(Tensor &input, Tensor &weight, Tensor &output);
    LinearImplType resolve_impl(Tensor &input, Tensor &weight) const;

    std::optional<Tensor> owned_weight_;

};