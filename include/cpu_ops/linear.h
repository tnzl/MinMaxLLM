#pragma once

#include <immintrin.h>
#include <omp.h>
#include <cstdio>
#include <memory>
#include <unordered_map>

class Tensor;

void linear_naive(const float *input, const float *weight, int M, int K, int N, float *output);
void linear_avx2_omp(const float *input, const float *weight, int M, int K, int N, float *output);

enum class MatmulImplType
{
    NAIVE,
    AVX2
};

class LinearOp
{
public:
    using ImplFunction = void (*)(Tensor &input, Tensor &weight, Tensor &output);

    LinearOp(MatmulImplType impl_type = MatmulImplType::AVX2);
    LinearOp(Tensor &&weight, MatmulImplType impl_type = MatmulImplType::AVX2);
    ~LinearOp();

    void prepare();

    void run(Tensor &input, Tensor &output);
    void run(Tensor &input, Tensor &weight, Tensor &output);

private:
    // all the validations specific to impl and kernel call will be done in the impl functions
    static void naive_impl(Tensor &input, Tensor &weight, Tensor &output);
    static void avx2_impl(Tensor &input, Tensor &weight, Tensor &output);

    void run_internal(Tensor &input, Tensor &weight, Tensor &output);
    MatmulImplType resolve_impl(Tensor &input, Tensor &weight) const;

    MatmulImplType impl_type_ = MatmulImplType::AVX2;
    std::unique_ptr<Tensor> owned_weight_;

    static std::unordered_map<MatmulImplType, ImplFunction> impl_registry_;
};