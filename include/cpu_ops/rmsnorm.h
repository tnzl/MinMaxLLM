#pragma once

#include <cpu_ops/base_op.h>
#include <optional>
#include <tensor/tensor.h>

class Tensor;

// Reference kernels
void rmsnorm_naive(const float *input, const float *weight, float *output, int batch_size, int hidden_size, float eps);
void rmsnorm_avx2(const float *input, const float *weight, float *output, int batch_size, int hidden_size, float eps);

enum class RMSNormImplType
{
    NAIVE,
    AVX2
};

class RMSNormOp : public BaseOp
{
public:
    RMSNormOp(OpBackend backend = OpBackend::AVX2, float epsilon = 1e-6f);
    RMSNormOp(Tensor &&weight, OpBackend backend = OpBackend::AVX2, float epsilon = 1e-6f);
    ~RMSNormOp();

    void prepare() override;

    void run(Tensor &input, Tensor &output);
    void run(Tensor &input, Tensor &output, float eps);
    void run(Tensor &input, Tensor &weight, Tensor &output);
    void run(Tensor &input, Tensor &weight, Tensor &output, float eps);

    void set_epsilon(float eps) { eps_ = eps; }
    float epsilon() const { return eps_; }

private:
    static void naive_impl(Tensor &input, Tensor &weight, Tensor &output, float eps);
    static void avx2_impl(Tensor &input, Tensor &weight, Tensor &output, float eps);

    void run_internal(Tensor &input, Tensor &weight, Tensor &output, float eps);
    RMSNormImplType resolve_impl(Tensor &input, Tensor &weight) const;

    std::optional<Tensor> owned_weight_;
    float eps_;
};
