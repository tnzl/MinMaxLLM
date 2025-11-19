#pragma once

#include <cpu_ops/base_op.h>
#include <memory>

class Tensor;
class KVCache;

// Reference kernels
void gqa_naive(
    const float *query, // [A, h]
    const float *key,   // [G, N_max, h]
    const float *value, // [G, N_max, h]
    float *output,      // [A, h]
    int N,              // actual sequence length
    int N_max,          // max sequence length
    int G,              // number of KV groups
    int A,              // number of attention heads
    int h,              // head dimension
    float scale         // scaling factor (typically 1/sqrt(h))
);

void optimized_gqa_forward(
    const float *query, // [A, h] - single token query for all attention heads
    const float *key,   // [G, N_max, h] - keys for all KV groups and positions
    const float *value, // [G, N_max, h] - values for all KV groups and positions
    float *output,      // [A, h] - output for all attention heads
    int A,              // number of attention heads
    int G,              // number of KV groups
    int h,              // head dimension
    int N,              // actual sequence length (N <= N_max)
    int N_max,          // max sequence length
    float scale         // scaling factor
);

enum class GQAImplType
{
    NAIVE,
    AVX2
};

class GQAOp : public BaseOp
{
public:
    GQAOp(OpBackend backend = OpBackend::AVX2, float scale = 0.0f);
    GQAOp(KVCache *kvcache, size_t layer_idx, size_t num_heads, size_t num_groups, size_t head_dim, OpBackend backend = OpBackend::AVX2, float scale = 0.0f);
    ~GQAOp();

    void prepare() override;

    void run(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N);
    void run(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N, float scale);
    void run(Tensor &query, Tensor &output, int N);  // Uses KV cache from constructor

    void set_scale(float scale) { scale_ = scale; }
    float scale() const { return scale_; }

private:
    static void naive_impl(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N, float scale);
    static void avx2_impl(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N, float scale);

    void run_internal(Tensor &query, Tensor &key, Tensor &value, Tensor &output, int N, float scale);
    GQAImplType resolve_impl(Tensor &query, Tensor &key, Tensor &value) const;

    float scale_;
    KVCache *kvcache_;
    size_t layer_idx_;
    size_t num_heads_;
    size_t num_groups_;
    size_t head_dim_;
};