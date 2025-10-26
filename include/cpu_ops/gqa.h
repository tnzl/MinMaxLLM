#pragma once
#include <vector>
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cpu_ops/exp_avx2.h>
#include <cpu_ops/softmax_avx2.h>

#include <vector>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

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