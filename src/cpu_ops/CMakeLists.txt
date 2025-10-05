# CMake file for cpu_ops library
add_library(cpu_ops STATIC
    ${CMAKE_SOURCE_DIR}/src/cpu_ops/matmul.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_ops/rotary_embedding.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_ops/gqa.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_ops/SkipSimplifiedLayerNormalization_AVX2.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_ops/SimplifiedLayerNormalization_AVX2.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_ops/silu_avx2.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_ops/softmax_avx2.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_ops/rmsnorm.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu_ops/elemwise_mul.cpp
)
