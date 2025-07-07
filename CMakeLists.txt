cmake_minimum_required(VERSION 3.10)
project(MinMaxLLM)

set(CMAKE_CXX_STANDARD 17)

option(USE_AVX2 "Enable AVX2 optimizations" ON)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Set compiler flags for AVX2 if enabled
if(USE_AVX2)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2 -mfma)
    endif()
    message(STATUS "Building with AVX2 optimizations")
else()
    message(STATUS "Building without AVX2 optimizations")
endif()

# Create a library for your CPU operations
add_library(cpu_ops STATIC
    src/cpu_ops/matmul.cpp
    src/cpu_ops/rotary_embedding.cpp
    src/cpu_ops/gqa.cpp
    src/cpu_ops/SkipSimplifiedLayerNormalization_AVX2.cpp
    src/cpu_ops/SimplifiedLayerNormalization_AVX2.cpp
    src/cpu_ops/silu_avx2.cpp
    src/cpu_ops/softmax_avx2.cpp # Added softmax AVX2 operator
)

# Add executable and link the library
add_executable(test_matmul
    tests/cpu_ops/test_matmul.cpp
)
add_executable(test_rotary_embedding
    tests/cpu_ops/test_rotary_embedding.cpp
)
add_executable(test_gqa
    tests/cpu_ops/test_gqa.cpp
)
add_executable(test_SkipSimplifiedLayerNormalization_AVX2
    tests/cpu_ops/test_SkipSimplifiedLayerNormalization_AVX2.cpp
)
add_executable(test_silu_avx2
    tests/cpu_ops/test_silu_avx2.cpp
)
add_executable(test_softmax_avx2
    tests/cpu_ops/test_softmax_avx2.cpp
)
add_executable(test_SimplifiedLayerNormalization_AVX2
    tests/cpu_ops/test_SimplifiedLayerNormalization_AVX2.cpp
)

target_link_libraries(test_matmul cpu_ops)
target_link_libraries(test_rotary_embedding cpu_ops)
target_link_libraries(test_gqa cpu_ops)
target_link_libraries(test_SkipSimplifiedLayerNormalization_AVX2 cpu_ops)
target_link_libraries(test_silu_avx2 cpu_ops)
target_link_libraries(test_softmax_avx2 cpu_ops)
target_link_libraries(test_SimplifiedLayerNormalization_AVX2 cpu_ops)