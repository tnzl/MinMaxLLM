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

target_link_libraries(test_matmul cpu_ops)
target_link_libraries(test_rotary_embedding cpu_ops)
target_link_libraries(test_gqa cpu_ops)