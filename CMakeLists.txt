cmake_minimum_required(VERSION 3.10)
project(ElementwiseAdd CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add executable
add_executable(elementwise_add 
    examples/elew_add/host.cpp
    src/kernels/elew_add.cu
)

# CUDA settings
set_target_properties(elementwise_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)