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

# Add subdirectories for cpu_ops and tests
add_subdirectory(src/cpu_ops)
add_subdirectory(src/tensor)
add_subdirectory(src/models)
add_subdirectory(tests/cpu_ops)
add_subdirectory(tests/modules)
add_subdirectory(tests/tensor)