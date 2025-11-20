# CMake file for ops library
add_library(ops STATIC
    ${CMAKE_SOURCE_DIR}/src/ops/matmul.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/rotary_embedding.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/gqa.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/silu.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/softmax.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/rmsnorm.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/elemwise_mul.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/elemwise_add.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/linear.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/self_attention.cpp
    ${CMAKE_SOURCE_DIR}/src/ops/decoder.cpp
)

target_link_libraries(ops PUBLIC tensor)