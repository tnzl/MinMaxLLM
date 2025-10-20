# CMake file for safetensors library
add_library(tensor STATIC
    ${CMAKE_SOURCE_DIR}/src/tensor/safetensors.cpp
)
