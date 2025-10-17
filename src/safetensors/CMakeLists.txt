# CMake file for safetensors library
add_library(safetensors STATIC
    ${CMAKE_SOURCE_DIR}/src/safetensors/safetensors.cpp
)
