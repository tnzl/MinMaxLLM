add_library(models STATIC
    ${CMAKE_SOURCE_DIR}/src/models/qwen3model.cpp
)

target_link_libraries(models PUBLIC cpu_ops tensor)

