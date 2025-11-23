add_library(models STATIC
    ${CMAKE_SOURCE_DIR}/src/models/qwen3model.cpp
    ${CMAKE_SOURCE_DIR}/src/models/inference_engine.cpp
)

target_link_libraries(models PUBLIC ops tensor)

