add_executable(run_Qwen3RMSNorm ${CMAKE_SOURCE_DIR}/tests/modules/Qwen3RMSNorm/run_Qwen3RMSNorm.cpp)

target_link_libraries(run_Qwen3RMSNorm cpu_ops)

set_target_properties(run_Qwen3RMSNorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})