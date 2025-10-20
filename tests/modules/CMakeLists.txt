add_executable(run_Qwen3RMSNorm ${CMAKE_SOURCE_DIR}/tests/modules/Qwen3RMSNorm/run_Qwen3RMSNorm.cpp)
add_executable(run_Qwen3MLPGate ${CMAKE_SOURCE_DIR}/tests/modules/Qwen3MLPGate/run_Qwen3MLPGate.cpp)
add_executable(run_safetensors ${CMAKE_SOURCE_DIR}/tests/modules/safetensors/run_safetensors.cpp)
add_executable(run_Qwen3MLP ${CMAKE_SOURCE_DIR}/tests/modules/Qwen3MLP/run_Qwen3MLP.cpp)

target_link_libraries(run_Qwen3RMSNorm cpu_ops)
target_link_libraries(run_Qwen3MLPGate cpu_ops)
target_link_libraries(run_safetensors tensor)
target_link_libraries(run_Qwen3MLP cpu_ops tensor)

set_target_properties(run_Qwen3RMSNorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(run_Qwen3MLPGate PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(run_safetensors PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(run_Qwen3MLP PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})