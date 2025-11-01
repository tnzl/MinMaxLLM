add_executable(test_tensor ${CMAKE_SOURCE_DIR}/tests/tensor/test_tensor.cpp)
add_executable(test_kvcache ${CMAKE_SOURCE_DIR}/tests/tensor/test_kvcache.cpp)

target_link_libraries(test_tensor tensor)
target_link_libraries(test_kvcache tensor)

set_target_properties(test_tensor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(test_kvcache PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})