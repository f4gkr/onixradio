
add_executable(onnxtest "main.cpp")
target_include_directories(onnxtest PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(onnxtest PRIVATE onnxruntime )