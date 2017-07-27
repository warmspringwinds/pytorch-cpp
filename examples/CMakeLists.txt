
ADD_EXECUTABLE(read_allocated_gpu_memory read_allocated_gpu_memory.cpp)
add_dependencies(read_allocated_gpu_memory ATen)
TARGET_LINK_LIBRARIES(read_allocated_gpu_memory ${CMAKE_CURRENT_BINARY_DIR}/../ATen/src/ATen/libATen.so.1 ${CUDA_LIBRARIES})

