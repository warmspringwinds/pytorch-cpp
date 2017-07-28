
#ADD_EXECUTABLE(read_allocated_gpu_memory read_allocated_gpu_memory.cpp)
#TARGET_LINK_LIBRARIES(read_allocated_gpu_memory ${ATen_BINARY_DIR}/src/ATen/libATen.so.1 ${CUDA_LIBRARIES})

#ADD_EXECUTABLE(sequential_test sequential_test.cpp)
#TARGET_LINK_LIBRARIES(sequential_test ${ATen_BINARY_DIR}/src/ATen/libATen.so.1 ${CUDA_LIBRARIES})

