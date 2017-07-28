cmake_minimum_required(VERSION 3.0)
project(boo)

set(ATen_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ATen)
set(ATen_BINARY_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/ATen/build)

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

find_package(CUDA 5.5)

include_directories(
  # dense
  ${ATen_SOURCE_DIR}/lib/TH
  ${ATen_SOURCE_DIR}/lib/THC
  ${ATen_BINARY_DIR}/lib/TH
  ${ATen_BINARY_DIR}/lib/THC
  # sparse
  ${ATen_SOURCE_DIR}/lib/THS
  ${ATen_SOURCE_DIR}/lib/THCS
  ${ATen_BINARY_DIR}/lib/THS
  ${ATen_BINARY_DIR}/lib/THCS

  ${ATen_SOURCE_DIR}/lib
  ${ATen_BINARY_DIR}/lib)

include_directories(
${ATen_SOURCE_DIR}/lib/THNN
${ATen_SOURCE_DIR}/lib/THCUNN)

include_directories(
${ATen_SOURCE_DIR}/src
${ATen_BINARY_DIR}/src/ATen)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

ADD_EXECUTABLE(read_allocated_gpu_memory examples/read_allocated_gpu_memory.cpp)
TARGET_LINK_LIBRARIES(read_allocated_gpu_memory ${ATen_BINARY_DIR}/src/ATen/libATen.so.1 ${CUDA_LIBRARIES})

ADD_EXECUTABLE(sequential_test examples/sequential_test.cpp)
TARGET_LINK_LIBRARIES(sequential_test ${ATen_BINARY_DIR}/src/ATen/libATen.so.1 ${CUDA_LIBRARIES})


#add_subdirectory(examples)