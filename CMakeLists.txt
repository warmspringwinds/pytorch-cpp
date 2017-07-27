cmake_minimum_required(VERSION 3.0)
project(boo)

#set(CMAKE_CURRENT_SOURCE_DIR /ATen /home/mvp/platform/repos/pytorch_sandbox/ATen)
#set(ATen_BINARY_DIR  /home/mvp/platform/repos/pytorch_sandbox/ATen/build)

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

find_package(CUDA 5.5)

# Build the ATen dependency
add_subdirectory(ATen)

include_directories(
  # dense
  ${CMAKE_CURRENT_SOURCE_DIR}/ATen/lib/TH
  ${CMAKE_CURRENT_SOURCE_DIR}/ATen/lib/THC
  ${CMAKE_CURRENT_BINARY_DIR}/ATen/lib/TH
  ${CMAKE_CURRENT_BINARY_DIR}/ATen/lib/THC
  # sparse
  ${CMAKE_CURRENT_SOURCE_DIR}/ATen/lib/THS
  ${CMAKE_CURRENT_SOURCE_DIR}/ATen/lib/THCS
  ${CMAKE_CURRENT_BINARY_DIR}/ATen/lib/THS
  ${CMAKE_CURRENT_BINARY_DIR}/ATen/lib/THCS

  ${CMAKE_CURRENT_SOURCE_DIR}/ATen/lib
  ${CMAKE_CURRENT_BINARY_DIR}/ATen/lib)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/ATen/lib/THNN
${CMAKE_CURRENT_SOURCE_DIR}/ATen/lib/THCUNN)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/ATen/src
${CMAKE_CURRENT_BINARY_DIR}/ATen/src/ATen)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

add_subdirectory(examples)