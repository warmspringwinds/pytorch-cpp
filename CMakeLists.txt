cmake_minimum_required(VERSION 3.0)
project(boo)


#set(ATen_SOURCE_DIR  /home/mvp/platform/repos/pytorch_sandbox/ATen)
#set(ATen_BINARY_DIR  /home/mvp/platform/repos/pytorch_sandbox/ATen/build)

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()


# Build the ATen dependency
add_subdirectory(ATen)



#
#include_directories(
#  # dense
#  ${ATen_SOURCE_DIR}/lib/TH
#  ${ATen_SOURCE_DIR}/lib/THC
#  ${ATen_BINARY_DIR}/lib/TH
#  ${ATen_BINARY_DIR}/lib/THC
#  # sparse
#  ${ATen_SOURCE_DIR}/lib/THS
#  ${ATen_SOURCE_DIR}/lib/THCS
#  ${ATen_BINARY_DIR}/lib/THS
#  ${ATen_BINARY_DIR}/lib/THCS
#
#  ${ATen_SOURCE_DIR}/lib
#  ${ATen_BINARY_DIR}/lib)
#
#
#find_package(CUDA 5.5)
#
#include_directories(
#${ATen_SOURCE_DIR}/lib/THNN
#${ATen_SOURCE_DIR}/lib/THCUNN)
#
#
#include_directories(
#${ATen_SOURCE_DIR}/src
#${ATen_BINARY_DIR}/src/ATen)
#
#INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
#
#ADD_EXECUTABLE(scalar_test scalar_test.cpp)
#TARGET_LINK_LIBRARIES(scalar_test /home/mvp/platform/repos/pytorch_sandbox/ATen/build/src/ATen/libATen.so.1)
#
#add_executable(atest atest.cpp)
#target_link_libraries(atest /home/mvp/platform/repos/pytorch_sandbox/ATen/build/src/ATen/libATen.so.1)
#
#
##ADD_EXECUTABLE(basic basic.cpp)
##TARGET_LINK_LIBRARIES(basic /home/mvp/platform/repos/pytorch_sandbox/ATen/build/src/ATen/libATen.so.1)
#
#ADD_EXECUTABLE(gpu_test gpu_test.cpp)
#TARGET_LINK_LIBRARIES(gpu_test /home/mvp/platform/repos/pytorch_sandbox/ATen/build/src/ATen/libATen.so.1 ${CUDA_LIBRARIES})
