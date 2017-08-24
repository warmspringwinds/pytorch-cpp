# Pytorch-C++

## Installation

### ATen

[ATen](https://github.com/zdevito/ATen) is a C++ 11 library that wraps a powerfull C Tensor library with
implementation of numpy-like operations (CPU/CUDA/SPARSE/CUDA-SPARSE backends).
Follow these steps to install it:  

0. Make sure you have [dependencies](https://github.com/zdevito/ATen#installation) of ```ATen``` installed.
1. ```git clone --recursive https://github.com/warmspringwinds/pytorch-cpp```
2. ```cd pytorch-cpp/ATen;mkdir build;cd build;cmake-gui .. ``` and specify ```CUDA_TOOLKIT_ROOT_DIR```.
3. ```make``` or better ```make -j7``` (replace ```7``` with a number of cores that you have).

### Pytorch-C++

```Pytorch-C++``` is a library on top of ```ATen``` that provides a [Pytorch](http://pytorch.org/)-like
interface for building neural networks and forward inference (so far only forward inference is supported)
inspired by [cunnproduction](https://github.com/szagoruyko/cunnproduction) library.
