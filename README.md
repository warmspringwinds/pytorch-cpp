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
4. ```cd ../../``` -- returns you back to the root directory (necessary for the next step).

### HDF5

We use ```HDF5``` to be able to [easily convert](convert_weights.ipynb) weigths between ```Pytorch``` and ```Pytorch-C++```.

0. ```wget https://support.hdfgroup.org/ftp/HDF5/current18/src/CMake-hdf5-1.8.19.tar.gz; tar xvzf CMake-hdf5-1.8.19.tar.gz```
1. ```cd CMake-hdf5-1.8.19; ./build-unix.sh```
2. ```cd ../``` -- return back.

### Opencv

We need ```OpenCV``` for a couple of examples which grab frames from a web camera.
It is not a dependency and can be removed if necessary.
This was tested on ```Ubuntu-16``` and might need some changes on a different system.

0. ```sudo apt-get install libopencv-dev python-opencv```


### Pytorch-C++

```Pytorch-C++``` is a library on top of ```ATen``` that provides a [Pytorch](http://pytorch.org/)-like
interface for building neural networks and forward inference (so far only forward inference is supported)
inspired by [cunnproduction](https://github.com/szagoruyko/cunnproduction) library. To install it, follow
these steps:

0. ```mkdir build; cd build; cmake-gui ..``` and specify ```CUDA_TOOLKIT_ROOT_DIR```.
1. ```make```
2. ```cd ../``` -- return back

If you face any problem or some steps are not clear, please open an issue. Note: every time you enter the ```cmake-gui``
press ```configure``` first, then specify your ```CUDA``` path and then press ```generate```, after that you can build.
