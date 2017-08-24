# pytorch-cpp

## Installation

1. git clone --recursive https://github.com/warmspringwinds/pytorch-cpp
2. ```cd pytorch-cpp/ATen;mkdir build;cd build;cmake-gui .. ``` and specify ```CUDA_TOOLKIT_ROOT_DIR```.
3. ```make``` or better ```make -j7``` (replace ```7``` with a number of cores that you have). 
