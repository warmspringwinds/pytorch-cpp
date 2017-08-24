# Pytorch-C++

```Pytorch-C++``` is a simple C++ 11 library which provides a [Pytorch](http://pytorch.org/)-like
interface for building neural networks and inference (so far only forward pass is supported). The library
respects the semantics of ```torch.nn``` module of PyTorch. Models from [pytorch/vision](https://github.com/pytorch/vision)
are supported and can be [easily converted](convert_weights.ipynb).

The library heavily relies on an amazing [ATen](https://github.com/zdevito/ATen) library and was inspired by
[cunnproduction](https://github.com/szagoruyko/cunnproduction).


## Use-cases

The library can be used in cases where you want to integrate your trained ```Pytorch```
networks into an existing C++ stack and you don't want to convert your weights to other libraries
like ```Caffe/Caffe2/Tensorflow```. The library respects the semantics of the ```Pytorch``` and uses
the same underlying C library to perform all the operations.

You can achieve more low-level control over your memory. For example,
you can use a memory that was already allocated on GPU. This way you can accept memory from other
application on GPU and avoid expensive transfer to CPU. See [this example](examples/read_allocated_gpu_memory.cpp).

Conversion from other image types like OpenCV's ```mat``` to ```Tensor``` can be easily performed and all the post-processing
can be done using numpy-like optimized operations, thanks to [ATen](https://github.com/zdevito/ATen) library.
See examples [here](examples/opencv_realtime_webcam_human_segmentation.cpp).


## Some examples

### Inference

```c++
auto net = torch::resnet50_imagenet();

net->load_weights("../resnet50_imagenet.h5");

# Transfer network to GPU
net->cuda();

# Generate a dummy tensor on GPU of type float
Tensor dummy_input = CUDA(kFloat).ones({1, 3, 224, 224});

# Perform inference
auto result = net->forward(dummy_input);

map<string, Tensor> dict;

# Get the result of the inference back to CPU
dict["main"] = result.toBackend(Backend::CPU);

# Save the result of the inference in the HDF5 file
torch::save("resnet50_output.h5", dict);
```

### Display network's architecture

```c++

auto net = torch::resnet50_imagenet();

net->load_weights("../resnet50_imagenet.h5");

cout << net->tostring() << endl;

```

Output:

```
ResNet (
 (conv1)  Conv2d( in_channels=3 out_channels=64 kernel_size=(7, 7) stride=(2, 2) padding=(3, 3) dilation=(1, 1) groups=1 bias=0 )
 (bn1)  BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
 (relu)  ReLU
 (maxpool)  MaxPool2d( kernel_size=(3, 3) stride=(2, 2) padding=(1, 1) )
 (layer1)  Sequential (
  (0)   Bottleneck (
   (conv1)    Conv2d( in_channels=64 out_channels=64 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn1)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv2)    Conv2d( in_channels=64 out_channels=64 kernel_size=(3, 3) stride=(1, 1) padding=(1, 1) dilation=(1, 1) groups=1 bias=0 )
   (bn2)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv3)    Conv2d( in_channels=64 out_channels=256 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn3)    BatchNorm2d( num_features=256 eps=0.000010 momentum=0.100000 )
   (downsample)    Sequential (
    (0)     Conv2d( in_channels=64 out_channels=256 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
    (1)     BatchNorm2d( num_features=256 eps=0.000010 momentum=0.100000 )
   )

  )

  (1)   Bottleneck (
   (conv1)    Conv2d( in_channels=256 out_channels=64 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn1)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv2)    Conv2d( in_channels=64 out_channels=64 kernel_size=(3, 3) stride=(1, 1) padding=(1, 1) dilation=(1, 1) groups=1 bias=0 )
   (bn2)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv3)    Conv2d( in_channels=256 out_channels=256 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn3)    BatchNorm2d( num_features=256 eps=0.000010 momentum=0.100000 )
  )

  (2)   Bottleneck (
   (conv1)    Conv2d( in_channels=256 out_channels=64 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn1)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv2)    Conv2d( in_channels=64 out_channels=64 kernel_size=(3, 3) stride=(1, 1) padding=(1, 1) dilation=(1, 1) groups=1 bias=0 )
   (bn2)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv3)    Conv2d( in_channels=256 out_channels=256 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn3)    BatchNorm2d( num_features=256 eps=0.000010 momentum=0.100000 )
  )

 )

 /*  .... */

 (avgpool)  AvgPool2d( kernel_size=(7, 7) stride=(1, 1) padding=(0, 0) )
 (fc)  nn.Linear( in_features=2048 out_features=1000 bias=1 )
)
```

### Inspect a Tensor


```c++
auto net = torch::resnet50_imagenet();

net->load_weights("../resnet50_imagenet.h5");
net->cuda();

Tensor dummy_input = CUDA(kFloat).ones({1, 3, 224, 224});

auto result = net->forward(dummy_input);

cout << result << endl;
```


```
Columns 1 to 10-0.3081  0.0798 -1.1900 -1.4837 -0.5136  0.3683 -2.1639 -0.8705 -1.8812 -0.1608

Columns 11 to 20 0.2168 -0.9283 -1.2954 -1.0791 -1.4445 -0.8946 -0.0959 -1.3099 -1.2062 -1.2327

Columns 21 to 30-1.0658  0.9427  0.5739 -0.2746 -1.0189 -0.3583 -0.1826  0.2785  0.2209 -0.3340

Columns 31 to 40-1.9800 -0.5552 -1.0804 -0.8056 -0.0005 -1.8402 -0.7979 -1.4823  1.3657 -0.8970

/*  .... */

Columns 961 to 970-0.0557 -0.7405 -0.5501 -1.7207 -0.7043 -1.0925  1.5812 -0.1215  0.8915  0.9794

Columns 971 to 980-1.1422 -0.1235 -0.5999 -2.1338 -0.0775 -0.8374 -0.2350 -0.0104 -0.0416 -1.0296

Columns 981 to 990-0.2914 -0.2242 -0.8063 -0.7818 -0.2714  0.0002 -1.2355  0.1238  0.0183 -0.6904

Columns 991 to 1000 0.5216 -1.8008 -1.7826 -1.2970 -1.6565 -1.3306 -0.6564 -1.6531  0.1178  0.2436
[ CUDAFloatTensor{1,1000} ]
```

### Create a network


```c++
auto new_net = std::make_shared<torch::Sequential>();
new_net->add(std::make_shared<torch::Conv2d>(3, 10, 3, 3));
new_net->add(std::make_shared<torch::BatchNorm2d>(10));
new_net->add(std::make_shared<torch::ReLU>());
new_net->add(std::make_shared<torch::Linear>(10, 3));
```
## Implemented layers

So far, these layers are available which respect the Pytorch's layers semantics which
can be found [here](http://pytorch.org/docs/0.1.12/nn.html#convolution-layers).


- [x] nn.Sequential
- [x] nn.Conv2d
- [x] nn.MaxPool2d
- [x] nn.AvgPool2d
- [x] nn.ReLU
- [x] nn.Linear
- [x] nn.SoftMax
- [x] nn.BatchNorm2d
- [ ] nn.Dropout2d
- [ ] nn.DataParallel
- [ ] nn.AdaptiveMaxPool2d
- [ ] nn.Sigmoid
and others.

## Implemented models

### Imagenet models

All models were converted from [pytorch/vision](https://github.com/pytorch/vision) and checked for
correctness.

- [x] Resnet-18
- [x] Resnet-34
- [x] Resnet-50
- [x] Resnet-101
- [x] Resnet-150
- [x] Resnet-152
- [ ] All VGG models
- [ ] All Densenet models
- [ ] All Inception models
- [ ] All squeezenet models
- [ ] Alexnet

### Segmentation PASCAL VOC 

All models were converted from [this repository](https://github.com/warmspringwinds/dense-ai) and checked for
correctness.

- [x] Resnet-18-8S
- [x] Resnet-34-8S
- [ ] Resnet-50-8S
- [ ] Resnet-101-8S
- [ ] Resnet-152-8S
- [x] FCN-32s
- [ ] FCN-16s
- [ ] FCN-8s


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
interface for building neural networks and inference (so far only forward pass is supported)
inspired by [cunnproduction](https://github.com/szagoruyko/cunnproduction) library. To install it, follow
these steps:

0. ```mkdir build; cd build; cmake-gui ..``` and specify ```CUDA_TOOLKIT_ROOT_DIR```.
1. ```make```
2. ```cd ../``` -- return back

### Problems with the build

If you face any problems or some steps are not clear, please open an issue. Note: every time you enter the ```cmake-gui```
press ```configure``` first, then specify your ```CUDA``` path and then press ```generate```, after that you can build.
