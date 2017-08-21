/*
Example shows how to run a resnet 50 imagenet-trained classification
model on a dummy input and save it to an hdf5 file. This output can be
later on compared to the output acquired from pytorch in a provided .ipynb
notebook -- results differ no more than 10^{-5}.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <map>

#include <pytorch.cpp>

using namespace at;

using std::map;
using std::string;


int main()
{

  auto net = torch::resnet50_imagenet();

  net->load_weights("../resnet50_imagenet.h5");
  net->cuda();

  Tensor dummy_input = CUDA(kFloat).ones({1, 3, 224, 224});

  auto result = net->forward(dummy_input);

  map<string, Tensor> dict;

  dict["main"] = result.toBackend(Backend::CPU);

  torch::save("resnet50_output.h5", dict);

  return 0;
}