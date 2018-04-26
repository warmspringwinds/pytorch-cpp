/*
Example shows how to measure the average execution time spent on one image.
Here we test resnet 18 with the output stride of 8 which shows execution time of 25 ms
per frame of size 512x512 on average.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <map>
#include <pytorch.cpp>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace at;

using std::map;
using std::string;

using namespace std;
using namespace std::chrono;


int main()
{
  
  // The reason we do a first run before measuring the time is
  // because first run is slow and doesn't represent the actual speed.
  auto net = torch::resnet18_8s_pascal_voc();
  
  net->cuda();

  Tensor dummy_input = CUDA(kFloat).ones({1, 3, 512, 512});
  
  high_resolution_clock::time_point t1;
  high_resolution_clock::time_point t2;
  
  cudaDeviceSynchronize();
  
  t1 = high_resolution_clock::now();
    
  auto result = net->forward(dummy_input);
  
  cudaDeviceSynchronize();
    
  t2 = high_resolution_clock::now();
  
  auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
  
  // Now running in a loop and getting an average result.
    
  int number_of_iterations = 100;
  int overall_miliseconds_count = 0;
    
  for (int i = 0; i < number_of_iterations; ++i)
  {
      
      t1 = high_resolution_clock::now();
  
      result = net->forward(dummy_input);
      
      cudaDeviceSynchronize();
    
      t2 = high_resolution_clock::now();
  
      duration = duration_cast<milliseconds>( t2 - t1 ).count();
      
      overall_miliseconds_count += duration;
      
  }
    
  cout << "Average execution time: " << overall_miliseconds_count / float(number_of_iterations) << " ms" << endl;
    
  // On our system it outpts: 25ms per frame.
    
  return 0;

}