/*
Example shows a real-time classification. The name of the most probable class
is printed over the image.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <map>

#include <opencv2/opencv.hpp>

#include <pytorch.cpp>
#include <imagenet_classes.cpp>

using namespace at;

using std::map;
using std::string;
using std::tie;

using namespace cv;

int main()
{

  auto net = torch::resnet50_imagenet();

  net->load_weights("../resnet50_imagenet.h5");
  net->cuda();

  VideoCapture cap(0); // open the default camera


  if(!cap.isOpened())  // check if we succeeded
      return -1;

  Mat frame;
  Mat resized_img;
  
  for(;;)
  { 

    cap.read(frame);

    // BGR to RGB which is what our network was trained on
    cvtColor(frame, resized_img, COLOR_BGR2RGB);

    // Should be resized while preserving an aspect ratio -- we didn't do it here
    // consider doing that to improve results
    resize(frame, resized_img, Size(224, 224));
      
    // Outputs height x width x 3 tensor converted from Opencv's Mat with 0-255 values
    // and convert to 0-1 range
    auto image_tensor = torch::convert_opencv_mat_image_to_tensor(resized_img).toType(CPU(kFloat)) / 255;

    // Reshape image into 1 x 3 x height x width
    auto image_batch_tensor = torch::convert_image_to_batch(image_tensor);

    auto image_batch_normalized_tensor = torch::preprocess_batch(image_batch_tensor);

    auto input_tensor_gpu = image_batch_normalized_tensor.toBackend(Backend::CUDA);

    auto full_prediction = net->forward(input_tensor_gpu);

    auto softmaxed = torch::softmax(full_prediction);

    Tensor top_probability_indexes;
    Tensor top_probabilies;

    tie(top_probabilies, top_probability_indexes) = topk(softmaxed, 5, 1, true);

    top_probability_indexes = top_probability_indexes.toBackend(Backend::CPU).view({-1});

    auto accessor = top_probability_indexes.accessor<long,1>();

    putText(frame, imagenet_classes[ accessor[0]  ], cvPoint(30,30), 
    FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);

    imshow("Masked image", frame);

    if(waitKey(30) >= 0 ) break;
  }

}