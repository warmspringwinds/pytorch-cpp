/*
Example shows a real-time segmentation of human class from PASCAL VOC.
The network ouputs probabilities of each pixels belonging to the human class.
These probabilities are later on are used as a transparancy mask for the input image.
The final fused image is displayed in the window of the application.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <map>

#include <opencv2/opencv.hpp>

#include <pytorch.cpp>

using namespace at;

using std::map;
using std::string;

using namespace cv;

int main()
{

 
  // Structure the project in a better way

  // Add a correct linking to Opencv on the local machine

  // Get the build running on laptop for demo

  // upload all the transferred models


  // -----

  // * Should we convert the renset 50 and 101?
  // * we don't have any segmentatin models trained using them
  // * maybe only to make the framework more complete? (check)

  // * Make the classification demo?
  // * need to put a softmax on top -- should be very easy
  // * need a dict with number --> class name mapping (check)

  // * Structure the whole project (check)

  // * write docs on how to build it
  
  // * write missing parts -- good for future contributions

  // * Write the dataloaders for the new surgical datasets

  // * start the training


  auto net = torch::resnet34_8s_pascal_voc();

  net->load_weights("../resnet34_fcn_pascal.h5");
  net->cuda();

  VideoCapture cap(0); // open the default camera

  if(!cap.isOpened())  // check if we succeeded
      return -1;

  Mat frame;
  
  for(;;)
  { 

    cap >> frame;
        
    // BGR to RGB which is what our network was trained on
    cvtColor(frame, frame, COLOR_BGR2RGB);

    // Resizing while preserving aspect ratio, comment out to run
    // it on the whole input image.
    resize(frame, frame, Size(0, 0), 0.5, 0.5, INTER_LINEAR);
      
    // Outputs height x width x 3 tensor converted from Opencv's Mat with 0-255 values
    // and convert to 0-1 range
    auto image_tensor = torch::convert_opencv_mat_image_to_tensor(frame).toType(CPU(kFloat)) / 255;

    auto output_height = image_tensor.size(0);
    auto output_width = image_tensor.size(1);

    // Reshape image into 1 x 3 x height x width
    auto image_batch_tensor = torch::convert_image_to_batch(image_tensor);

    // Subtract the mean and divide by standart deivation
    auto image_batch_normalized_tensor = torch::preprocess_batch(image_batch_tensor);

    auto input_tensor_gpu = image_batch_normalized_tensor.toBackend(Backend::CUDA);

    auto full_prediction = net->forward(input_tensor_gpu);

    // This is necessary to correctly apply softmax,
    // last dimension should represent logits
    auto full_prediction_flattned = full_prediction.squeeze(0)
                                                   .view({21, -1})
                                                   .transpose(0, 1);

    // Converting logits to probabilities                                               
    auto softmaxed = torch::softmax(full_prediction_flattned).transpose(0, 1);

    // 15 is a class for a person
    auto layer = softmaxed[15].contiguous().view({output_height, output_width, 1}).toBackend(Backend::CPU);

    // Fuse the prediction probabilities and the actual image to form a masked image.
    auto masked_image = ( image_tensor  * layer.expand({output_height, output_width, 3}) ) * 255 ;

    // A function to convert Tensor to a Mat
    auto layer_cpu = masked_image.toType(CPU(kByte));

    auto converted = Mat(output_height, output_width, CV_8UC3, layer_cpu.data_ptr());

    // OpenCV wants BGR not RGB
    cvtColor(converted, converted, COLOR_RGB2BGR);

    imshow("Masked image", converted);

    if(waitKey(30) >= 0 ) break;
  }

  
  return 0;
}