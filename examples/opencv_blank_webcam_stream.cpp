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

 
  VideoCapture cap(0); // open the default camera

  if(!cap.isOpened())  // check if we succeeded
      return -1;

  Mat frame;
  
  for(;;)
  { 

    cap >> frame;
        
    imshow("Masked image", frame);

    if(waitKey(30) >= 0 ) break;
  }

  
  return 0;
}