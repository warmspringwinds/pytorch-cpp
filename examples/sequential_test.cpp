/*
Example shows how to define an architecture, visualize it later on
using std tools, and get a forward pass from that model.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <sstream>

#define TENSOR_DEFAULT_TYPE CPU(kFloat)

using namespace at; // assumed in the following

namespace torch 
{

   /*
    * Abstract class as nn.Module
    */

   class Module
   {

      public:

        typedef std::shared_ptr<Module> Ptr;

        Module() {};

        ~Module() {};

        virtual Tensor forward(Tensor input) = 0;

        virtual const std::string tostring() const { return std::string("name not defined"); }

   };


   /*
    * nn.Sequential
    */

   class Sequential : public Module 
   {
      public:

         typedef std::shared_ptr<Sequential> Ptr;

         Sequential() {};

         ~Sequential() {};

         Tensor forward(Tensor input)
         {
            Tensor out = input;

            for(auto& it: modules)
            {
               out = it->forward(out);
            }

            return out;
         }

         
         const std::string tostring() const
         {

            std::stringstream s;

            s << "nn.Sequential {\n";

            int counter = 1;

            for(auto &it: modules)
            {

               s << "  (" << counter++ << ") " <<  it->tostring() << std::endl;
            }

            s << "}\n";
            
            return s.str();
         }

         Module::Ptr get(int i) const { return modules[i];  }
           
         void add(Module::Ptr module)
         {

            modules.push_back(module);
         }
         

         std::vector<Module::Ptr> modules;
   };


   /*
    * nn.ReLU
    */

   class ReLU : public Module
   {
      public:

         ReLU() {};
         ~ReLU() {};

         Tensor forward(Tensor input) 
         { 
            Threshold_updateOutput(input, input, 0, 0, true) ;
              return input; 
         };

         const std::string tostring() const { return std::string("nn.ReLU"); }
   };


   class Conv2d : public Module
   {

      public:

         Tensor convolution_weight;
         Tensor bias_weight;

         Tensor finput;
         Tensor fgradInput;
         Tensor ones;
         Tensor columns;

          int in_channels;
          int out_channels;
          int kernel_width;
          int kernel_height;
          int stride_width;
          int stride_height;
          int dilation_width;
          int dilation_height;
          int padding_width;
          int padding_height;
          int groups;
          int bias;
          bool dilated;

          Conv2d( int in_channels,
                  int out_channels,
                  int kernel_width,
                  int kernel_height,
                  int stride_width=1,
                  int stride_height=1,
                  int padding_width=0,
                  int padding_height=0,
                  int dilation_width=1,
                  int dilation_height=1,
                  int groups=1,
                  int bias=true) :

                in_channels(in_channels),
                out_channels(out_channels),
                kernel_width(kernel_width),
                kernel_height(kernel_height),
                stride_width(stride_width),
                stride_height(stride_height),
                padding_width(padding_width),
                padding_height(padding_height),
                dilation_width(dilation_width),
                dilation_height(dilation_height),
                groups(groups),
                bias(bias)
          {

            // Initialize weights here

            convolution_weight = TENSOR_DEFAULT_TYPE.zeros({out_channels, in_channels, kernel_width, kernel_height});
            bias_weight = TENSOR_DEFAULT_TYPE.zeros({out_channels});

            // These variables are not needed for forward inferece,
            // but we need them in order to call an underlying C
            // function. Later they will be used for backward pass
            finput = TENSOR_DEFAULT_TYPE.tensor();
            fgradInput = TENSOR_DEFAULT_TYPE.tensor();

            ones = TENSOR_DEFAULT_TYPE.tensor();
            columns = TENSOR_DEFAULT_TYPE.tensor();


            // There are separate functions for dilated and non-dilated convolutions
            dilated = false;

            if( (dilation_width > 1) || (dilation_height > 1) )
            {
              dilated = true;
            }

          };

          ~Conv2d() {};

          const std::string tostring() const
          {

            std::stringstream string_stream;

            string_stream << "nn.Conv2d( "
                          << "in_channels=" << std::to_string(in_channels) << " "
                          << "out_channels=" << std::to_string(out_channels) << " "
                          << "kernel_size=(" << std::to_string(kernel_width) << ", " << std::to_string(kernel_height) << ") "
                          << "stride=(" << std::to_string(stride_width) << ", " << std::to_string(stride_height) << ") "
                          << "padding=(" << std::to_string(padding_width) << ", " << std::to_string(padding_height) << ") "
                          << "dilation=(" << std::to_string(dilation_width) << ", " << std::to_string(dilation_height) << ") "
                          << "groups=" << std::to_string(groups) << " "
                          << "bias=" << std::to_string(bias) << " )";

            return string_stream.str();

          };

          Tensor forward(Tensor input) 
          { 

            Tensor output = TENSOR_DEFAULT_TYPE.tensor();

            if (dilated)
            {

              SpatialDilatedConvolution_updateOutput(input,
                                                     output,
                                                     convolution_weight,
                                                     bias_weight,
                                                     columns,
                                                     ones,
                                                     kernel_width,
                                                     kernel_height,
                                                     stride_width,
                                                     stride_height,
                                                     padding_width,
                                                     padding_height,
                                                     dilation_width,
                                                     dilation_height);
            }
            else
            {

              SpatialConvolutionMM_updateOutput(input,
                                              output,
                                              convolution_weight,
                                              bias_weight,
                                              finput,
                                              fgradInput,
                                              kernel_width,
                                              kernel_height,
                                              stride_width,
                                              stride_height,
                                              padding_width,
                                              padding_height);
            }

            
            return output; 
          };
    };

    class BatchNorm2d : public Module
    {
      public:

        Tensor gamma_weight;
        Tensor beta_bias_weight;
        Tensor running_mean;
        Tensor running_var;
        Tensor save_mean;
        Tensor save_std;
        int num_features;
        bool affine;
        bool training;
        double momentum;
        double eps;



        BatchNorm2d( int num_features,
                     double eps=1e-5,
                     double momentum=0.1,
                     bool affine=true,
                     bool training=false) :

                     num_features(num_features),
                     eps(eps),
                     momentum(momentum),
                     affine(affine),
                     training(training)
                     
        {

          // Initialize weights here

          // Ones initialization is temporarry -- just to avoid
          // division by zero during testing
          gamma_weight = TENSOR_DEFAULT_TYPE.ones(num_features);
          beta_bias_weight = TENSOR_DEFAULT_TYPE.zeros(num_features);
          running_mean = TENSOR_DEFAULT_TYPE.zeros(num_features);
          running_var = TENSOR_DEFAULT_TYPE.ones(num_features);

          // We don't recompute the mean and var during inference
          // So, some variables are initialized for possible future use case.
          save_mean = TENSOR_DEFAULT_TYPE.ones(num_features);
          save_std = TENSOR_DEFAULT_TYPE.ones(num_features);

        };

        ~BatchNorm2d() {};

        const std::string tostring() const
        {

          std::stringstream string_stream;

          string_stream << "nn.BatchNorm2d( "
                        << "num_features=" << std::to_string(num_features) << " "
                        << "eps=" << std::to_string(eps) << " "
                        << "momentum=" << std::to_string(momentum) << " )";

          return string_stream.str();

        };


        Tensor forward(Tensor input) 
        {

          Tensor output = TENSOR_DEFAULT_TYPE.tensor();

          BatchNormalization_updateOutput(input,
                                          output,
                                          gamma_weight,
                                          beta_bias_weight,
                                          running_mean,
                                          running_var,
                                          save_mean,
                                          save_std,
                                          training,
                                          momentum,
                                          eps);
          return output; 
        };
        


    };


}

int main()
{


   auto net = std::make_shared<torch::Sequential>();
   net->add( std::make_shared<torch::ReLU>() );
   net->add( std::make_shared<torch::Conv2d>(3, 10, 3, 3, 1, 1, 1, 1, 2, 2) );
   net->add( std::make_shared<torch::ReLU>() );
   net->add( std::make_shared<torch::BatchNorm2d>(10) );

   //Visualize the architecture
   std::cout << net->tostring() << std::endl;

   Tensor dummy_input = TENSOR_DEFAULT_TYPE.ones({1, 3, 5, 5}) * (-10);

   Tensor output = net->forward(dummy_input);

   // Print out the results -- should be zeros, because we applied RELU
   std::cout << output << std::endl;


   // Overall output:

   //    nn.Sequential {
   //   (1) nn.ReLU
   //   (2) nn.Conv2d( in_channels=3 out_channels=10 kernel_size=(3, 3) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=1 )
   //   (3) nn.ReLU
   // }

   // (1,1,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0

   // (1,2,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0

   // (1,3,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0

   // (1,4,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0

   // (1,5,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0

   // (1,6,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0

   // (1,7,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0

   // (1,8,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0

   // (1,9,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0

   // (1,10,.,.) = 
   //   0  0  0
   //   0  0  0
   //   0  0  0
   // [ CPUFloatTensor{1,10,3,3} ]


   return 0;
}