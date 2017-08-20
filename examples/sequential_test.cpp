/*
Example shows how to define an architecture, visualize it later on
using std tools, and get a forward pass from that model.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <sstream>
#include <map>
#include "H5Cpp.h"

#include <opencv2/opencv.hpp>

#define TENSOR_DEFAULT_TYPE CPU(kFloat)

using namespace at;


using std::map;
using std::string;
using std::vector;
using std::pair;
using std::shared_ptr;
using std::make_shared;
using std::cout;
using std::endl;
using std::tie;

using namespace cv;


namespace torch
{   


   map<string, Tensor> load(string hdf5_filename);
   void save( string hdf5_filename, map<string, Tensor> dict_to_write );

   class Module
   {

      public:

        // Sequential module needs the counter
        // as names of submodules are not provided
        // sometimes.
        int submodule_counter;

        Module() : submodule_counter(0) {};

        ~Module() {};

        // We will use pointer to other modules a lot
        // This is done to automatically handle deallocation of created
        // module objects
        typedef shared_ptr<Module> Ptr;

        virtual Tensor forward(Tensor input) { return input; };

        string module_name = "Module";

        // This function gets overwritten
        // for the leafnodes like Conv2d, AvgPool2d and so on
        virtual string tostring(int indentation_level=0)
        {

          std::stringstream s;

          string indentation = string(indentation_level, ' ');

          s << indentation << module_name << " (" << std::endl;

          for(auto name_module_pair: modules)
          {

              s << indentation << " (" << name_module_pair.first << ") "
                << name_module_pair.second->tostring(indentation_level + 1) << std::endl;
          }

          s << indentation << ")" << std::endl;

          return s.str();

        }

        // vector<pair<string, Ptr>> because we want to emulate
        // the ordered dict this way, meaning that elements
        // are stored in the same order they were added

        // Like in Pytorch each module stores the modules that it uses
        vector<pair<string, Ptr>> modules;

        // And parameters that are explicitly used by the current module
        map<string, Tensor> parameters;

        // Plus buffers which are meant to store running mean and var for batchnorm layers
        map<string, Tensor> buffers;

        // We store parameter related to gradient computation here and other
        // tensors so far
        // TODO: some members of grads are not related to gradient computation
        //       and were put there temporary -- put them in a more relevant container.
        map<string, Tensor> grads;
        
        // A function to add another modules inside current module
        // Acts as Pytorch's Module.add_module() function
        void add_module(string module_name, Module::Ptr module)
        {


          modules.push_back(pair<string, Module::Ptr>(module_name, module));
        }



        // Sometimes, when modules are being added, not all of them
        // have weights, like RELU. In this case the weights can be
        // numerated out of order. For example:
        // net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        // net.state_dict().keys()
        // output: ['0.weight', '0.bias', '2.weight', '2.bias']

        // Equivalent behaviour will be seen with the add() function
        // described below: if relu is added, the counter for weights will
        // be increased.

        void add(Module::Ptr module)
        {

          string module_name = std::to_string(submodule_counter);

          add_module(module_name, module);

          submodule_counter++;
        }


        map<string, Tensor> state_dict( map<string, Tensor> & destination,
                                        string prefix="")
        {

          // TODO: add another function that will not accept any parameters
          // and just return the state_dict()

          for(auto name_parameter_pair: parameters)
          {

            // Check if the parameter defined -- for example if we don't use bias
            // in the convolution, the bias weight will be undefined.
            // We need this in order to match the state_dict() function of Pytorch
            if(name_parameter_pair.second.defined())
            {


              destination[prefix + name_parameter_pair.first] = name_parameter_pair.second;
            }
          }

          for(auto name_buffer_pair: buffers)
          {


            destination[prefix + name_buffer_pair.first] = name_buffer_pair.second;
          }

          for(auto name_module_pair: modules)
          {

            name_module_pair.second->state_dict(destination, prefix + name_module_pair.first + '.');
          }

          return destination;

        }


        template<typename Func>
        void apply(Func closure) 
        {


            for(auto name_parameter_pair: parameters)
            {

              if(name_parameter_pair.second.defined())
              {
                // maybe catch if it is undefined here
                parameters[name_parameter_pair.first] = closure(name_parameter_pair.second);
              }
            }

            for(auto name_buffer_pair: buffers)
            {

              buffers[name_buffer_pair.first] = closure(name_buffer_pair.second);
            }

            for(auto name_grad_pair: grads)
            {

              grads[name_grad_pair.first] = closure(name_grad_pair.second);
            }

            for(auto name_module_pair: modules)
            {

              name_module_pair.second->apply(closure);
            }
        }

        void cuda()
        {

          // Transfer each tensor to GPU
          this->apply([](Tensor & tensor) {

            return tensor.toBackend(Backend::CUDA);

          });

        }

        void cpu()
        {

          // Transfer each tensor to CPU
          this->apply([](Tensor & tensor) {

            return tensor.toBackend(Backend::CPU);

          });
          
        }

        void save_weights(string hdf5_filename)
        {

          map<string, Tensor> model_state_dict;

          this->state_dict(model_state_dict);

          save(hdf5_filename, model_state_dict);
        }

        
        void load_weights(string hdf5_filename)
        {

    
          // TODO:
          // (1) Add check to make sure that the network is on cpu
          //     before loading weights
          // (2) Add support for not float. So far only works with
          //     float weights only.

          map<string, Tensor> model_state_dict;
          map<string, Tensor> checkpoint_dict;

          this->state_dict(model_state_dict);
          checkpoint_dict = load(hdf5_filename);

          // Compare model_state_dict -> checkpoint_dict keys consistency

          for(auto name_tensor_pair : model_state_dict)
          {

            if(checkpoint_dict.count(name_tensor_pair.first) != 1)
            {

              cout << "WARNING: model requires parameter ('" << name_tensor_pair.first << "') "
                   << "which is not present in the checkpoint file. Using model's default." << endl;
            }
          }

          // Compare checkpoint_dict -> model_state_dict keys consistency

          for(auto name_tensor_pair : checkpoint_dict)
          {

            if(model_state_dict.count(name_tensor_pair.first) != 1)
            {

              cout << "WARNING: checkpoint file contains parameter ('" << name_tensor_pair.first << "') "
                   << "which is is not required by the model. The parameter is not used." << endl;
            }
          }

          for(auto name_tensor_pair : model_state_dict)
          {

            if(checkpoint_dict.count(name_tensor_pair.first) == 1)
            {

              // Copy in-place
              name_tensor_pair.second.copy_(checkpoint_dict[name_tensor_pair.first]);
            }
          }

        }

   };


   class Sequential : public Module 
   {
      public:

        Sequential() 
        {

          module_name = "Sequential";
        };

        ~Sequential() { };

        // Forward for sequential block makes forward pass
        // for each submodule and passed it to the next one
        Tensor forward(Tensor input)
        {
          Tensor out = input;

          for(auto name_module_pair: modules)
          {
             out = name_module_pair.second->forward(out);
          }

          return out;
        }


        Module::Ptr get(int i) const { return modules[i].second;  }

   };


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


        string tostring(int indentation_level=0)
        { 

          string indentation = string(indentation_level, ' ');

          return indentation + std::string("ReLU"); 
        }
   };


   class Conv2d : public Module
   {

      public:

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

            // Register "wight" as a parameter in order to be able to
            // restore it from a file later on
            parameters["weight"] = TENSOR_DEFAULT_TYPE.zeros({out_channels,
                                                              in_channels,
                                                              kernel_width,
                                                              kernel_height});


            // Check if we need bias for our convolution
            if(bias)
            {
              parameters["bias"] = TENSOR_DEFAULT_TYPE.zeros({out_channels});
            }
            else
            {
              
              // Doesn't work with TENSOR_DEFAULT_TYPE.tensor();,
              // This is why we use Tensor()
              parameters["bias"] = Tensor(); 
            }

            // These variables are not needed for forward inferece,
            // but we need them in order to call an underlying C
            // function. Later they will be used for backward pass

            grads["finput"] = TENSOR_DEFAULT_TYPE.tensor();
            grads["fgradInput"] = TENSOR_DEFAULT_TYPE.tensor(); 

            // These variables depend on # of groups, so far only
            // one group is supported. Needs to be changed to tensor_list
            // in order to support multiple groups.
            grads["ones"] = TENSOR_DEFAULT_TYPE.tensor(); 
            grads["columns"] = TENSOR_DEFAULT_TYPE.tensor();

            // There are separate functions for dilated and non-dilated convolutions
            dilated = false;

            if( (dilation_width > 1) || (dilation_height > 1) )
            {
              dilated = true;
            }

          };

          ~Conv2d() {};

          
          string tostring(int indentation_level=0)
          {

            std::stringstream string_stream;

            string indentation = string(indentation_level, ' ');

            string_stream << indentation << "Conv2d( "
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

            Tensor output = input.type().tensor();

            if (dilated)
            {

              SpatialDilatedConvolution_updateOutput(input,
                                                     output,
                                                     parameters["weight"],
                                                     parameters["bias"],
                                                     grads["columns"],
                                                     grads["ones"],
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
                                                parameters["weight"],
                                                parameters["bias"],
                                                grads["finput"],
                                                grads["fgradInput"],
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
          parameters["weight"] = TENSOR_DEFAULT_TYPE.ones(num_features);
          parameters["bias"] = TENSOR_DEFAULT_TYPE.zeros(num_features);

          buffers["running_mean"] = TENSOR_DEFAULT_TYPE.zeros(num_features);
          buffers["running_var"] = TENSOR_DEFAULT_TYPE.ones(num_features);

          // We don't recompute the mean and var during inference
          // So, some variables are initialized for possible future use case.
          grads["save_mean"] = TENSOR_DEFAULT_TYPE.ones(num_features);
          grads["save_std"] = TENSOR_DEFAULT_TYPE.ones(num_features);

        };

        ~BatchNorm2d() {};

        string tostring(int indentation_level=0)
        {

          std::stringstream string_stream;

          string indentation = string(indentation_level, ' ');

          string_stream << indentation
                        << "BatchNorm2d( "
                        << "num_features=" << std::to_string(num_features) << " "
                        << "eps=" << std::to_string(eps) << " "
                        << "momentum=" << std::to_string(momentum) << " )";

          return string_stream.str();

        };


        Tensor forward(Tensor input) 
        {

          Tensor output = input.type().tensor();

          BatchNormalization_updateOutput(input,
                                          output,
                                          parameters["weight"],
                                          parameters["bias"],
                                          buffers["running_mean"],
                                          buffers["running_var"],
                                          grads["save_mean"],
                                          grads["save_std"],
                                          training,
                                          momentum,
                                          eps);
          return output; 
        };
        
    };


    // TODO: move this thing out in a separate logical unit: models/resnet

    // Helper functions for a 3 by 3 convolution without bias
    // Which is used in every resnet architecture.
    Tensor compute_full_padding_for_dilated_conv(Tensor kernel_size, int dilation=1)
    {

      // Convert IntList to Tensor to be able to use element-wise operations
      Tensor kernel_size_tensor = kernel_size.toType(CPU(kFloat));
                                  
      // Compute the actual kernel size after dilation
      auto actual_kernel_size = (kernel_size_tensor - 1) * (dilation - 1) + kernel_size_tensor;

      // Compute the padding size in order to achieve the 'full padding' mode
      auto full_padding = (actual_kernel_size / 2).floor_()
                                                  .toType(CPU(kInt));
                                                  
      return full_padding;
    };

    Module::Ptr conv3x3(int in_planes, int out_planes, int stride=1, int dilation=1)
    {

      // {3, 3} tuple in tensor form.
      // We need this because next function accepts Tensor
      Tensor kernel_size = CPU(kInt).tensor({2})
                                    .fill_(3);

      Tensor padding = compute_full_padding_for_dilated_conv(kernel_size, dilation);

      auto padding_accessor = padding.accessor<int,1>(); 

      return std::make_shared<Conv2d>(in_planes,
                                      out_planes,
                                      3, 3,
                                      stride, stride,
                                      padding_accessor[0], padding_accessor[1],
                                      dilation, dilation,
                                      1, false);
    };


    Module::Ptr resnet_base_conv7x7()
    {

      return make_shared<Conv2d>(3,      /* in_planes */
                                 64,     /* out_planes */
                                 7,      /* kernel_w */
                                 7,      /* kernel_h */
                                 2,      /* stride_w */
                                 2,      /* stride_h */
                                 3,      /* padding_w */
                                 3,      /* padding_h */
                                 1,      /* dilation_w */
                                 1,      /* dilation_h */
                                 1,      /* groups */
                                 false); /* bias */
    }


    class MaxPool2d : public Module
    {
      public:

        Tensor indices;

        bool ceil_mode;
        int kernel_width;
        int kernel_height;
        int stride_width;
        int stride_height;
        int padding_width;
        int padding_height;

       
        MaxPool2d(int kernel_width,
                  int kernel_height,
                  int stride_width=1,
                  int stride_height=1,
                  int padding_width=0,
                  int padding_height=0,
                  bool ceil_mode=false) :

                  kernel_width(kernel_width),
                  kernel_height(kernel_height),
                  stride_width(stride_width),
                  stride_height(stride_height),
                  padding_width(padding_width),
                  padding_height(padding_height),
                  ceil_mode(ceil_mode)
        {

          // TODO: so far this one is hardcoded.
          // Change to make it gpu or cpu depending
          // on the network placement
          grads["indices"] = CPU(kLong).tensor(); 
        };


        ~MaxPool2d() {};

        Tensor forward(Tensor input)
        {

          Tensor output = input.type().tensor();

          SpatialMaxPooling_updateOutput(input,
                                         output,
                                         grads["indices"],
                                         kernel_width,
                                         kernel_width,
                                         stride_width,
                                         stride_height,
                                         padding_width,
                                         padding_height,
                                         ceil_mode);

          return output; 
        };

        string tostring(int indentation_level=0)
        {

          std::stringstream string_stream;

          string indentation = string(indentation_level, ' ');

          string_stream << indentation
                        << "MaxPool2d( "
                        << "kernel_size=(" << std::to_string(kernel_width) << ", " << std::to_string(kernel_height) << ") "
                        << "stride=(" << std::to_string(stride_width) << ", " << std::to_string(stride_height) << ") "
                        << "padding=(" << std::to_string(padding_width) << ", " << std::to_string(padding_height) << ") )";

          return string_stream.str();

        };
   };


   class AvgPool2d : public Module
   {
      public:

        bool ceil_mode;
        bool count_include_pad;
        int kernel_width;
        int kernel_height;
        int stride_width;
        int stride_height;
        int padding_width;
        int padding_height;

       
        AvgPool2d(int kernel_width,
                  int kernel_height,
                  int stride_width=1,
                  int stride_height=1,
                  int padding_width=0,
                  int padding_height=0,
                  bool ceil_mode=false,
                  bool count_include_pad=true) :

                  kernel_width(kernel_width),
                  kernel_height(kernel_height),
                  stride_width(stride_width),
                  stride_height(stride_height),
                  padding_width(padding_width),
                  padding_height(padding_height),
                  ceil_mode(ceil_mode),
                  count_include_pad(count_include_pad)
        { };


        ~AvgPool2d() {};

        Tensor forward(Tensor input)
        {

          Tensor output = input.type().tensor();

          SpatialAveragePooling_updateOutput(input,
                                             output,
                                             kernel_width,
                                             kernel_height,
                                             stride_width,
                                             stride_height,
                                             padding_width,
                                             padding_height,
                                             ceil_mode,
                                             count_include_pad);

          return output; 
        };

        string tostring(int indentation_level=0)
        {

          std::stringstream string_stream;

          string indentation = string(indentation_level, ' ');

          string_stream << indentation
                        << "AvgPool2d( "
                        << "kernel_size=(" << std::to_string(kernel_width) << ", " << std::to_string(kernel_height) << ") "
                        << "stride=(" << std::to_string(stride_width) << ", " << std::to_string(stride_height) << ") "
                        << "padding=(" << std::to_string(padding_width) << ", " << std::to_string(padding_height) << ") )"; 

          return string_stream.str();

        };
   };


   class Linear : public Module
   {

      public:


          int in_features;
          int out_features;
          bool bias;

          Linear( int in_features,
                  int out_features,
                  bool bias=true) :

                in_features(in_features),
                out_features(out_features),
                bias(bias)
          {

            // Initialize weights here

            parameters["weight"] = TENSOR_DEFAULT_TYPE.zeros({out_features, in_features});

            // Check if we need bias for our convolution
            if(bias)
            {

              parameters["bias"] = TENSOR_DEFAULT_TYPE.ones({out_features});
            }
            else
            {

              // don't know why this works yet, doesn't work with TENSOR_DEFAULT_TYPE.tensor();
              parameters["bias"] = Tensor();
            }

          };

          ~Linear() {};

          string tostring(int indentation_level=0)
          {

            std::stringstream string_stream;

            string indentation = string(indentation_level, ' ');

            string_stream << indentation
                          << "nn.Linear( "
                          << "in_features=" << std::to_string(in_features) << " "
                          << "out_features=" << std::to_string(out_features) << " "
                          << "bias=" << std::to_string(bias) << " )";

            return string_stream.str();

          };

          Tensor forward(Tensor input)
          {

            // https://github.com/pytorch/pytorch/blob/49ec984c406e67107aae2891d24c8839b7dc7c33/torch/nn/_functions/linear.py

            Tensor output = input.type().zeros({input.size(0), parameters["weight"].size(0)});

            output.addmm_(0, 1, input, parameters["weight"].t());
            
            if(bias)
            {
              // TODO: check if in-place resize affects the result
              output.add_(parameters["bias"].expand({output.size(0), output.size(1)}));  
            }
            
            return output; 
          };
    };



    class BasicBlock : public Module
    {

      public:

        static const int expansion = 1;

        int stride;
        Module::Ptr conv1;
        Module::Ptr bn1;
        Module::Ptr relu;
        Module::Ptr conv2;
        Module::Ptr bn2;
        Module::Ptr downsample;

        // Make a standart value
        BasicBlock(int inplanes, int planes, int stride=1, int dilation=1, Module::Ptr downsample=nullptr)
        {

          conv1 = conv3x3(inplanes, planes, stride, dilation);
          bn1 = std::make_shared<BatchNorm2d>(planes);
          relu = std::make_shared<ReLU>();
          conv2 = conv3x3(planes, planes, 1, dilation);
          bn2 = std::make_shared<BatchNorm2d>(planes);

          // This doesn't work
          // downsample = downsample because
          // the argument gets assigned instead of a class member,
          // Should probably change the name of the member and argument
          // to be different
          this->downsample = downsample;

          stride = stride;

          add_module("conv1", conv1);
          add_module("bn1", bn1);
          add_module("conv2", conv2);
          add_module("bn2", bn2);

          if( downsample != nullptr )
          {

            add_module("downsample", downsample);
          }

          module_name = "BasicBlock";

        };

        ~BasicBlock() {};

        Tensor forward(Tensor input)
        {

          // This is done in case we don't have the
          // downsample module
          Tensor residual = input;
          Tensor out;

          out = conv1->forward(input);
          out = bn1->forward(out);
          out = relu->forward(out);
          out = conv2->forward(out);
          out = bn2->forward(out);

          if(downsample != nullptr)
          {
        
            residual = downsample->forward(input);
          }

          out += residual;
          out = relu->forward(out);

          return out;
        }

    };


    template <class BlockType>
    class ResNet : public Module
    {

      public:

        int output_stride;
        int in_planes;

        // Helper variables to help track
        // dilation factor and output stride
        int current_stride;
        int current_dilation;

        // Variables realted to the type of architecture.
        // Image Segmentation models don't have average pool
        // layer and Linear layers are converted to 1x1 convolution
        bool fully_conv;
        bool remove_avg_pool;

        Module::Ptr conv1;
        Module::Ptr bn1;
        Module::Ptr relu;
        Module::Ptr maxpool;
        Module::Ptr layer1;
        Module::Ptr layer2;
        Module::Ptr layer3;
        Module::Ptr layer4;
        Module::Ptr avgpool;
        Module::Ptr fc;

        // block, layers, num_classes=1000):
        ResNet(IntList layers,
               int num_classes=1000,
               bool fully_conv=false,
               bool remove_avg_pool=false,
               int output_stride=32) :

        // First depth input is the same for all resnet models
        in_planes(64),
        output_stride(output_stride),
        fully_conv(fully_conv),
        remove_avg_pool(remove_avg_pool)

        {

          // Stride is four after first convolution and maxpool layer.
          // We use this class member to track current output stride in make_layer()
          current_stride = 4;

          // Dilation hasn't been applied after convolution and maxpool layer.
          // We use this class member to track dilation factor in make_layer()
          current_dilation = 1;

          conv1 = resnet_base_conv7x7();
          bn1 = std::make_shared<BatchNorm2d>(64);
          relu = std::make_shared<ReLU>();
          // Kernel size: 3, Stride: 2, Padding, 1 -- full padding 
          maxpool = std::make_shared<MaxPool2d>(3, 3, 2, 2, 1, 1);

          layer1 = make_layer(64, layers[0], 1);
          layer2 = make_layer(128, layers[1], 2);
          layer3 = make_layer(256, layers[2], 2);
          layer4 = make_layer(512, layers[3], 2);

          avgpool = std::make_shared<AvgPool2d>(7, 7);
          fc = std::make_shared<Linear>(512 * BlockType::expansion, num_classes);

          if(fully_conv)
          {

            // Average pooling with 'full padding' mode
            avgpool = std::make_shared<AvgPool2d>(7, 7,
                                                  1, 1,
                                                  3, 3 );

            // 1x1 Convolution -- Convolutionalized Linear Layer
            fc = std::make_shared<Conv2d>(512 * BlockType::expansion,
                                          num_classes,
                                          1, 1);
          }

          add_module("conv1", conv1);
          add_module("bn1", bn1);
          add_module("relu", relu);

          add_module("maxpool", maxpool);

          add_module("layer1", layer1);
          add_module("layer2", layer2);
          add_module("layer3", layer3);
          add_module("layer4", layer4);

          add_module("avgpool", avgpool);

          add_module("fc", fc);

          module_name = "ResNet";

        }

        Tensor forward(Tensor input)
        {

          Tensor output = input.type().tensor();

          output = conv1->forward(input);
          output = bn1->forward(output);
          output = relu->forward(output);
          output = maxpool->forward(output);

          output = layer1->forward(output);
          output = layer2->forward(output);
          output = layer3->forward(output);
          output = layer4->forward(output);

          if(!remove_avg_pool)
          {

            output = avgpool->forward(output);
          }

          if(!fully_conv)
          {

            // Flatten the output in order to apply linear layer
            output = output.view({output.size(0), -1});
          }

          output = fc->forward(output);

          return output;

        }

        
        Module::Ptr make_layer(int planes, int blocks, int stride)
        {

          auto new_layer = std::make_shared<torch::Sequential>();

          Module::Ptr downsample = nullptr;

          // Check if we need to downsample
          if(stride != 1 || in_planes != planes * BlockType::expansion)
          {

            // See if we already achieved desired output stride
            if(current_stride == output_stride)
            {

              // If so, replace subsampling with dilation to preserve
              // current spatial resolution
              current_dilation = current_dilation * stride;
              stride = 1;
            }
            else
            {

              // If not, we perform subsampling
              current_stride = current_stride * stride;
            }


            downsample = std::make_shared<torch::Sequential>();

            downsample->add( std::make_shared<torch::Conv2d>(in_planes,
                                                             planes * BlockType::expansion,
                                                             1, 1,
                                                             stride, stride,
                                                             0, 0,
                                                             1, 1,
                                                             1,
                                                             false) );

            downsample->add(std::make_shared<BatchNorm2d>(planes * BlockType::expansion));

          }

          auto first_block = std::make_shared<BlockType>(in_planes,
                                                         planes,
                                                         stride,
                                                         current_dilation,
                                                         downsample);
          new_layer->add(first_block);

          in_planes = planes * BlockType::expansion;

          for (int i = 0; i < blocks - 1; ++i)
          {
            
            new_layer->add(std::make_shared<BlockType>(in_planes,
                                                       planes,
                                                       1,
                                                       current_dilation));
          }

          return new_layer;

        }

    };


    Tensor preprocess_batch(Tensor input_batch)
    {

      // Subtracts mean and divides by std.
      // Important: image should be in a 0-1 range and not in 0-255

      // TODO: create a pull request to add broadcastable
      // operations

      auto mean_value = CPU(kFloat).ones({1, 3, 1, 1});

      mean_value[0][0][0][0] = 0.485f;
      mean_value[0][1][0][0] = 0.456f;
      mean_value[0][2][0][0] = 0.406f;

      // Broadcast the value
      auto mean_value_broadcasted = mean_value.expand(input_batch.sizes());

      auto std_value = CPU(kFloat).ones({1, 3, 1, 1});

      std_value[0][0][0][0] = 0.229f;
      std_value[0][1][0][0] = 0.224f;
      std_value[0][2][0][0] = 0.225f;

      auto std_value_broadcasted = std_value.expand(input_batch.sizes());

      return (input_batch - mean_value_broadcasted) / std_value_broadcasted;

    }

    vector<string> get_hdf5_file_keys(string hdf5_filename)
    {

      // We open and close hdf5 file here. It might be an overkill
      // as we can open the file once, read keys and read tensors outright,
      // but this way we also add a simple debugging function to be able to
      // easily get keys without dealing with HDF5 API directly.

      // Open the file
      H5::H5File file = H5::H5File(hdf5_filename, H5F_ACC_RDONLY);

      vector<string> names;

      // Define a closure to populate our names array
      auto closure = [] (hid_t loc_id, const char *name, const H5L_info_t *linfo, void *opdata) 
      {

        vector<string> * names_array_pointer = reinterpret_cast< vector<string> *>(opdata);

        names_array_pointer->push_back(string(name));

        return 0;
      };

      // Run our closure and populate array
      H5Literate(file.getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, closure, &names);

      file.close();

      return names;

    }

    map<string, Tensor> load(string hdf5_filename)
    {

      map<string, Tensor> tensor_dict;

      // use our get_names function
      vector<string> tensor_names = get_hdf5_file_keys(hdf5_filename);

      H5::H5File file = H5::H5File(hdf5_filename, H5F_ACC_RDONLY);

      // Array to store the shape of the current tensor
      hsize_t * dims_hsize_t;

      // We need this because one function can't accept hsize_t
      vector<long> dims_int;

      // Float buffer to intermediately store weights
      float * float_buffer;

      // 'Rank' of the tensor
      int ndims;

      // Number of elements in the current tensor
      hsize_t tensor_flattened_size;

      Tensor buffer_tensor;


      for (auto tensor_name: tensor_names)
      {

        dims_int.clear();

        // Open a 'dataset' which stores current tensor
        H5::DataSet current_dataset = file.openDataSet(tensor_name);

        // We can infer the sizes of a store tensor from H5::DataSpace
        H5::DataSpace dataspace = current_dataset.getSpace();
        ndims = dataspace.getSimpleExtentNdims();

        // Get the overall number of elements -- we need this
        // to allocate the temporary buffer
        tensor_flattened_size = dataspace.getSimpleExtentNpoints();

        // Get the shame of the tensor
        dims_hsize_t = new hsize_t[ndims];
        dataspace.getSimpleExtentDims(dims_hsize_t, NULL);

        for (int i = 0; i < ndims; ++i)
        {

          // Converting hsize_t to int
          dims_int.push_back(long(dims_hsize_t[i]));
        }

        // Allocate temporary float buffer
        // TODO: add support for other types like int
        // and make automatic type inference
        float_buffer = new float[tensor_flattened_size];

        current_dataset.read(float_buffer, H5::PredType::NATIVE_FLOAT,
                             dataspace, dataspace);


        buffer_tensor = CPU(kFloat).tensorFromBlob(float_buffer, dims_int);

        tensor_dict[tensor_name] = buffer_tensor.type().copy(buffer_tensor);

        delete[] float_buffer;
        delete[] dims_hsize_t;

      }

      file.close();

      return tensor_dict;
    }

    void save( string hdf5_filename, map<string, Tensor> dict_to_write)
    {

      H5::H5File file = H5::H5File(hdf5_filename, H5F_ACC_TRUNC);

      for(auto name_tensor_pair : dict_to_write)
      {

        auto tensor_to_write = name_tensor_pair.second.contiguous();
        auto tensor_name = name_tensor_pair.first;

        auto dims = tensor_to_write.sizes();

        // The dimensionality of the tensor
        auto ndims = tensor_to_write.ndimension();
        auto tensor_flattened_size = tensor_to_write.numel();
        auto tensor_to_write_flatten = tensor_to_write.view({-1});
        auto tensor_to_write_flatten_accessor = tensor_to_write_flatten.accessor<float,1>();

        float * float_buffer = new float[tensor_flattened_size];

        // Convert an array of ints into an array of hsize_t
        auto dims_hsize_t = new hsize_t[ndims];

        for (int i = 0; i < ndims; ++i)
        {
          dims_hsize_t[i] = dims[i];
        }

        for (int i = 0; i < tensor_flattened_size; ++i)
        {

           float_buffer[i] = tensor_to_write_flatten_accessor[i];
        }

        H5::DataSpace space(ndims, dims_hsize_t);

        H5::DataSet dataset = H5::DataSet(file.createDataSet(tensor_name,
                                                             H5::PredType::NATIVE_FLOAT,
                                                             space));


        dataset.write(float_buffer, H5::PredType::NATIVE_FLOAT);

        delete[] float_buffer;
        
      }

      file.close();

    }

    void inspect_checkpoint(string hdf5_filename)
    {

      auto dict = load(hdf5_filename);

      for (auto name_tensor_pair : dict)
      {
        cout << name_tensor_pair.first << ": " << name_tensor_pair.second.sizes() <<endl;
      }
    }

    Tensor upsample_bilinear(Tensor input_tensor, int output_height, int output_width)
    {

      Tensor output = input_tensor.type().tensor();

      SpatialUpSamplingBilinear_updateOutput(input_tensor, output, output_height, output_width);

      return output;
    }

    Tensor softmax(Tensor input_tensor)
    {

      Tensor output = input_tensor.type().tensor();

      SoftMax_updateOutput(input_tensor, output);

      return output;

    }

    Tensor convert_opencv_mat_image_to_tensor(Mat input_mat)
    {

      // Returns Byte Tensor with 0-255 values and (height x width x 3) shape
      // TODO: 
      // (1) double-check if this kind of conversion will always work
      //     http://docs.opencv.org/3.1.0/d3/d63/classcv_1_1Mat.html in 'Detailed Description'
      // (2) so far only works with byte representation of Mat

      unsigned char *data = (unsigned char*)(input_mat.data);

      int output_height = input_mat.rows;
      int output_width = input_mat.cols;

      auto output_tensor = CPU(kByte).tensorFromBlob(data, {output_height, output_width, 3});

      return output_tensor;

    }

    Tensor convert_image_to_batch(Tensor input_img)
    {

      // Converts height x width x depth Tensor to
      // 1 x depth x height x width Float Tensor

      // It's necessary because network accepts only batches

      auto output_tensor =  input_img.transpose(0, 2)
                                     .transpose(1, 2)
                                     .unsqueeze(0);

      return output_tensor;
    }

    Module::Ptr resnet18(int num_classes, bool fully_conv, int output_stride, bool remove_avg_pool);
    Module::Ptr resnet34(int num_classes, bool fully_conv, int output_stride, bool remove_avg_pool);


    // This one is just to build architecture, we can create functions to actually load
    // pretrained models like in pytorch

    class Resnet18_8s : public Module
    {

      public:

        int num_classes;
        Module::Ptr resnet18_8s;
        
        Resnet18_8s(int num_classes=21):
                    num_classes(num_classes)

        {

          resnet18_8s = torch::resnet18(num_classes,    
                                        true,           /* fully convolutional model */
                                        8,              /* we want subsampled by 8 prediction*/
                                        true);          /* remove average pooling layer */

          // Adding a module with this name to be able to easily load
          // weights from pytorch models
          add_module("resnet18_8s", resnet18_8s);

        }

        Tensor forward(Tensor input)
        {

          // probably we can add some utility functions to add softmax on top 
          // resize the ouput in a proper way

          // input is a tensor of shape batch_size x #channels x height x width
          int output_height = input.size(2);
          int output_width = input.size(3);

          auto subsampled_prediction = resnet18_8s->forward(input);

          auto full_prediction = upsample_bilinear(subsampled_prediction, output_height, output_width);

          return full_prediction;
        }
    };


    class Resnet34_8s : public Module
    {

      public:

        int num_classes;
        Module::Ptr resnet34_8s;
        
        Resnet34_8s(int num_classes=21):
                    num_classes(num_classes)

        {

          resnet34_8s = torch::resnet34(num_classes,    
                                        true,           /* fully convolutional model */
                                        8,              /* we want subsampled by 8 prediction*/
                                        true);          /* remove average pooling layer */

          // Adding a module with this name to be able to easily load
          // weights from pytorch models
          add_module("resnet34_8s", resnet34_8s);

        }

        Tensor forward(Tensor input)
        {

          // TODO:

          // (1) This part with upsampling is the same for all fully conv models
          //     Might make sense to write an abstract class to avoid duplication
          // (2) Probably we can add some utility functions to add softmax on top 
          //      resize the ouput in a proper way

          // input is a tensor of shape batch_size x #channels x height x width
          int output_height = input.size(2);
          int output_width = input.size(3);

          auto subsampled_prediction = resnet34_8s->forward(input);

          auto full_prediction = upsample_bilinear(subsampled_prediction, output_height, output_width);

          return full_prediction;
        }
    };


    Module::Ptr resnet18(int num_classes=1000, bool fully_conv=false, int output_stride=32, bool remove_avg_pool=false)
    {

      return std::shared_ptr<torch::ResNet<torch::BasicBlock>>(
             new torch::ResNet<torch::BasicBlock>({2, 2, 2, 2},
                                                  num_classes,
                                                  fully_conv,
                                                  remove_avg_pool,
                                                  output_stride ));
    }


    Module::Ptr resnet34(int num_classes=1000, bool fully_conv=false, int output_stride=32, bool remove_avg_pool=false)
    {

      return std::shared_ptr<torch::ResNet<torch::BasicBlock>>(
             new torch::ResNet<torch::BasicBlock>({3, 4, 6, 3},
                                                  num_classes,
                                                  fully_conv,
                                                  remove_avg_pool,
                                                  output_stride ));
    }




}


int main()
{

 
  // Structure the project in a better way

  // Add a correct linking to Opencv on the local machine

  // Get the build running on laptop for demo

  // upload all the transferred models

  // write the fcn wrapper

  // test the resnet-34


  // -----

  // first convert 34 also -- convert weights, run a test (check)

  // write wrappers for our fcns so that it would be easier to use
  // and there will be no need to  (check)
  // write separate function like resnet_34_imagenet resnet_34_8s_pascal_voc and so on ()


  // auto net = torch::resnet34(21,    /* pascal # of classes */
  //                            true,  /* fully convolutional model */
  //                            8,     /* we want subsampled by 8 prediction*/
  //                            true); /* remove avg pool layer */   


  auto net = make_shared<torch::Resnet34_8s>(21);

  net->load_weights("../resnet34_fcn_new.h5");
  net->cuda();

  VideoCapture cap(0); // open the default camera

  if(!cap.isOpened())  // check if we succeeded
      return -1;

  Mat frame;
  
  for(;;)
  { 

    cap >> frame;
        
    // BGR to RGB which is what our networks was trained on
    cvtColor(frame, frame, COLOR_BGR2RGB);
      
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

    // auto subsampled_prediction = net->forward(input_tensor_gpu);
    // auto full_prediction = torch::upsample_bilinear(subsampled_prediction, output_height, output_width);

     auto full_prediction = net->forward(input_tensor_gpu);

    // This is necessary to correctly apply softmax,
    // last dimension should represent logits
    auto full_prediction_flattned = full_prediction.squeeze(0)
                                                   .view({21, -1})
                                                   .transpose(0, 1);

    auto softmaxed = torch::softmax(full_prediction_flattned).transpose(0, 1);


    // 15 is a class for a person
    auto layer = softmaxed[15].contiguous().view({output_height, output_width, 1}).toBackend(Backend::CPU);


    // Fuse the prediction probabilities and the actual image to form a masked image.
    auto masked_image = ( image_tensor  * layer.expand({output_height, output_width, 3}) ) * 255 ;


    // A function to convert Tensor to a Mat
    auto layer_cpu = masked_image.toType(CPU(kByte));

    auto converted = Mat(output_height, output_width, CV_8UC3, layer_cpu.data_ptr());

    // OpenCV want BGR not RGB
    cvtColor(converted, converted, COLOR_RGB2BGR);

    imshow("Masked image", converted);

    if(waitKey(30) >= 0 ) break;
  }

  
  return 0;
}