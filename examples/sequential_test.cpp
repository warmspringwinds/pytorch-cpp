/*
Example shows how to define an architecture, visualize it later on
using std tools, and get a forward pass from that model.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <sstream>
#include <map>
#include "H5Cpp.h"

#define TENSOR_DEFAULT_TYPE CPU(kFloat)

using namespace at;


using std::map;
using std::string;
using std::vector;
using std::pair;
using std::shared_ptr;
using std::make_shared;


namespace torch 
{   

   class Module
   {

      public:

        // Sequential module need the counter
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

        // Like in Pytorch each module stores the modules that it uses
        vector<pair<string, Ptr>> modules;

        // And parameters that are explicitly used by the current module
        vector<pair<string, Tensor>> parameters;

        // Plus buffers which are meant to store running mean and var for batchnorm layers
        vector<pair<string, Tensor>> buffers;

        
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

        // A function to add another modules inside current module
        // Acts as Pytorch's Module.register_parameter() function
        void register_parameter(string parameter_name, Tensor parameter)
        {


          parameters.push_back(pair<string, Tensor>(parameter_name, parameter));
        }

        // A function to add another modules inside current module
        // Acts as Pytorch's Module.register_buffer() function
        void register_buffer(string buffer_name, Tensor buffer)
        {


          buffers.push_back(pair<string, Tensor>(buffer_name, buffer));
        }

        map<string, Tensor> state_dict( map<string, Tensor> & destination,
                                        string prefix="")
        {


          for(auto name_parameter_pair: parameters)
          {

            destination[prefix + name_parameter_pair.first] = name_parameter_pair.second;
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

            // Register "wight" as a parameter in order to be able to
            // restore it from a file later on
            register_parameter("weight", convolution_weight);

            // don't know why this works yet, doesn't work with TENSOR_DEFAULT_TYPE.tensor();
            bias_weight = Tensor();

            // Check if we need bias for our convolution
            if(bias)
            {

              bias_weight = TENSOR_DEFAULT_TYPE.zeros({out_channels});

              // Register "bias" as a parameter in order to be able to
              // restore it from a file later on
              register_parameter("bias", bias_weight);
            }

            // These variables are not needed for forward inferece,
            // but we need them in order to call an underlying C
            // function. Later they will be used for backward pass
            finput = TENSOR_DEFAULT_TYPE.tensor();
            fgradInput = TENSOR_DEFAULT_TYPE.tensor();

            // These variables depend on # of groups, so far only
            // one group is supported. Needs to be changed to tensor_list
            // in order to support multiple groups.
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

          register_parameter("weight", gamma_weight);
          register_parameter("bias", beta_bias_weight);
          register_buffer("running_mean", running_mean);
          register_buffer("running_var", running_var);

          // We don't recompute the mean and var during inference
          // So, some variables are initialized for possible future use case.
          save_mean = TENSOR_DEFAULT_TYPE.ones(num_features);
          save_std = TENSOR_DEFAULT_TYPE.ones(num_features);

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


    // TODO: move this thing out in a separate logical unit: models/resnet

    // A helper function for a 3 by 3 convolution without bias
    // Which is used in every resnet architecture.
    Module::Ptr conv3x3(int in_planes, int out_planes, int stride=1)
    {


      return std::make_shared<Conv2d>(in_planes, out_planes, 3, 3, stride, stride, 1, 1, 1, 1, 1, false);
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
          indices =  CPU(kLong).tensor();
        };


        ~MaxPool2d() {};

        Tensor forward(Tensor input)
        {

          Tensor output = TENSOR_DEFAULT_TYPE.tensor();

          SpatialMaxPooling_updateOutput(input,
                                         output,
                                         indices,
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

        Tensor indices;
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

          Tensor output = TENSOR_DEFAULT_TYPE.tensor();

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

          Tensor weight;
          Tensor bias_weight;

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

            weight = TENSOR_DEFAULT_TYPE.zeros({out_features, in_features});
            register_parameter("weight", weight);

            // don't know why this works yet, doesn't work with TENSOR_DEFAULT_TYPE.tensor();
            bias_weight = Tensor();

            // Check if we need bias for our convolution
            if(bias)
            {

              bias_weight = TENSOR_DEFAULT_TYPE.ones({out_features});

              // Register "bias" as a parameter in order to be able to
              // restore it from a file later on
              register_parameter("bias", bias_weight);
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

            Tensor output = input.type().zeros({input.size(0), weight.size(0)});

            output.addmm_(0, 1, input, weight.t());
            
            if(bias)
            {
              // TODO: check if in-place resize affects the result
              output.add_(bias_weight.expand({output.size(0), output.size(1)}));  
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
        BasicBlock(int inplanes, int planes, int stride=1, Module::Ptr downsample=nullptr)
        {

          conv1 = conv3x3(inplanes, planes, stride);
          bn1 = std::make_shared<BatchNorm2d>(planes);
          relu = std::make_shared<ReLU>();
          conv2 = conv3x3(planes, planes);
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

        int stride;
        int in_planes;
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
        ResNet(IntList layers, int num_classes=1000) :

        // First depth input is the same for all resnet models
        in_planes(64)

        {

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

          Tensor output = TENSOR_DEFAULT_TYPE.tensor();

          output = conv1->forward(input);
          output = bn1->forward(output);
          output = relu->forward(output);
          output = maxpool->forward(output);

          output = layer1->forward(output);
          output = layer2->forward(output);
          output = layer3->forward(output);
          output = layer4->forward(output);

          output = avgpool->forward(output);

          // Flatten the output in order to apply linear layer
          output = output.view({output.size(0), -1});
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

          auto first_block = std::make_shared<BlockType>(in_planes, planes, stride, downsample);
          new_layer->add(first_block);

          in_planes = planes * BlockType::expansion;

          for (int i = 0; i < blocks - 1; ++i)
          {
            
            new_layer->add(std::make_shared<BlockType>(in_planes, planes, 1));
          }

          return new_layer;

        }

    };


    Module::Ptr resnet18(int num_classes=1000)
    {

      return std::shared_ptr<torch::ResNet<torch::BasicBlock>>(
          new torch::ResNet<torch::BasicBlock>({2, 2, 2, 2}, num_classes) );
    }


    Module::Ptr resnet34(int num_classes=1000)
    {

      return std::shared_ptr<torch::ResNet<torch::BasicBlock>>(
          new torch::ResNet<torch::BasicBlock>({3, 4, 6, 3}, num_classes) );
    }


}



int main()
{

  auto net = torch::resnet18();
  
  // Dict provides us with mapping from weights' names
  // to the pointers to Tensors  
  map<string, Tensor> dict;
  net->state_dict(dict);

  // Open the hdf-5 file with weights
  H5::H5File file = H5::H5File("resnet18.h5", H5F_ACC_RDONLY);
  // Array to store the shape of the current tensor
  hsize_t * dims;
  // Float buffer to intermediately store weights
  float * float_buffer;
  // 'Rank' of the tensor
  int ndims;
  // Number of elements in the current tensor
  hsize_t tensor_flattened_size;

  for (auto x : dict)
  {


    H5::DataSet dset = file.openDataSet(x.first);

    // Get the number of dimensions for the current tensor
    H5::DataSpace dspace = dset.getSpace();
    ndims = dspace.getSimpleExtentNdims();
    tensor_flattened_size = dspace.getSimpleExtentNpoints();
    dims = new hsize_t[ndims];
    dspace.getSimpleExtentDims(dims, NULL);
    float_buffer = new float[tensor_flattened_size];
    H5::DataSpace mspace(ndims, dims);
    dset.read(float_buffer, H5::PredType::NATIVE_FLOAT, mspace, 
    dspace);

    // Reading the raw floats into the Tensor
    // But the buffer is not owned by Tensor -- we still need to free it up
    // afterwards
    auto buffer_tensor = CPU(kFloat).tensorFromBlob(float_buffer, x.second.sizes());

    x.second.copy_(buffer_tensor);

    delete[] dims;
    delete[] float_buffer;
  }

  file.close();

  // Save the actual weights and read them (check) -- weights are different from 0 and 1 s, but still not a proof
  // thest the outcome by comparing the pytorch and the cpp
  // -- first test on a dummy image
  // Wrap up the upper part into separate function
  // write a function to convert the weights to gpu
  // write a dataloader for the surgical tools 
  // train

  // Create a dummy and run it

  // How the input should be formed in terms of size
  
  // First one is always for batch size, for sure
  // Second one is most probably depth
  // other dims represent the spatial 

  auto dummy_input = CPU(kFloat).ones({1, 3, 224, 224});

  auto result_tensor = net->forward(dummy_input);

  std::cout << result_tensor;

  return 0;
}