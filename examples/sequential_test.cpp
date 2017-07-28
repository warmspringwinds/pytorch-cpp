/*
Example shows how to define an architecture, visualize it later on
using std tools, and get a forward pass from that model.
*/

#include "ATen/ATen.h"
#include <sstream>

using namespace at; // assumed in the following

namespace pytorch 
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


		  Tensor output;
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

			const std::string tostring() const { return std::string("cunn.ReLU"); }
	};


}

int main()
{

	auto net = std::make_shared<pytorch::Sequential>();
	net->add(std::make_shared<pytorch::ReLU>());
	net->add(std::make_shared<pytorch::ReLU>());
	net->add(std::make_shared<pytorch::ReLU>());

	//Visualize the architecture
	std::cout << net->tostring() << std::endl;

	Tensor dummy_input = CUDA(kFloat).ones({3, 4}) * (-10);

	Tensor output = net->forward(dummy_input);

	// Print out the results -- should be zeros, because we applied RELU
	std::cout << output << std::endl;

	// Overall output:

	// nn.Sequential {
	//   (1) cunn.ReLU
	//   (2) cunn.ReLU
	//   (3) cunn.ReLU
	// }

	//  0  0  0  0
	//  0  0  0  0
	//  0  0  0  0
	// [ CUDAFloatTensor{3,4} ]


	return 0;
}