/*
Example shows how an already allocated memory can be reused.
It's a common case when the memory has to be used without transferring it to CPU
and back to GPU.
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

		  Module();

		  ~Module();

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

			Sequential() : Module() {}

			~Sequential() {}

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


}

int main()
{




	return 0;
}