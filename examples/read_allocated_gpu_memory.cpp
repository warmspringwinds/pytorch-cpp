/*
Example shows how an already allocated memory can be reused.
It's a common case when the memory has to be used without transferring it to CPU
and back to GPU.
*/

#include "ATen/ATen.h"
#include <cuda_runtime.h>

using namespace at; // assumed in the following


int main()
{

	int width = 300;
	int height = 300;

	// Dummy CPU image -- RGBA
	std::vector<unsigned char> image(4 * width * height);
	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			size_t idx = y * width + x;
			unsigned char value = (float) (y + 1) / height * 255;
			if (x < 0.03125*width) {
				image[ idx * 4 + 0] = 255-value;
				image[ idx * 4 + 1] = 255-value;
				image[ idx * 4 + 2] = 255-value;
				image[ idx * 4 + 3] = 255;
			}
			else if (x < 0.34375*width) {
				image[ idx * 4 + 0] = value;
				image[ idx * 4 + 1] = 0;
				image[ idx * 4 + 2] = 0;
				image[ idx * 4 + 3] = 255;
			}
			else if (x < 0.65625*width) {
				image[ idx * 4 + 0] = 0;
				image[ idx * 4 + 1] = 255-value;
				image[ idx * 4 + 2] = 0;
				image[ idx * 4 + 3] = 255;
			}
			else if (x < 0.96875*width) {
				image[ idx * 4 + 0] = 0;
				image[ idx * 4 + 1] = 0;
				image[ idx * 4 + 2] = value;
				image[ idx * 4 + 3] = 255;
			}
			else {
				image[ idx * 4 + 0] = value;
				image[ idx * 4 + 1] = value;
				image[ idx * 4 + 2] = value;
				image[ idx * 4 + 3] = 255;
			}
		}
	}

	// Load the dummy image to GPU
	unsigned char * cuda_pointer;
	cudaMalloc(&cuda_pointer, 4 * width * height * sizeof(unsigned char));
	cudaMemcpy(cuda_pointer, image.data(), sizeof(unsigned char) * 4 * width * height, cudaMemcpyHostToDevice);

	// Read the dummy image from GPU and use it as a tensor later on
	auto f = CUDA(kByte).tensorFromBlob(cuda_pointer, {4 * width * height});
	auto new_one = f.toType(CPU(kByte));

	// Nicely print out the contents of the variable
	std::cout << f << std::endl;
	
	cudaFree(cuda_pointer);

}
