/*
The CUDA programming model assumes a system composed of a host and a device, each with their own separate
memory. Kernels operate out of device memory, so the runtime provides functions to allocate, deallocate,
and copy device memory, as well as transfer data between host memory and device memory.

Example adapted from the nVIDIA CUDA 9.1 samples
*/
#include <iostream>
#include <algorithm>
#include <memory>

// Device Code
__global__ 
void vectorAdd(const float *A, const float *B, float *C, int vector_length){

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < vector_length){
		C[index] = A[index] + B[index];
	}
}

// Host Code
int main(){
	
	size_t vector_length;
	std::cout << "Enter with the size of the vector" << '\n';
	std::cin >> vector_length;
	if (vector_length > std::numeric_limits<int>::max()) {
		throw std::logic_error("This program only accepts lengths which fit in an int-type variable");
	}

	// Host Vector using C++14 smart pointers
	auto h_a = std::make_unique<float[]>(vector_length);
	auto h_b = std::make_unique<float[]>(vector_length);
	auto h_c = std::make_unique<float[]>(vector_length);

	auto generate_number = [n = 0]() mutable {return ++n;};
	std::generate(h_a.get(), h_a.get() + vector_length, generate_number);
	std::generate(h_b.get(), h_b.get() + vector_length, generate_number);

	// Device Vectors using C pointers
	float *d_a, *d_b, *d_c;

	// Allocating memory on device
	cudaMalloc(&d_a, vector_length * sizeof(float));
	cudaMalloc(&d_b, vector_length * sizeof(float));
	cudaMalloc(&d_c, vector_length * sizeof(float));

	// Copy from Host to Device
	cudaMemcpy(d_a, h_a.get(), vector_length * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.get(), vector_length * sizeof(float), cudaMemcpyHostToDevice);

	// Launching kernel 
	size_t threads_per_block = 32;
	size_t blocks_per_grid = (vector_length + (threads_per_block - 1))/ threads_per_block;
	std::cout << "\nLaunching CUDA kernel vectorAdd<<<" << blocks_per_grid 
		<< ", " << threads_per_block << ">>>" << '\n'; 
	vectorAdd<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, vector_length);

  	// Copy from Device to Host
	cudaMemcpy(h_c.get(), d_c, vector_length * sizeof(float), cudaMemcpyDeviceToHost);

	// Deallocating memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Check Results
	for(size_t i = 0; i != vector_length; ++i){
		if(h_a[i] + h_b[i] != h_c[i]){
			std::cerr << "Mismatch found in position " << i << ": Expected = "<< h_a[i] + h_b[i] 
				<< " Obtained = " << h_c[i] << '\n';
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "\nSUCCESSFULLY EXECUTED!\n" << std::endl;
	return 0;
	
}
