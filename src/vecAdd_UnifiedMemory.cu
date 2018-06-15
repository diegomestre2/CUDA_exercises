/*
Unified Memory in CUDA makes this easy by providing a single memory space accessible by all GPUs
and CPUs in your system. To allocate data in unified memory, call cudaMallocManaged(), which 
returns a pointer that you can access from host (CPU) code or device (GPU) code. To free the data,
just pass the pointer to cudaFree().
*/

#include <math.h>
#include <iostream>

// serial
__global__
void add(size_t n, float *x, float *y){

	for(size_t i = 0; i < n; ++i){
		y[i] = x[i] + y[i];
	}
}

// parallel using 1 block
__global__
void add2(int n, float *x, float *y){

	int index = threadIdx.x;
	int stride = blockDim.x;
	for(int i = index; i < n; i += stride){
		y[i] = x[i] + y[i];
	}
}

// parallel using grid stride loop
__global__
void add3(int n, float *x, float *y){

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	//printf(" N %d threadIdx.x %d blockIdx.x %d index %d stride %d \n", n, threadIdx.x, blockIdx.x, index, stride);
	for(int i = index; i < n; i += stride){
		y[i] = x[i] + y[i];
	}
}

//initialize data in kernel to avoid page fault
__global__
void init(int n, float *x, float *y){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = index; i < n; i += stride){
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
}

int main(int argc, char *argv[]){

	size_t N  = 1 << 20;
	float *x, *y;

	// Allocate Unified Memory - acessible by GPU and CPU
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// Initiallize x and y on the Host
	for(size_t i = 0; i != N; ++i){
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// kernel parameters
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	// Run kernel with 1M elements on the GPU
	std::cout << "\n Launching CUDA kernel add<<<" << numBlocks << ", " 
		<< blockSize << ">>>" << '\n'; 
	add<<<numBlocks, blockSize>>>(N, x, y);

	// Initialize data on GPU to avoid page fault
	std::cout << "\n Launching CUDA kernel init<<<" << numBlocks << ", " 
		<< blockSize << ">>>" << '\n'; 
	init<<<numBlocks, blockSize>>>(N, x, y);

	// Run kernel with 1M elements on the GPU
	std::cout << "\n Launching CUDA kernel add2<<<" << numBlocks << ", " 
		<< blockSize << ">>>" << '\n'; 
	add2<<<numBlocks, blockSize>>>(N, x, y);


  	// Prefetch the data to the GPU
 	int device = -1;
 	cudaGetDevice(&device);
 	cudaMemPrefetchAsync(x, N * sizeof(float), device, NULL);
 	cudaMemPrefetchAsync(y, N * sizeof(float), device, NULL);

	std::cout << "\n Launching CUDA kernel add3<<<" << numBlocks << ", " 
		<< blockSize << ">>>" << '\n'; 
	add3<<<numBlocks, blockSize>>>(N, x, y);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Check for errors, all values should be 3.0f
	float max_error = 0.0f;
	for(size_t i = 0; i != N; ++i){
		max_error = fmax(max_error, fabs(y[i] - 3.0f));
	}
	std::cout << "Max error: " << max_error << std::endl;

	// Free memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}