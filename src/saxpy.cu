/*
SAXPY stands for “Single-Precision A·X Plus Y”.  It is a function in the standard Basic Linear Algebra
Subroutines (BLAS)library. SAXPY is a combination of scalar multiplication and vector addition, and it’s
very simple: it takes as input two vectors of 32-bit floats X and Y with N elements each, and a scalar
value A. It multiplies each element X[i] by A and adds the result to Y[i]. 
*/
#include <iostream>
#include <math.h>

__global__
void saxpy(size_t n, float a, float *x, float *y){
	
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < n)
		y[index] = a * x[index] + y[index];
}

int main(int argc, char* argv[]){

	size_t N = 20 * 1 << 20;
	// Host Vectors
	float *x, *y;
	x = (float *)malloc( N * sizeof(float));
	y = (float *)malloc( N * sizeof(float));
	
	// Initialize on Host
	for(size_t i = 0; i != N; ++i){
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Device Vectors
	float *d_x, *d_y;
	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	// Record time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Copy data from host to device
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements
	int threads_per_block = 1024;
	int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
	std::cout << " Launching saxpy<<<" << blocks_per_grid << ", " << threads_per_block << ">>>\n";
	cudaEventRecord(start);
	saxpy<<<blocks_per_grid, threads_per_block>>>(N, 2.0f, d_x, d_y);
	cudaEventRecord(stop);

	// Copy result back to Host
	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Check results
	float max_error = 0.0f;
	for(size_t i = 0; i != N; ++i){
		max_error = max(max_error, abs(y[i] - 4.0f));
	}

	std::cout << "\n Max error: " << max_error << std::endl;
	std::cout << " Theoretical Bandwidth (GB/s): " << (5505 * 10e06 * (352 / 8) * 2) / 10e09 << std::endl;
	std::cout << " Effective Bandwidth (GB/s)  : " << N * 4 * 3 / milliseconds / 1e6 << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

	return 0;
}