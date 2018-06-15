/*
The shared memory is allocated using the __shared__ memory space specifier. 
Shared memory is expected to be much faster than global memory.

The following code sample is a straightforward implementation of matrix multiplication
that does not take advantage of shared memory. Each thread reads one row of A and one 
column of B and computes the corresponding element of C. A is therefore read B.columns 
times from global memory and B is read A.rows times.

Example adapted from the nVIDIA CUDA 9.1 samples
*/
#include <iostream>
#include <memory>
#include <algorithm>


struct Matrix{
	int num_rows;
	int num_columns;
	float* elements;
};

__global__
void matrixMult(const Matrix a, const Matrix b, Matrix c){
	
	float accumulate = 0.f;
	int row    = blockDim.y * blockIdx.y + threadIdx.y;
	int column = blockDim.x * blockIdx.x + threadIdx.x;
	printf("\nthreadIdx(%d) threadIdy(%d)\n",threadIdx.x,threadIdx.y);

	for(int i = 0; i != a.num_rows; ++i){
		accumulate += a.elements[row * a.num_columns + i] * b.elements[i * c.num_columns + column];
	}

	c.elements[row * c.num_columns + column] = accumulate;
}

int main(){
	
	size_t dimension = 3;
	size_t dimension_matrix = dimension * dimension; 
	Matrix h_A, h_B, h_C;
	
	h_A.num_rows 	= h_B.num_rows 	  = h_C.num_rows 	= dimension;
	h_A.num_columns = h_B.num_columns = h_C.num_columns = dimension;

	h_A.elements = (float*)malloc(dimension_matrix * sizeof(float));
	h_B.elements = (float*)malloc(dimension_matrix * sizeof(float));
	h_C.elements = (float*)malloc(dimension_matrix * sizeof(float));
	
	auto generate_element = [n = 1.f]() mutable {return (float)++n;};

	std::generate(h_A.elements, h_A.elements + (dimension_matrix), generate_element);
	std::generate(h_B.elements, h_B.elements + (dimension_matrix), generate_element);

	Matrix d_A, d_B, d_C;
	size_t size_bytes = dimension_matrix * sizeof(float);

	d_A.num_rows 	= d_B.num_rows 	  = d_C.num_rows 	= dimension;
	d_A.num_columns = d_B.num_columns = d_C.num_columns = dimension;

	cudaMalloc(&d_A.elements, size_bytes);
	cudaMalloc(&d_B.elements, size_bytes);
	cudaMalloc(&d_C.elements, size_bytes);

	cudaMemcpy(d_A.elements, h_A.elements, size_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.elements, h_B.elements, size_bytes, cudaMemcpyHostToDevice);

	// Launching kernel 
	size_t threads_per_block = dimension;
	dim3 dimBlock(threads_per_block,threads_per_block);
	dim3 dimGrid(h_B.num_columns / dimBlock.x, h_A.num_rows / dimBlock.y);
	std::cout << "\nLaunching CUDA kernel matrixMult<<<" << dimGrid.x 
		<< ", " << dimBlock.x << ">>>" << '\n'; 
		
	matrixMult<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	cudaMemcpy(h_C.elements, d_C.elements, size_bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	
	// Check Results
	for(size_t i = 0; i != dimension; ++i){
		for(size_t j = 0; j  != dimension; ++j){
			int accumulator = 0;
			for(size_t k = 0; k  != dimension; ++k)
				accumulator += h_A.elements[i * h_A.num_columns + k] * h_B.elements[k * h_B.num_columns + j];
			if(accumulator != h_C.elements[i * h_A.num_columns + j]){
					std::cerr << "Mismatch found in position " << i <<", " << j
						<< ": Expected = " << accumulator
						<< "  Obtained = " << h_C.elements[i * h_A.num_columns + j] << '\n';
					free(h_A.elements);
					free(h_B.elements);
					free(h_C.elements);
					exit(EXIT_FAILURE);
			}
		}
	}

	free(h_A.elements);
	free(h_B.elements);
	free(h_C.elements);

	std::cout << "\nSUCCESSFULLY EXECUTED!\n" << std::endl;
	return 0;

}
