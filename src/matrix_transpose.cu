/*
All kernels in this study launch blocks of 32×8 threads (TILE_DIM=32, BLOCK_ROWS=8 in the code), and
each thread block transposes (or copies) a tile of size 32×32. Using a thread block with fewer threads
than elements in a tile is advantageous for the matrix transpose because each thread transposes four 
matrix elements, so much of the index calculation cost is amortized over these elements.

The kernels in this example map threads to matrix elements using a Cartesian (x,y) mapping rather than
a row/column mapping to simplify the meaning of the components of the automatic variables in CUDA C: 
threadIdx.x is horizontal and threadIdx.y is vertical.
*/

#include <iostream>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__
void matrix_copy(float *input_matrix, float *output_matrix){

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y
	size_t width = gridDim.x * TILE_DIM;

	for(size_t i = 0; i != TILE_DIM; i+= BLOCK_ROWS){
		output_matrix[(y + i) * width + x] = input_matrix[(y + i) * width + x];
	}
}

__global__
void matrix_transpose_naive(float *input_matrix, float *output_matrix){

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for(size_t i = 0; i != TILE_DIM; i+= BLOCK_ROWS){
		output_matrix[x * width + (y + i)] = input_matrix[(y + i) * width + x];
	}
}

int main(){

	const size_t x_dimension = 1024;
	const size_t y_dimension = 1024;
	const size_t matrix_size = x_dimension * y_dimension * sizeof(float);

	dim3 dimGrid(x_dimension / TILE_DIM, y_dimension / TILE_DIM);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

}