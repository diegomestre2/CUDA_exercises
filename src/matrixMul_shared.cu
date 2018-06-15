/*The following code sample is an implementation of matrix multiplication that does take advantage of shared memory.
In this implementation, each thread block is responsible for computing one square sub-matrix Csub of C and each thread
within the block is responsible for computing one element of Csub. As illustrated in Figure 10, Csub is equal to the
product of two rectangular matrices: the sub-matrix of A of dimension (A.width, block_size) that has the same row
indices as Csub, and the sub-matrix of B of dimension (block_size, A.width )that has the same column indices as 
Csub. In order to fit into the device's resources, these two rectangular matrices are divided into as many square
matrices of dimension block_size as necessary and Csub is computed as the sum of the products of these square 
matrices. Each of these products is performed by first loading the two corresponding square matrices from global
memory to shared memory with one thread loading one element of each matrix, and then by having each thread compute
one element of the product. Each thread accumulates the result of each of these products into a register and once
done writes the result to global memory.*/

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)

#define BLOCK_SIZE 16

typedef struct {
	int width;
	int height;
	int stride;
	float* elements;
} Matrix;


__device__ float getElement(const Matrix A, int row, int column){

	return A.elements[row * A.stride + column];
} 

__device__ void setElement(Matrix A, int row, int column, float value){

	A.elements[row * A.stride + column] = value;
}

/*
SubMatrix located column times to the right and row times down
*/
__device__ Matrix getSubMatrix(Matrix A, int row, int column){

	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * column];

	return Asub;
}

void matrixMul(const Matrix A, const Matrix B, Matrix C){


}