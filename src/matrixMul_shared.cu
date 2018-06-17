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

/*
Matrix Multiplication Device kernel
*/
__global__
void matrixMul(const Matrix A, const Matrix B, Matrix C){

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	Matrix Csub = getSubMatrix(C, blockRow, blockCol);

	float Cvalue = 0;
	int row = threadIdx.y;
	int col = threadIdx.x;

	for(int m = 0; m < (A.width / BLOCK_SIZE); ++m){

		Matrix Asub = getSubMatrix(A, blockRow, m);
		Matrix Bsub = getSubMatrix(B, m, blockCol);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = getElement(Asub, row, col);
		Bs[row][col] = getElement(Bsub, row, col);

		_syncthreads();

		for(int e = 0; e < BLOCK_SIZE; ++e){
			Cvalue += As[row][e] * Bs[e][col];
		}
		_syncthreads(); 
	}
	setElement(Csub, row, col, Cvalue);
}

/*
Matrix Multiplication Host
*/
void matrixMultiplicaiton(Matrix A, Matrix B, Matrix C){

	Matrix d_A, d_B, d_C;

	d_A.width = d_A.stride = A.width;
	d_B.width = d_B.stride = B.width;
	d_C.width = d_C.stride = C.width;
	
	d_A.height = A.height;
	d_B.height = B.height;
	d_C.height = C.height;

	size_t size_A = A.width * A.height * sizeof(float);
	size_t size_B = B.width * B.height * sizeof(float);
	size_t size_C = C.width * C.height * sizeof(float);

	cudaMalloc(&d_A.elements, size_A);
	cudaMalloc(&d_B.elements, size_B);
	cudaMalloc(&d_C.elements, size_C);

	cudaMemcpy(d_A.elements, A.elements, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.elements, B.elements, size_B, cudaMemcpyHostToDevice);

	dim3 dimBLock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBLock.x, A.height / dimBLock.y);

	matrixMul<<<dimGrid, dimBLock>>>(d_A, d_B, d_C);

	cudaMemcpy(C.elements, d_C.elements, size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}
























