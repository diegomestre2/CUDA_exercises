/*
During execution there is a finer grouping of threads into warps. Multiprocessors on the GPU execute
instructions for each warp in SIMD (Single Instruction Multiple Data) fashion. The warp size 
(effectively the SIMD width) of all current CUDA-capable GPUs is 32 threads. Grouping of threads into
warps is not only relevant to computation, but also to global memory accesses. The device coalesces 
lobal memory loads and stores issued by threads of a warp into as few transactions as possible to 
minimize DRAM bandwidth
*/
#include <iostream>
#include <assert.h>

// Wrapper for cuda status
inline cudaError_t checkCuda(cudaError_t result){

#if defined(DEBUG) || defined(_DEBUG)
	if(result != cudaSuccess){
		std::cout << " Cuda Runtime Error: " << result << std::endl;
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

template<typename T>
__global__
void offset(T* vector, int stride){

	int index = blockDim.x * blockIdx.x + threadIdx.x + stride;
	vector[index]++;
}

template<typename T>
__global__
void stride(T* vector, int stride){
	
	int index = (blockDim.x * blockIdx.x + threadIdx.x) * stride;
	vector[index]++;
}

template<typename T>
void runTest(int device_id, size_t size_MB){

	size_t block_size = 256;
	float milliseconds;

	T *d_a;
	size_t size_bytes = (size_MB * 1024 * 1024) / sizeof(T);
	checkCuda(cudaMalloc(&d_a, size_bytes * 33 * sizeof(T)));

	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	std::cout << " Offset - Bandwidth (GB/s): " << '\n';
	offset<<<size_bytes / block_size, block_size>>>(d_a, 0); //warm up

	for(size_t i = 0; i!= 32; ++i){
		checkCuda(cudaMemset(d_a, 0.0, size_bytes * sizeof(T)));

		checkCuda(cudaEventRecord(startEvent, 0));
		offset<<<size_bytes / block_size, block_size>>>(d_a, i);
		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));

		checkCuda(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
		std::cout << " " << i << " - " << (2 * size_MB) / milliseconds << std::endl;
	}

	std::cout << " Stride - Bandwidth (GB/s): " << std::endl;
	stride<<<size_bytes / block_size, block_size>>>(d_a, 0); //warm up

	for(size_t i = 0; i != 32; ++i){
		checkCuda(cudaMemset(d_a, 0.0, size_bytes * sizeof(T)));

		checkCuda(cudaEventRecord(startEvent, 0));
		stride<<<size_bytes / block_size, block_size>>>(d_a, i);
		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));

		checkCuda(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
		std::cout << " " << i << " - " << (2 * size_MB) / milliseconds <<std::endl;
	}

	// Clean up
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	checkCuda(cudaFree(d_a));
}

int main(int argc, char* argv[]){

	size_t size_MB = 4;
	int device_id = 0;
	bool is_doublePrecision = false;

	for(size_t i = 1; i != argc; ++i){
		if(!strncmp(argv[i], "dev=", 4)){
			device_id = atoi((char*)(&argv[i][4]));
		} else if(!strcmp(argv[i], "fp64")){
			is_doublePrecision = true;
		}
	}

	cudaDeviceProp device;
	checkCuda(cudaSetDevice(device_id));
	checkCuda(cudaGetDeviceProperties(&device, device_id));
	std::cout << "\n Device: " << device.name << '\n';
	std::cout << " Transfer size (MB): " << size_MB << '\n';
	std::string precision = (is_doublePrecision) ? "Double" : "Single";
	std::cout << " Precision : " << precision << '\n';
	if(is_doublePrecision)
		runTest<double>(device_id, size_MB);
	else
		runTest<float>(device_id, size_MB);

	return 0;
}