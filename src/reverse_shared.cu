/*
Shared memory is allocated per thread block, so all threads in the block have access to the same
shared memory. Threads can access data in shared memory loaded from global memory by other threads
within the same thread block.

To ensure correct results when parallel threads cooperate, we must synchronize the threads. CUDA 
provides a simple barrier synchronization primitive, __syncthreads(). A thread’s execution can only
proceed past a __syncthreads() after all threads in its block have executed the __syncthreads().

This code reverses the data in a 64-element array using shared memory. The two kernels are very 
similar, differing only in how the shared memory arrays are declared and how the kernels are invoked.
*/

#include <iostream>

__global__
void staticReverse(int *device_array, size_t array_size){

	__shared__ int shared_array[64];

	int thread_id = threadIdx.x;
	int thread_position_to_swap = array_size - thread_id - 1;

	shared_array[thread_id] = device_array[thread_id];
	__syncthreads();
	device_array[thread_id] = shared_array[thread_position_to_swap];
}

__global__
void dynamicReverse(int *device_array, size_t array_size){

	extern __shared__ int shared_array[];

	int thread_id = threadIdx.x;
	int thread_position_to_swap = array_size - thread_id - 1;

	shared_array[thread_id] = device_array[thread_id];
	__syncthreads();
	device_array[thread_id] = shared_array[thread_position_to_swap];
}

int main(){

	const size_t array_size = 64;
	int host_array[array_size], host_result[array_size], host_check[array_size];

	for(size_t i = 0; i != array_size; ++i){
		host_array[i] = i;
		host_check[i] = array_size - i - 1;
		host_result[i] = 0;
	}

	int *device_array;
	cudaMalloc(&device_array, array_size * sizeof(int));

	// Run version with static shared memory
	cudaMemcpy(device_array, host_array, array_size * sizeof(int), cudaMemcpyHostToDevice);
	staticReverse<<<1, array_size>>>(device_array, array_size);
	cudaMemcpy(host_result, device_array, array_size * sizeof(int), cudaMemcpyDeviceToHost);

	// Check result
	for(size_t i = 0; i != array_size; ++i){
		
		if(host_result[i] != host_check[i]){
			std::cout << " Static - Mismatch found in position : " << i << " - " 
			<< host_result[i] << " != " << host_check[i] << '\n';
		}
	}

	// Run version with dynamic shared memory
	cudaMemcpy(device_array, host_array, array_size * sizeof(int), cudaMemcpyHostToDevice);
	dynamicReverse<<<1, array_size, array_size * sizeof(int)>>>(device_array, array_size);
	cudaMemcpy(host_result, device_array, array_size * sizeof(int), cudaMemcpyDeviceToHost);

	// Check result
	for(size_t i = 0; i != array_size; ++i){

		if(host_result[i] != host_check[i]){
			std::cout << " Dynamic - Mismatch found in position : " << i << " - " 
			<< host_result[i] << " != " << host_check[i] << '\n';
		}
	}

	std::cout << "***SUCCESSFULLY EXECUTED!***" <<std::endl;
}