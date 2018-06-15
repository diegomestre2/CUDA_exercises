/*
Pinned memory is used as a staging area for transfers from the device to the host. We can avoid 
the cost of the transfer between pageable and pinned host arrays by directly allocating our host
arrays in pinned memory. Allocate pinned host memory in CUDA C/C++ using cudaMallocHost() or 
cudaHostAlloc(), and deallocate it with cudaFreeHost(). It is possible for pinned memory allocation
to fail, so you should always check for errors. The following code excerpt demonstrates allocation 
of pinned memory with error checking.
*/

#include <iostream>
#include <assert.h>
#include <string>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result){
#if defined(DEBUG) || defined(_DEBUG)
	if(result != cudaSuccess){
		std::cerr << "Cuda Runtime Error: " << cudaGetErrorString(result) << std::endl;
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

void profileCopies(float *h_a, float *h_b, float *d, unsigned int n, std::string desc){

	std::cout << '\n' << desc <<" transfers\n";
	unsigned int bytes = n * sizeof(float);

	// Events timing
	cudaEvent_t startEvent, stopEvent;

	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	// Measuring transfer Host to Device
	checkCuda(cudaEventRecord(startEvent, 0));
	checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));

	float time;
	checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
	std::cout << " Host to Device Bandwidth  (GB/s): " << (bytes * 1e-6) / time << std::endl;

	// Measuring transfer Device to Host
	checkCuda(cudaEventRecord(startEvent, 0));
	checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));

	checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
	std::cout << " Device to Host Bandwidth  (GB/s): " << (bytes * 1e-6) / time << std::endl;

	// Check result
	for(size_t i = 0; i != n; ++i){
		if(h_a[i] != h_b[i]){
			std::cout << " Transfers failed " << desc << std::endl;
			break;
		}
	}

	// Cleaning up events
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
} 

int main(int argc, char* argv[]){

	uint32_t n = 4 * 1024 * 1024;
	const uint32_t bytes = n * sizeof(float);

	// Host arrays
	float *h_aPageable, *h_bPageable;
	float *h_aPinned, *h_bPinned;

	// Device array
	float *d_a;

	// Allocate and initialize
	h_aPageable = (float *)malloc(n * sizeof(float));
	h_bPageable = (float *)malloc(n * sizeof(float));

	checkCuda(cudaMallocHost((void**)&h_aPinned, bytes));
	checkCuda(cudaMallocHost((void**)&h_bPinned, bytes));
	
	checkCuda(cudaMalloc((void**)&d_a, bytes));

	// Out device info and transfer size
	cudaDeviceProp device;
	checkCuda(cudaGetDeviceProperties(&device, 0));

	std::cout << "\n Device            : " << device.name << std::endl;
	std::cout << " Transfer size (MB): " << bytes / (1024 * 1024) << std::endl;

	// Perform copies and report results
	profileCopies(h_aPageable, h_bPageable, d_a, n, "Pageable");
	profileCopies(h_aPinned, h_bPinned, d_a, n, "Pinned");

	// Cleanup
	cudaFree(d_a);
	cudaFreeHost(h_aPinned);
	cudaFreeHost(h_bPinned);
	free(h_aPageable);
	free(h_bPageable);

	return 0;
}