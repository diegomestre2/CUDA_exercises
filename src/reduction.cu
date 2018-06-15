/*
Parallel reduction device function written using Cooperative Groups. When the threads of a group
call it, they cooperatively compute the sum of the values passed by each thread in the group 
(through the val argument).
*/

#include <iostream>/* cout */
#include <cooperative_groups.h> /* thread_groups */
#include <stdio.h>/* printf */

using namespace cooperative_groups;
using u64_t = unsigned long long int;

void get_device_properties(){
    int32_t device_cnt = 0;
    cudaGetDeviceCount(&device_cnt);
    cudaDeviceProp device_prop;

    for (int i = 0; i < device_cnt; i++) {
        cudaGetDeviceProperties(&device_prop, i);
        std::cout << "+-------------------------------------------------------------------------------+\n";
        printf("|  Device id: %d\t", i);
        printf("  Device name: %s\t", device_prop.name);
        printf("  Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        std::cout << std::endl;
        printf("|  Memory Clock Rate [KHz]: %d\n",
               device_prop.memoryClockRate);
        printf("|  Memory Bus Width [bits]: %d\n",
               device_prop.memoryBusWidth);
        printf("|  Peak Memory Bandwidth [GB/s]: %f\n",
               2.0*device_prop.memoryClockRate*(device_prop.memoryBusWidth/8)/1.0e6);
        printf("|  L2 size [KB]: %d\n",
               device_prop.l2CacheSize/1024);
        std::cout << std::endl;
        printf("|  Number of SMs: %d\n",
               device_prop.multiProcessorCount);
        printf("|  Max. number of threads per SM: %d\n",
               device_prop.maxThreadsPerMultiProcessor);
        printf("|  Concurrent kernels: %d\n",
               device_prop.concurrentKernels);
        printf("|  warpSize: %d\n",
               device_prop.warpSize);
        printf("|  maxThreadsPerBlock: %d\n",
               device_prop.maxThreadsPerBlock);
        printf("|  maxThreadsDim[0]: %d\n",
               device_prop.maxThreadsDim[0]);
        printf("|  maxGridSize[0]: %d\n",
               device_prop.maxGridSize[0]);
        printf("|  pageableMemoryAccess: %d\n",
               device_prop.pageableMemoryAccess);
        printf("|  concurrentManagedAccess: %d\n",
               device_prop.concurrentManagedAccess);
        printf("|  Number of async. engines: %d\n",
               device_prop.asyncEngineCount);
        std::cout << "+-------------------------------------------------------------------------------+\n";
    }
}

template <typename group_t>
__device__ 
int reduce_sum_warp(group_t g, int *temp, int val)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    #pragma unroll
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if (lane < i) val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }

    return val; // note: only thread 0 will return full sum
}

__device__
int reduce_sum(thread_group t_group, int *temp, int value){

  	int lane = t_group.thread_rank();

  	// Each iteration halves the number of active threads
  	// Each thread adds its partial sum[i] to sum[lane + i]
  	for(size_t i = t_group.size() / 2; i > 0; i /= 2){

  		temp[lane] = value;
  		t_group.sync(); // wait for all threads to store
  		if(lane < i )
  			value += temp[lane + i];
  		t_group.sync(); // wait for all threads to load
  	}

  	return value;
}

__device__
int thread_sum(int *input, int n){

  	int sum = 0;
  	int index = blockIdx.x * blockDim.x + threadIdx.x;
  	int stride = blockDim.x * gridDim.x;

  	for(; index < n / 4; index += stride){

    		int4 in = ((int4*)input)[index];
    		sum += in.x + in.y + in.z + in.w;
  	}

  	return sum;
}

__global__
void sum_kernel_block(int *sum, int *input, int n){

  	u64_t my_sum = thread_sum(input, n);

    extern __shared__ int temp[];
    auto group = this_thread_block();
    //auto group = tiled_partition<32>(this_thread_block());
    u64_t block_sum = reduce_sum(group, temp, my_sum);

  	if(group.thread_rank() == 0)
  		atomicAdd(sum, block_sum);
}

int main(){

  	int n = 1 << 23;
  	std::cout << n << std::endl;
  	int blockSize = 256;
  	int numBlocks = (n + blockSize - 1) / blockSize;
  	int sharedBytes = blockSize * sizeof(int);

  	int *sum; int *data;
  	cudaMallocManaged(&sum, sizeof(int));
  	cudaMallocManaged(&data, n * sizeof(int));
  	std::fill_n(data, n, 1);
  	cudaMemset(sum, 0, sizeof(int));
  	// Prefetch the data to the GPU
   	int device = -1;
   	cudaGetDevice(&device);
   	cudaMemPrefetchAsync(data, n * sizeof(int), device, NULL);
   	std::cout << "Launching kernel <<<" << numBlocks << ", " << blockSize << ">>>" << std::endl;
  	sum_kernel_block<<<numBlocks, blockSize, sharedBytes>>>(sum, data, n);
    cudaDeviceSynchronize();
  	std::cout << sum << " - " << *sum << " - " << &sum << std::endl;

}