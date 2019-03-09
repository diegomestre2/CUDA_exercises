#define WARP_SZ 32
#include <memory>
#include <algorithm>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
 __device__ inline int lane_id(void) { return threadIdx.x % WARP_SZ; }

 __global__ 
 void parsel_kernel_phase1(uint32_t *input, uint32_t *counter, uint32_t *pred, const size_t num_items) {
 	int tid = blockIdx.x * blockDim.x + threadIdx.x;

 	if (tid >= (num_items >> 5) ) // divide by 32
 		return;

 	int lnid = lane_id();
 	int warp_id = tid >> 5; // global warp number

 	unsigned int mask;
 	int cnt;

 	for(int i = 0; i < 32 ; i++) {
 		mask = __ballot(input[(warp_id<<10)+(i<<5)+lnid]);

 		if (lnid == 0)
 			pred[(warp_id<<5)+i] = mask;

 		if (lnid == i)
 			cnt = __popc(mask);
 	}
 	// para reduction to a sum of 1024 elements
 	#pragma unroll
 	for (int offset = 16 ; offset > 0; offset >>= 1)
 		cnt += __shfl_down(cnt, offset);

 	if (lnid == 0)
 		counter[warp_id] = cnt; // store the sum of the group
 }

 /* PHASE3: produce final result array */
 __global__ 
 	void parsel_kernel_phase3(uint32_t *output, uint32_t *counter, uint32_t *pred, const size_t num_items) {
 	int tid = blockIdx.x * blockDim.x + threadIdx.x;
 	if (tid >= (num_items >> 5) ) // divide by 32
 		return;

 	int lnid = lane_id();
 	int warp_id = tid >> 5; // global warp number

	unsigned int predmask;
 	int cnt;

 	for(int i = 0; i < 32 ; i++) {
 		if (lnid == i) {
 			// each thr take turns to load its local var (i.e regs)
 			predmask = pred[(warp_id<<5)+i];
 			cnt = __popc(predmask);
 		}
 	}
 // parallel prefix sum

 	#pragma unroll
 	for (int offset=1; offset<32; offset<<=1) {
 		int n = __shfl_up(cnt, offset) ;
 		printf("\n n %d\n", n);
 		if (lnid >= offset) cnt += n;
 		printf("\n cnt %d\n", cnt);
 	}

 	int global_index =0 ;
 	if (warp_id > 0)
 		global_index = counter[warp_id -1];
 		printf("\n global_index %d\n", global_index);

 	for(int i = 0; i < 32 ; i++) {
 		int mask = __shfl(predmask, i); // broadcast from thr i
 		int subgroup_index = 0;
 		if (i > 0)
 			subgroup_index = __shfl(cnt, i-1); // broadcast from thr i-1 if i>0

 		if (mask & (1 << lnid ) ) // each thr extracts its pred bit
 			output[global_index + subgroup_index + __popc(mask & ((1 << lnid) - 1))] = (warp_id<<10)+ (i<<5) + lnid ;
 	}
 }
 
 int main() {
 	size_t vector_length = 32;
 	auto h_input =   std::make_unique<uint32_t[]>(vector_length);
 	auto h_counter = std::make_unique<uint32_t[]>(vector_length);
 	auto h_output =  std::make_unique<uint32_t[]>(vector_length);
 	auto h_pred =    std::make_unique<uint32_t[]>(vector_length);
 
 	auto generate_number_inc  = [n = 3]() mutable { return ++n; };
 	auto generate_number_pred = [n = 0]() mutable { return 0;   };

 	std::generate(h_input.get(), h_input.get() + vector_length, generate_number_inc);
 	std::generate(h_pred.get(), h_pred.get() + vector_length, generate_number_pred);

 	h_pred[5] = 1;
 	auto printer = [&](auto& value) { std::cout << value << ' '; };

 	std::cout << "Bitmap ";
 	std::for_each(h_pred.get(), h_pred.get() + vector_length, printer);
 	std::cout << std::endl;

 	std::cout << "Input ";
 	std::for_each(h_input.get(), h_input.get() + vector_length, printer);
	std::cout << std::endl;

 	// Device Vectors using C pointers
	uint32_t *d_input, *d_counter, *d_output, *d_pred;

	// Allocating memory on device
	cudaMalloc(&d_input,   vector_length * sizeof(uint32_t));
	cudaMalloc(&d_counter, vector_length * sizeof(uint32_t));
	cudaMalloc(&d_output,  vector_length * sizeof(uint32_t));
	cudaMalloc(&d_pred,    vector_length * sizeof(uint32_t));

	// Copy from Host to Device
	cudaMemcpy(d_input, h_input.get(), vector_length * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pred,  h_pred.get(),  vector_length * sizeof(uint32_t), cudaMemcpyHostToDevice);


 	size_t threads_per_block = 32;
	size_t blocks_per_grid = (vector_length + (threads_per_block - 1))/ threads_per_block;

 	std::cout << " Blocks " << blocks_per_grid << " threads " << threads_per_block << std::endl;
 	parsel_kernel_phase1<<<blocks_per_grid, threads_per_block>>>(d_input, d_counter, d_pred, vector_length);
 	cudaDeviceSynchronize();

 	std::cout << "Counter ";
 	cudaMemcpy(h_counter.get(), d_counter, vector_length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	std::for_each(h_counter.get(), h_counter.get() + vector_length, printer);
	std::cout << std::endl;

 	std::cout << "Predicate ";
 	cudaMemcpy(h_pred.get(), d_pred, vector_length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	std::for_each(h_pred.get(), h_pred.get() + vector_length, printer);
	std::cout << std::endl;

 	thrust::inclusive_scan(thrust::device, d_counter, d_counter + vector_length, d_counter);

 	parsel_kernel_phase3<<<blocks_per_grid, threads_per_block>>>(d_output, d_counter, d_pred, vector_length);
 	cudaDeviceSynchronize();

 	// Copy from Device to Host
	cudaMemcpy(h_output.get(), d_output, vector_length * sizeof(uint32_t), cudaMemcpyDeviceToHost);

 	std::cout << "Output ";
	std::for_each(h_output.get(), h_output.get() + vector_length, printer);
	std::cout << std::endl;

 }