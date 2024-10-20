#include <iostream>
#include <cuda.h>
#include <chrono>
#include "opencv2/opencv.hpp"
#define ARRAY_SIZE 1000000
#define BLOCK_SIZE 1024
__global__ void add_one(int *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ARRAY_SIZE) {
        array[idx] += 1;
    }
}

int main() {
    // allocate memory for the array on the host (CPU)
    int h_array[ARRAY_SIZE];
    // initialize the array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] = i;
    }
    // measure time for CPU execution
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] += 1;
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cpu = stop_cpu - start_cpu;
    // reset the array for the CUDA execution
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] = i;
    }
    // allocate memory on the device (GPU)
    int* d_array;
    cudaMalloc(&d_array, ARRAY_SIZE * sizeof(int));
    // copy data from host to device
    cudaMemcpy(d_array, h_array, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    // compute total blocks required
    unsigned total_threads = ARRAY_SIZE;
    unsigned total_blocks = int(total_threads/BLOCK_SIZE)+1;
    std::cout << total_blocks << std::endl;
    // warmup
    for (int i = 0; i < 1000; i++){
        add_one<<<total_blocks, BLOCK_SIZE>>>(d_array);
    }
    cudaDeviceSynchronize();
    // launch the kernel with 1000 blocks of 128 threads each
    auto start_cuda = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++){
        add_one<<<total_blocks, BLOCK_SIZE>>>(d_array);
    }
    cudaDeviceSynchronize();
    auto stop_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cuda = stop_cuda - start_cuda;
    cudaMemcpy(h_array, d_array, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    std::cout << "Time taken by the CPU for loop: " << duration_cpu.count() << " ms" << std::endl;
    std::cout << "Time taken by the CUDA kernel: " << duration_cuda.count()/1000 << " ms" << std::endl;
    std::cout << "CUDA kernel runs " << duration_cpu.count()/(duration_cuda.count()/1000) << "x times faster" << std::endl;
    return 0;
}