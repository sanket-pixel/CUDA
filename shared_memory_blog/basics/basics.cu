#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void add_one(int *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 100000) {
        array[idx] += 1;
    }
}


#define ARRAY_SIZE 1000000

int main() {
    // Allocate memory for the array on the host (CPU)
    int h_array[ARRAY_SIZE];
    // Initialize the array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] = i;
    }
    // Measure time for CPU execution
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] += 1;
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cpu = stop_cpu - start_cpu;
    // Reset the array for the CUDA execution
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] = i;
    }
    // Allocate memory on the device (GPU)
    int* d_array;
    cudaMalloc(&d_array, ARRAY_SIZE * sizeof(int));
    // Copy data from host to device
    cudaMemcpy(d_array, h_array, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    // Warmup
    for (int i = 0; i < 1000; i++){
        add_one<<<1000, 128>>>(d_array);
    }
    // Create CUDA events for timing
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    // Record the start event
    cudaEventRecord(start_cuda);
    // Launch the kernel with 1000 blocks of 128 threads each
    for (int i = 0; i < 1000; i++){
        add_one<<<1000, 128>>>(d_array);
    }
    // Record the stop event
    cudaEventRecord(stop_cuda);
    // Wait for the stop event to complete
    cudaEventSynchronize(stop_cuda);
    // Calculate the elapsed time for CUDA
    float milliseconds_cuda = 0;
    cudaEventElapsedTime(&milliseconds_cuda, start_cuda, stop_cuda);
    // Copy the results back to the host
    cudaMemcpy(h_array, d_array, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    // Output the time taken by the CPU and CUDA
    std::cout << "Time taken by the CPU for loop: " << duration_cpu.count() << " ms" << std::endl;
    std::cout << "Time taken by the CUDA kernel: " << milliseconds_cuda/1000 << " ms" << std::endl;
    std::cout << "CUDA kernel runs " << duration_cpu.count()/(milliseconds_cuda/1000) << "x times faster" << std::endl;
    // Destroy the CUDA events
    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);
    return 0;
}