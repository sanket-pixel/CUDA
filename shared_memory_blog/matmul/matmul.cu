#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

// Size of the square matrix (for simplicity, use multiples of block size)
#define N 1024
#define BLOCK_SIZE 32

// CPU matrix multiplication
void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// CUDA kernel for naive matrix multiplication
__global__ void matmul_cuda_naive(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CUDA kernel for shared memory matrix multiplication
__global__ void matmul_cuda_shared(float* A, float* B, float* C, int n) {
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int blockIdx = 0; blockIdx < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; blockIdx++) {
        // Load tiles into shared memory
        if (row < n && blockIdx * BLOCK_SIZE + threadIdx.x < n)
            Asub[threadIdx.y][threadIdx.x] = A[row * n + blockIdx * BLOCK_SIZE + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0;

        if (col < n && blockIdx * BLOCK_SIZE + threadIdx.y < n)
            Bsub[threadIdx.y][threadIdx.x] = B[(blockIdx * BLOCK_SIZE + threadIdx.y) * n + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Compute matrix multiplication for the tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Helper function to fill matrices with random data
void fill_matrix(std::vector<float>& matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Measure time for each version
void compare_performance() {
    // Create matrices
    std::vector<float> A(N * N), B(N * N), C_cpu(N * N), C_cuda(N * N), C_shared(N * N);
    fill_matrix(A, N);
    fill_matrix(B, N);

    int num_iters = 5;

    // Measure CPU performance
    double cpu_time_total = 0.0;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        matmul_cpu(A, B, C_cpu, N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
        cpu_time_total += cpu_duration.count();
    }
    std::cout << "Average CPU Time: " << (cpu_time_total / num_iters)*1000 << "ms\n";

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Measure CUDA naive performance
    double cuda_naive_time_total = 0.0;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto start_cuda = std::chrono::high_resolution_clock::now();
        matmul_cuda_naive<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        auto end_cuda = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cuda_duration = end_cuda - start_cuda;
        cuda_naive_time_total += cuda_duration.count();
    }
    std::cout << "Average CUDA Naive Time: " << (cuda_naive_time_total / num_iters)*1000 << "ms\n";

    // Measure CUDA shared memory performance
    double cuda_shared_time_total = 0.0;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto start_shared = std::chrono::high_resolution_clock::now();
        matmul_cuda_shared<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        auto end_shared = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> shared_duration = end_shared - start_shared;
        cuda_shared_time_total += shared_duration.count();
    }
    std::cout << "Average CUDA Shared Memory Time: " << (cuda_shared_time_total / num_iters)*1000 << "ms\n";

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main() {
    compare_performance();
    return 0;
}
