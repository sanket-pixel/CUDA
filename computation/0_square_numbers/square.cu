#include <iostream>
#include <cuda.h>
#include "cmath"
#define N 100
#define BLOCK_SIZE 1024
__global__ void square(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        printf("%d \n", idx * idx);
    }
}

int main() {
    unsigned grid_dim = ceil(double(N)/BLOCK_SIZE);
    unsigned block_dim = BLOCK_SIZE;
    square<<<grid_dim,block_dim>>>(N);
    cudaDeviceSynchronize();
    return 0;
}