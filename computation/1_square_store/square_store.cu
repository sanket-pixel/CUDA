// Write CUDA code to store squares from 0 to N in an aray

#include <iostream>
#include <cuda.h>
#include "cmath"
#define N 100
#define BLOCK_SIZE 1024
__global__ void square_store(int *d_a, int n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_a[idx] = idx * idx;
    }
}

int main() {
    int a[N], *d_a;
    //    a = (int*)malloc(sizeof(int) * N);
    cudaMalloc(&d_a,sizeof(int) * N);
    unsigned grid_dim = ceil(double(N)/BLOCK_SIZE);
    unsigned block_dim = BLOCK_SIZE;
    square_store<<<grid_dim,block_dim>>>(d_a,N);
    cudaMemcpy(a,d_a,sizeof(int)*N, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < N; i++){
        std::cout << a[i] << std::endl;
    }
    return 0;
}