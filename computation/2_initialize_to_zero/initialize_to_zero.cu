// Write CUDA program to intiailze an array of size 1024 and 1025
// Concept of more than one thread block
#include <iostream>
#include <cuda.h>
#include "cmath"
#define N 1025
__global__ void init(int *d_a, int n) {
    unsigned idx = threadIdx.x;
    if (idx < n) {
        d_a[idx] = 0;
    }
}

__global__ void add(int *d_a, int n) {
    unsigned idx = threadIdx.x;
    if (idx < n) {
        d_a[idx] += idx;
    }
}

int main() {
    int a[N], *d_a;
    //    a = (int*)malloc(sizeof(int) * N);
    cudaMalloc(&d_a,sizeof(int) * N);
//    unsigned grid_dim = ceil(double(N)/BLOCK_SIZE);
    unsigned block_dim = N;
    init<<<1,block_dim>>>(d_a,N);
    add<<<1,block_dim>>>(d_a,N);
    cudaMemcpy(a,d_a,sizeof(int)*N, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < N; i++){
        std::cout << a[i] << std::endl;
    }
    return 0;
}