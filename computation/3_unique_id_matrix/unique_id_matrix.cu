// Initialize 5x6 matrix to unique ID
// Concept of 2D matrix in CUDA
#include <iostream>
#include <cuda.h>
#include "cmath"
#define N 5
#define M 6
__global__ void init2d(int *dmatrix, int n, int m) {
    unsigned idx = blockDim.x * threadIdx.y + threadIdx.x;
    dmatrix[idx] = idx;
}

int main() {
    int *hmatrix, *dmatrix;
    cudaMalloc(&dmatrix,sizeof(int) * N * M);
    hmatrix = (int*) malloc(sizeof(int)*M*N);
    dim3 block_dim(M,N,1);
    init2d<<<1,block_dim>>>(dmatrix,N,M);
    cudaMemcpy(hmatrix,dmatrix,sizeof(int)*N*M, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << hmatrix[i * M + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}