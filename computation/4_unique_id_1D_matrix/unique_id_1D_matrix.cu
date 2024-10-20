// Initialize 5x6 matrix to unique ID
// Concept of 2D matrix in CUDA using 1D representation
#include <iostream>
#include <cuda.h>
#include "cmath"
#define N 5
#define M 6
__global__ void init1d(int *dmatrix) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    dmatrix[idx] = idx;
    printf("%d ", idx);
}

int main() {
    int *hmatrix, *dmatrix;
    cudaMalloc(&dmatrix,sizeof(int) * N * M);
    hmatrix = (int*) malloc(sizeof(int)*M*N);
    init1d<<<N,M>>>(dmatrix);
    cudaMemcpy(hmatrix,dmatrix,sizeof(int)*N*M, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << hmatrix[i * M + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}