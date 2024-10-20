// Initialize array of generic size with fixed block size
// concept of generic grid and block size
#include <iostream>
#include <cuda.h>
#include "cmath"
#define DATA_SIZE 2000
#define BLOCKSIZE 1024
__global__ void init1d(int *dmatrix) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < DATA_SIZE)
        dmatrix[idx] = idx;
}

int main() {
    int *hdata, *ddata;
    cudaMalloc(&ddata,sizeof(int) * DATA_SIZE );
    hdata = (int*) malloc(sizeof(int)*DATA_SIZE);
    int grid_dim = ceil(double(DATA_SIZE)/BLOCKSIZE);
    init1d<<<grid_dim,BLOCKSIZE>>>(ddata);
    cudaMemcpy(hdata,ddata,sizeof(int)*DATA_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << std::endl;
    for (int i = 0; i < DATA_SIZE; i++) {
        std::cout << hdata[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}