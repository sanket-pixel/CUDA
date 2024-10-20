// Initialize array of generic size with fixed block size
// concept of generic grid and block size
#include <iostream>
#include <cuda.h>
#include "cmath"
#include "vector"
#define N 3
#define BLOCKSIZE 1024
struct Point{
    float x,y;
};


__global__ void pairwise_distance(Point *dpoints, float *ddistance) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned point_x = idx % N;
    unsigned point_y = floor(float(idx)/N);
    if( idx < N * N)
    {   float x2 = dpoints[point_y].x;
        float x1 = dpoints[point_x].x;

        float y2 = dpoints[point_y].y;
        float y1 = dpoints[point_x].y;

        float distance = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
        ddistance[point_y * N + point_x] = distance;
    }

}



int main() {
    Point hpoints[N] = {{1.0f,2.0f},{2.0f,3.0f},{0.0f,0.0f}}, *dpoints;
    float *hdistance, *ddistance;
    // points
    cudaMalloc(&dpoints,sizeof(Point) * N);

    // pairwise distance
    cudaMalloc(&ddistance,sizeof(float) * N *N );
    hdistance = (float*) malloc(sizeof(float) * N * N);

    cudaMemcpy(dpoints,hpoints,sizeof(Point)* N, cudaMemcpyKind::cudaMemcpyHostToDevice);
    int grid_dim = ceil(double(N * N)/BLOCKSIZE);
    pairwise_distance<<<grid_dim, BLOCKSIZE>>>(dpoints, ddistance);
    cudaMemcpy(hdistance,ddistance,sizeof(float)* N * N, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << hdistance[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}