#include <iostream>
#include <cuda.h>
#include <chrono>
#include "opencv2/opencv.hpp"
#define ARRAY_SIZE 1000000
#define BLOCK_SIZE 1024

__global__ void blur_shared(unsigned char* d_mat, unsigned width, unsigned height, unsigned window) {
    extern __shared__ unsigned char shared_mem[];  // Shared memory array for block's pixels

    // Thread and block indices
    unsigned global_x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned global_y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned pixel_idx = width * global_y + global_x;

    // Half window size for boundary checks
    unsigned half_window = window / 2;

    // Calculate shared memory dimensions (block size + halo region)
    unsigned shared_x = threadIdx.x + half_window;
    unsigned shared_y = threadIdx.y + half_window;

    // Load pixels into shared memory (including halo/border pixels)
    if (global_x < width && global_y < height) {
        // Global memory pixel index
        unsigned global_idx = 3 * (global_y * width + global_x);

        // Load center pixel
        shared_mem[(shared_y * (blockDim.x + window - 1) + shared_x) * 3] = d_mat[global_idx];
        shared_mem[(shared_y * (blockDim.x + window - 1) + shared_x) * 3 + 1] = d_mat[global_idx + 1];
        shared_mem[(shared_y * (blockDim.x + window - 1) + shared_x) * 3 + 2] = d_mat[global_idx + 2];

        // Load halo pixels (check for boundary conditions)
        if (threadIdx.x < half_window) {
            // Left halo
            if (global_x >= half_window) {
                unsigned halo_left_idx = 3 * (global_y * width + (global_x - half_window));
                shared_mem[(shared_y * (blockDim.x + window - 1) + threadIdx.x) * 3] = d_mat[halo_left_idx];
                shared_mem[(shared_y * (blockDim.x + window - 1) + threadIdx.x) * 3 + 1] = d_mat[halo_left_idx + 1];
                shared_mem[(shared_y * (blockDim.x + window - 1) + threadIdx.x) * 3 + 2] = d_mat[halo_left_idx + 2];
            }
        }

        if (threadIdx.y < half_window) {
            // Top halo
            if (global_y >= half_window) {
                unsigned halo_top_idx = 3 * ((global_y - half_window) * width + global_x);
                shared_mem[(threadIdx.y * (blockDim.x + window - 1) + shared_x) * 3] = d_mat[halo_top_idx];
                shared_mem[(threadIdx.y * (blockDim.x + window - 1) + shared_x) * 3 + 1] = d_mat[halo_top_idx + 1];
                shared_mem[(threadIdx.y * (blockDim.x + window - 1) + shared_x) * 3 + 2] = d_mat[halo_top_idx + 2];
            }
        }
    }

    // Wait for all threads to load data into shared memory
    __syncthreads();

    // Perform blur operation within the valid pixel area
    if (global_x >= half_window && global_y >= half_window && global_x < width - half_window && global_y < height - half_window) {
        int sum_red = 0, sum_green = 0, sum_blue = 0;

        // Loop over the window and accumulate pixel values
        for (int j = 0; j < window; ++j) {
            for (int i = 0; i < window; ++i) {
                unsigned local_idx = ((shared_y + j - half_window) * (blockDim.x + window - 1) + (shared_x + i - half_window)) * 3;
                sum_red += shared_mem[local_idx];
                sum_green += shared_mem[local_idx + 1];
                sum_blue += shared_mem[local_idx + 2];
            }
        }

        unsigned num_pixels = window * window;
        d_mat[pixel_idx * 3] = sum_red / num_pixels;
        d_mat[(pixel_idx * 3) + 1] = sum_green / num_pixels;
        d_mat[(pixel_idx * 3) + 2] = sum_blue / num_pixels;
    }
}


__global__ void cuda_blur(unsigned char* d_mat, unsigned width, unsigned height, unsigned window){
    unsigned global_x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned global_y =  blockDim.y * blockIdx.y + threadIdx.y;
    unsigned pixel_idx = width * global_y + global_x;
    int sum_red = 0;
    int sum_green = 0;
    int sum_blue = 0;
    if(global_x > window/2 && global_y > window/2 && global_x < width - window/2 && global_y < height - window/2){
        unsigned top_left_x = global_x - window/2;
        unsigned top_left_y = global_y - window/2;
        for (int j = 0; j < window; j++)
        {
            for(int i = 0; i<window; i++){
                int neighbor_index = width * (top_left_y + j) + (top_left_x + i) ;
                sum_red += d_mat[ neighbor_index * 3 ] ;
                sum_green += d_mat[(neighbor_index * 3) + 1] ;
                sum_blue += d_mat[(neighbor_index * 3)+ 2] ;
            }
        }
        d_mat[pixel_idx * 3] = sum_red/(window*window);
        d_mat[(pixel_idx * 3) + 1] = sum_green/(window*window);
        d_mat[(pixel_idx * 3) + 2] = sum_blue/(window*window);
    }
}

void cpu_blur(cv::Mat& image, unsigned cols, unsigned rows, unsigned window_size) {
    // Create a copy of the image to store the output
    cv::Mat output = image.clone();

    // Calculate half window size for convenience
    unsigned half_window = window_size / 2;

    // Loop through each pixel in the image
    for (unsigned y = half_window; y < rows - half_window; y++) {
        for (unsigned x = half_window; x < cols - half_window; x++) {

            // Initialize sums for B, G, R channels
            int sum_blue = 0, sum_green = 0, sum_red = 0;

            // Loop over the window around the pixel
            for (int wy = -half_window; wy <= half_window; wy++) {
                for (int wx = -half_window; wx <= half_window; wx++) {
                    cv::Vec3b pixel = image.at<cv::Vec3b>(y + wy, x + wx);
                    sum_blue += pixel[0];
                    sum_green += pixel[1];
                    sum_red += pixel[2];
                }
            }

            // Compute the mean for each channel and assign it to the output image
            int window_area = window_size * window_size;
            output.at<cv::Vec3b>(y, x)[0] = sum_blue / window_area;
            output.at<cv::Vec3b>(y, x)[1] = sum_green / window_area;
            output.at<cv::Vec3b>(y, x)[2] = sum_red / window_area;
        }
    }

    // Copy the result back to the original image
    image = output;
}


int main() {
    // Read the image
    cv::Mat image = cv::imread("../dogs.jpg");
    if (image.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }

    // Define window size and iteration count
    unsigned window_size = 3;
    int iterations = 100;

    // ==================== CPU BLUR ====================
    cv::Mat cpu_image = image.clone();
    double cpu_total_time = 0.0;
    for (int i = 0; i < iterations; ++i) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_blur(cpu_image, cpu_image.cols, cpu_image.rows, window_size);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
        cpu_total_time += cpu_duration.count();
    }
    std::cout << "Average CPU blur time: " << (cpu_total_time / iterations) * 1000 << "ms" << std::endl;

    // ==================== GPU BLUR ====================
    unsigned char* d_mat, *d_out;
    cudaMalloc(&d_mat, sizeof(unsigned char) * image.cols * image.rows * image.channels());
    cudaMalloc(&d_out, sizeof(unsigned char) * image.cols * image.rows * image.channels());
    float gpu_total_time = 0.0;
    for (int i = 0; i < iterations; ++i) {
        cudaMemcpy(d_mat, image.data, sizeof(unsigned char) * image.cols * image.rows * image.channels(), cudaMemcpyHostToDevice);
        dim3 blockSize(32, 32);
        dim3 gridSize((image.cols + blockSize.x - 1) / blockSize.x, (image.rows + blockSize.y - 1) / blockSize.y);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        size_t shared_mem_size = (blockSize.x + window_size - 1) * (blockSize.y + window_size - 1) * 3 * sizeof(unsigned char);
        cuda_blur<<<gridSize, blockSize, shared_mem_size>>>(d_mat, image.cols, image.rows, window_size);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        gpu_total_time += gpu_time / 1000.0;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    std::cout << "Average GPU blur time: " << (gpu_total_time / iterations) * 1000 << "ms" << std::endl;
    cudaFree(d_mat);

    return 0;
}