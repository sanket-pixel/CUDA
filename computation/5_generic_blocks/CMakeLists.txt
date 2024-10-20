cmake_minimum_required(VERSION 3.18)
project(CudaSquare LANGUAGES CXX CUDA)
# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set the sources for the executable
add_executable(cuda_square generic_blocks.cu)
# Specify that this target requires CUDA
set_target_properties(cuda_square PROPERTIES CUDA_ARCHITECTURES "86")
