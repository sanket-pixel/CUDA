cmake_minimum_required(VERSION 3.18)
project(CudaAddExample LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Specify the CUDA architecture (optional, modify according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)  # For example, 75 is for Turing architecture (RTX 20 series)

# Set the output directory for the compiled executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Set the sources for the executable
add_executable(cuda_add_example basics.cu)

# Specify that this target requires CUDA
set_target_properties(cuda_add_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
