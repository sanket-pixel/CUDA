cmake_minimum_required(VERSION 3.18)
project(CudaAddExample LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
# Set the output directory for the compiled executable
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
# Set the sources for the executable
add_executable(cuda_blur blur.cu)
target_include_directories(cuda_blur PUBLIC ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(cuda_blur PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES})
# Specify that this target requires CUDA
set_target_properties(cuda_blur PROPERTIES CUDA_ARCHITECTURES "86")
