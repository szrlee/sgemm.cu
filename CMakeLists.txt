cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgemm_fast LANGUAGES CUDA CXX)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/common")
include_directories("${PROJECT_SOURCE_DIR}/src")

if((NOT DEFINED GPUCC) OR (GPUCC STREQUAL ""))
  set(GPUCC 86)
endif()

if((NOT DEFINED CUBLAS) OR (CUBLAS STREQUAL ""))
  set(CUBLAS 0)
endif()

add_executable(benchmark benchmark.cu)
set_target_properties(benchmark PROPERTIES CUDA_ARCHITECTURES "${GPUCC}" CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(benchmark -lcudart -lcublas)

add_executable(test test.cu)
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "${GPUCC}" CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test -lcudart -lcublas)

add_executable(benchmark_strassen benchmark_strassen.cu)
set_target_properties(benchmark_strassen PROPERTIES CUDA_ARCHITECTURES "${GPUCC}" CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(benchmark_strassen -lcudart -lcublas)

add_executable(test_optimized_strassen test_optimized_strassen.cu)
set_target_properties(test_optimized_strassen PROPERTIES CUDA_ARCHITECTURES "${GPUCC}" CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_optimized_strassen -lcudart -lcublas)

add_executable(test_simple_strassen test_simple_strassen.cu)
set_target_properties(test_simple_strassen PROPERTIES CUDA_ARCHITECTURES "${GPUCC}" CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_simple_strassen -lcudart -lcublas)

add_executable(benchmark_optimized_strassen benchmark_optimized_strassen.cu)
set_target_properties(benchmark_optimized_strassen PROPERTIES CUDA_ARCHITECTURES "${GPUCC}" CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(benchmark_optimized_strassen -lcudart -lcublas)

target_compile_options(benchmark PRIVATE -DGPUCC=${GPUCC} -DCUBLAS=${CUBLAS})
target_compile_options(test PRIVATE -DGPUCC=${GPUCC})
target_compile_options(benchmark_strassen PRIVATE -DGPUCC=${GPUCC})
target_compile_options(test_optimized_strassen PRIVATE -DGPUCC=${GPUCC})

# target_compile_options(bench PRIVATE --ptxas-options=-v -O3)
# target_compile_options(test PRIVATE -Xptxas -O3)
