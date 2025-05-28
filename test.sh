#!/bin/bash

matsize_max=64  # Reduced for quicker sanitizer run
matsize_min=64  # Focus on the failing case
matsize_step=1

save_dir="test_results"
mkdir -p ${save_dir}

# It seems you don't have a variable test_save_dir, using save_dir
# Also, DGPUCC=$1 might be problematic if $1 is not passed. Assuming nvcc for now.

export SGEMM_SPECIFIC_STRASSEN_TEST_ID="64x64x64_beta0_DEBUG_ONLY"

sudo rm -r -f build
# Added -DCMAKE_BUILD_TYPE=Debug for more debug info
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target test --verbose

echo "Running test normally first:"
./build/test --savedir=${save_dir} --mmax=${matsize_max} --mmin=${matsize_min} --mstep=${matsize_step}

echo "\nRunning test with compute-sanitizer --tool memcheck:"
compute-sanitizer --tool memcheck ./build/test --savedir=${save_dir} --mmax=${matsize_max} --mmin=${matsize_min} --mstep=${matsize_step}

echo "\nRunning test with compute-sanitizer --tool synccheck:"
compute-sanitizer --tool synccheck ./build/test --savedir=${save_dir} --mmax=${matsize_max} --mmin=${matsize_min} --mstep=${matsize_step}