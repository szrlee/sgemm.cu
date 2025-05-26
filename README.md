# High-Performance SGEMM on NVIDIA GPUs

This project provides implementations of Single-Precision General Matrix Multiplication (SGEMM) optimized for NVIDIA GPUs. It includes a standard high-performance SGEMM kernel and explores advanced Strassen-based algorithms for potential further speedups on large matrices.

> **Important note:** while the implementations are expected to be high-performant on supported Ada/Ampere/Volta/Turing devices, they were specifically fine-tuned for and tested on NVIDIA RTX 3090 (GA102 chip - RTX 3080, A10, A40, A6000).

## Implemented Algorithms

1.  **Standard High-Performance SGEMM (`src/sgemm.cuh`)**
    *   This is a highly optimized conventional SGEMM implementation, serving as a baseline. It uses techniques like shared memory tiling, warp-level operations, and careful instruction scheduling.

2.  **1-Level Fused Strassen SGEMM (`sgemm_strassen_1level_fused_vP` in `src/sgemm_strassen_fused.cuh`)**
    *   **Core Idea:** Implements one level of the Strassen algorithm. The key feature is its "fused" kernel (`strassen_fused_128x128x8_kernel` from "Version P"), which performs the additions/subtractions for matrix sub-blocks (e.g., $X \pm Y$) and the subsequent matrix multiplication $M_{tile} = (X \pm \delta Y)(V \pm \epsilon W)$ within a single kernel launch, loading data for X, Y, V, W directly and producing parts of the $M_{tile}$ result.
    *   **Workspace:** This 1-level implementation is designed to be workspace-efficient at its level of recursion, meaning it doesn't require extra global memory buffers for intermediate $S_i$ or $T_i$ matrices if used as a direct replacement for a standard GEMM. The 7 subproblems are computed and accumulated directly to the output C submatrices.
    *   **Kernel:** Based on the "Version P" kernel design, which uses dynamic shared memory.

3.  **Hybrid 2-Level Strassen SGEMM (`sgemm_strassen_hybrid_2level_detailed` in `src/sgemm_hybrid_2level_detailed.cuh`)**
    *   **Architecture:** This implementation applies Strassen's algorithm recursively for two levels.
        *   **Outer Level:** Uses conventional Strassen. The necessary sums and differences of submatrices of A and B (i.e., $S_i$ and $T_i$ terms) are computed using cuBLAS helper functions (`matrix_add_gpu`, `matrix_subtract_gpu` from `src/cublas_helpers.cuh`).
        *   **Inner Level:** The 7 matrix products ($M_i$) required by the outer Strassen level are computed using the refactored `sgemm_strassen_1level_fused_vP` (which operates on device pointers).
    *   **Workspace:** This version **requires significant GPU workspace** to store the intermediate $S_i, T_i,$ and $M_i$ matrices from the outer Strassen layer. This is consistent with traditional Strassen algorithm implementations.
    *   **Performance Potential:** Designed to potentially offer higher peak performance for larger matrices where the reduction in FLOPs from two levels of Strassen can outweigh the overheads of workspace management and additional kernel calls.
    *   **Dimensionality Requirement:** Matrix dimensions M, N, and K must be divisible by 4 for this implementation.
    *   **Structure:** Based on "Version D detailed" which involves explicit management of temporary matrices and CUDA streams for parallelism.

## Benchmark

>Avoid using WSL for performance measurements. To ensure accurate and reliable results, please use a native Linux environment.

To benchmark the different SGEMM implementations, specify the compute capability of your CUDA device and run `benchmark.sh`. For example, on RTX 3090:

```bash
bash benchmark.sh 86
```

The benchmark settings such as minimum/maximum matrix sizes (default 512 to 4096, step 256), step size, number of warm-up iterations etc. can be adjusted in the `benchmark.sh` file.

The benchmark output (`benchmark_results/all_sgemm_benchmarks_final.txt`) will contain timings for:
1.  **Standard SGEMM:** GPU-centric execution time.
2.  **Strassen 1-Level (vP, GPU-centric):** Execution time of `sgemm_strassen_1level_fused_vP` operating on pre-allocated device memory (measures kernel performance excluding H2D/D2H for its main inputs/outputs).
3.  **Strassen 2-Level Hybrid (Full):** End-to-end execution time of `sgemm_strassen_hybrid_2level_detailed`, including its internal H2D/D2H copies and workspace management.

To visualize benchmark results, please install `matplotlib` and run:

```bash
python plot_benchmark_data.py benchmark_results/all_sgemm_benchmarks_final.txt 
```
(Adjust filename if necessary)

## Tests

Use `test.sh` to test all implementations for correctness against reference CPU and GPU GEMM computations. For example, on RTX 3090:

```bash
bash test.sh 86
```

The `test.cu` executable (run via `test.sh`) now includes tests for:
- Standard SGEMM (`sgemm` vs `sgemm_basic`).
- 1-Level Fused Strassen (`sgemm_strassen_1level_fused_vP`) vs standard GPU SGEMM.
- Hybrid 2-Level Strassen ("Version C" - `sgemm_strassen_hybrid_2level_vC`) vs standard GPU SGEMM.

A separate, more detailed test suite for the "Version D detailed" Hybrid 2-Level Strassen implementation (`sgemm_strassen_hybrid_2level_detailed`) is available in `test_hybrid_detailed.cu`. This can be built and run separately (e.g., by adding it as a target in `CMakeLists.txt` and creating a corresponding `test_detailed.sh` or similar). For general verification of multiple versions, `test.sh` covers the primary implementations.

## Building the Code

The project uses CMake for building. The `benchmark.sh` and `test.sh` scripts handle the CMake configuration and build process. Ensure you have a compatible CUDA toolkit and C++ compiler installed. New files like `src/sgemm_hybrid_2level_detailed.cuh`, `src/sgemm_hybrid_2level_vC.cuh`, `src/cublas_helpers.cuh`, and `test_hybrid_detailed.cu` are expected to be compiled by the existing CMake setup if appropriately added or covered by glob patterns.

## Key Files

*   `src/sgemm.cuh`: Standard high-performance SGEMM kernel and launcher.
*   `src/kernels/strassen_fused_128x128x8.cuh`: Contains the "Version P" fused Strassen device kernel (`strassen_fused_kernel_128x128x8_vP`).
*   `src/sgemm_strassen_fused.cuh`: Host launcher for the 1-Level Fused Strassen (`sgemm_strassen_1level_fused_vP`).
*   `src/cublas_helpers.cuh`: Helper functions wrapping cuBLAS routines (e.g., `matrix_add_gpu`).
*   `src/sgemm_hybrid_2level_vC.cuh`: Host launcher for the Hybrid 2-Level Strassen ("Version C" structure, `sgemm_strassen_hybrid_2level_vC`). (This was the precursor to detailed).
*   `src/sgemm_hybrid_2level_detailed.cuh`: Host launcher for the Hybrid 2-Level Strassen ("Version D" detailed structure, `sgemm_strassen_hybrid_2level_detailed`).
*   `test.cu`: Main test suite covering all SGEMM versions.
*   `test_hybrid_detailed.cu`: Specific detailed test suite for `sgemm_strassen_hybrid_2level_detailed`.
*   `benchmark.cu`: Comprehensive benchmark suite for comparing all SGEMM versions.
*   `common/`: Contains helper utilities for CUDA, matrices, strings.
*   `CMakeLists.txt`: Build configuration.
*   `benchmark.sh`, `test.sh`: Scripts for easy benchmarking and testing.

## Performance

Test environment:

- OS: Ubuntu 24.04.1 LTS
- GPU: NVIDIA RTX 3090
- Driver Version: 550.120
- CUDA Driver: 12.4, CUDA Runtime: 12.6, V12.6.85
- CMake 3.28.3
- g++ 13.3

<p align="center">
  <img src="assets/perf.png" alt="perf" width="85%">
</p>

<p align="center">
  <img src="assets/perf_locked.png" alt="perf" width="85%">
</p>
