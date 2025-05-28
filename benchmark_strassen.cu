#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "helper_matrix.h"
#include "sgemm.cuh"
#include "sgemm_strassen_fused.cuh"

// CUDA Headers
#include <cublas_v2.h>
#include <cuda_runtime.h>

struct BenchmarkResult {
    int M, N, K;
    double cublas_time_ms;
    double strassen_time_ms;
    double speedup;
    double gflops_cublas;
    double gflops_strassen;
};

double
calculate_gflops(int M, int N, int K, double time_ms) {
    double flops = 2.0 * M * N * K;            // 2 * M * N * K FLOPs for GEMM
    return (flops / 1e9) / (time_ms / 1000.0); // GFLOPs
}

int
main() {
    // Initialize CUDA and cuBLAS
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Benchmark configurations (powers of 2 for Strassen)
    std::vector<std::tuple<int, int, int>> configs = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048}, // If memory allows
    };

    std::vector<BenchmarkResult> results;
    const int warmup_runs = 3;
    const int benchmark_runs = 10;

    std::cout << "=== CUDA SGEMM Strassen vs cuBLAS Benchmark ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    for (const auto& [M, N, K] : configs) {
        std::cout << "\\nBenchmarking M=" << M << ", N=" << N << ", K=" << K << std::endl;

        // Check memory requirements
        size_t total_memory_needed = (size_t)M * K * sizeof(float) + (size_t)K * N * sizeof(float)
                                     + (size_t)M * N * sizeof(float);

        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);

        if (total_memory_needed > free_memory * 0.8) { // Use 80% of available memory
            std::cout << "Skipping due to insufficient memory" << std::endl;
            continue;
        }

        // Allocate matrices
        size_t lda = K, ldb = N, ldc = N;
        float alpha = 1.0f, beta = 0.0f;

        float* h_A = alloc_mat_host(M * lda * sizeof(float));
        float* h_B = alloc_mat_host(K * ldb * sizeof(float));

        float* d_A = alloc_mat_device(M * lda * sizeof(float));
        float* d_B = alloc_mat_device(K * ldb * sizeof(float));
        float* d_C_cublas = alloc_mat_device(M * ldc * sizeof(float));
        float* d_C_strassen = alloc_mat_device(M * ldc * sizeof(float));

        // Initialize with random data
        init_random(h_A, M * lda);
        init_random(h_B, K * ldb);

        checkCudaErrors(cudaMemcpy(d_A, h_A, M * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, K * ldb * sizeof(float), cudaMemcpyHostToDevice));

        // Warm up GPU
        for (int i = 0; i < warmup_runs; i++) {
            checkCudaErrors(cudaMemset(d_C_cublas, 0, M * ldc * sizeof(float)));
            CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     N,
                                     M,
                                     K,
                                     &alpha,
                                     d_B,
                                     ldb,
                                     d_A,
                                     lda,
                                     &beta,
                                     d_C_cublas,
                                     ldc));
            cudaDeviceSynchronize();
        }

        // Benchmark cuBLAS
        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < benchmark_runs; run++) {
            checkCudaErrors(cudaMemset(d_C_cublas, 0, M * ldc * sizeof(float)));
            CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     N,
                                     M,
                                     K,
                                     &alpha,
                                     d_B,
                                     ldb,
                                     d_A,
                                     lda,
                                     &beta,
                                     d_C_cublas,
                                     ldc));
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double cublas_time_ms = std::chrono::duration<double, std::milli>(end - start).count()
                                / benchmark_runs;

        // Warm up Strassen
        for (int i = 0; i < warmup_runs; i++) {
            checkCudaErrors(cudaMemset(d_C_strassen, 0, M * ldc * sizeof(float)));
            sgemm_strassen_1level_fused_vP(M,
                                           N,
                                           K,
                                           alpha,
                                           d_A,
                                           lda,
                                           d_B,
                                           ldb,
                                           beta,
                                           d_C_strassen,
                                           ldc,
                                           cublas_handle);
            cudaDeviceSynchronize();
        }

        // Benchmark Strassen
        start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < benchmark_runs; run++) {
            checkCudaErrors(cudaMemset(d_C_strassen, 0, M * ldc * sizeof(float)));
            sgemm_strassen_1level_fused_vP(M,
                                           N,
                                           K,
                                           alpha,
                                           d_A,
                                           lda,
                                           d_B,
                                           ldb,
                                           beta,
                                           d_C_strassen,
                                           ldc,
                                           cublas_handle);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        double strassen_time_ms = std::chrono::duration<double, std::milli>(end - start).count()
                                  / benchmark_runs;

        // Calculate metrics
        double speedup = cublas_time_ms / strassen_time_ms;
        double gflops_cublas = calculate_gflops(M, N, K, cublas_time_ms);
        double gflops_strassen = calculate_gflops(M, N, K, strassen_time_ms);

        BenchmarkResult result =
            {M, N, K, cublas_time_ms, strassen_time_ms, speedup, gflops_cublas, gflops_strassen};
        results.push_back(result);

        std::cout << "cuBLAS:   " << cublas_time_ms << " ms, " << gflops_cublas << " GFLOPs"
                  << std::endl;
        std::cout << "Strassen: " << strassen_time_ms << " ms, " << gflops_strassen << " GFLOPs"
                  << std::endl;
        std::cout << "Speedup:  " << speedup << "x" << std::endl;

        // Cleanup
        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C_cublas));
        checkCudaErrors(cudaFree(d_C_strassen));
    }

    // Print summary table
    std::cout << "\\n=== BENCHMARK SUMMARY ===" << std::endl;
    std::cout << "Size      | cuBLAS (ms) | Strassen (ms) | Speedup | cuBLAS (GF) | Strassen (GF)"
              << std::endl;
    std::cout << "----------|-------------|---------------|---------|-------------|-------------"
              << std::endl;

    for (const auto& result : results) {
        std::cout << std::setw(4) << result.M << "x" << std::setw(4) << result.N << " | "
                  << std::setw(11) << result.cublas_time_ms << " | " << std::setw(13)
                  << result.strassen_time_ms << " | " << std::setw(7) << result.speedup << " | "
                  << std::setw(11) << result.gflops_cublas << " | " << std::setw(11)
                  << result.gflops_strassen << std::endl;
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}