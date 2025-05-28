#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>

#include "common/helper_matrix.h"
#include "src/sgemm_strassen_fused_single.cuh"

// CUDA Headers
#include <cublas_v2.h>
#include <cuda_runtime.h>

double
get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

int
main() {
    printf("========================================================\n");
    printf("Benchmarking Both Strassen Kernels vs cuBLAS\n");
    printf("========================================================\n");

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    int test_sizes[] = {64, 128, 256, 512, 1024};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int warmup_runs = 3;
    int benchmark_runs = 10;

    for (int test_idx = 0; test_idx < num_tests; test_idx++) {
        int M = test_sizes[test_idx];
        int N = test_sizes[test_idx];
        int K = test_sizes[test_idx];

        printf("\n=== Matrix Size: %dx%dx%d ===\n", M, N, K);

        float alpha = 1.0f;
        float beta = 0.0f;
        size_t lda = K;
        size_t ldb = N;
        size_t ldc = N;

        // Allocate host memory
        float* h_A = alloc_mat_host(M * lda * sizeof(float));
        float* h_B = alloc_mat_host(K * ldb * sizeof(float));
        float* h_C_cublas = alloc_mat_host(M * ldc * sizeof(float));
        float* h_C_strassen_simple = alloc_mat_host(M * ldc * sizeof(float));
        float* h_C_strassen_optimized = alloc_mat_host(M * ldc * sizeof(float));

        // Initialize matrices
        init_random(h_A, M * K);
        init_random(h_B, K * N);
        init_const(h_C_cublas, M * N, 0.0f);
        init_const(h_C_strassen_simple, M * N, 0.0f);
        init_const(h_C_strassen_optimized, M * N, 0.0f);

        // Allocate device memory
        float* d_A = alloc_mat_device(M * lda * sizeof(float));
        float* d_B = alloc_mat_device(K * ldb * sizeof(float));
        float* d_C_cublas = alloc_mat_device(M * ldc * sizeof(float));
        float* d_C_strassen_simple = alloc_mat_device(M * ldc * sizeof(float));
        float* d_C_strassen_optimized = alloc_mat_device(M * ldc * sizeof(float));

        // Copy to device
        checkCudaErrors(cudaMemcpy(d_A, h_A, M * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, K * ldb * sizeof(float), cudaMemcpyHostToDevice));

        // Warmup runs
        for (int i = 0; i < warmup_runs; i++) {
            checkCudaErrors(cudaMemcpy(d_C_cublas,
                                       h_C_cublas,
                                       M * ldc * sizeof(float),
                                       cudaMemcpyHostToDevice));
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

            checkCudaErrors(cudaMemcpy(d_C_strassen_simple,
                                       h_C_strassen_simple,
                                       M * ldc * sizeof(float),
                                       cudaMemcpyHostToDevice));
            sgemm_strassen_fused_single(M,
                                        N,
                                        K,
                                        alpha,
                                        d_A,
                                        lda,
                                        d_B,
                                        ldb,
                                        beta,
                                        d_C_strassen_simple,
                                        ldc,
                                        cublas_handle);

            checkCudaErrors(cudaMemcpy(d_C_strassen_optimized,
                                       h_C_strassen_optimized,
                                       M * ldc * sizeof(float),
                                       cudaMemcpyHostToDevice));
            sgemm_strassen_fused_single_optimized(M,
                                                  N,
                                                  K,
                                                  alpha,
                                                  d_A,
                                                  lda,
                                                  d_B,
                                                  ldb,
                                                  beta,
                                                  d_C_strassen_optimized,
                                                  ldc,
                                                  cublas_handle);
        }
        checkCudaErrors(cudaDeviceSynchronize());

        // Benchmark cuBLAS
        double cublas_total_time = 0.0;
        for (int i = 0; i < benchmark_runs; i++) {
            checkCudaErrors(cudaMemcpy(d_C_cublas,
                                       h_C_cublas,
                                       M * ldc * sizeof(float),
                                       cudaMemcpyHostToDevice));

            double start_time = get_time_ms();
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
            checkCudaErrors(cudaDeviceSynchronize());
            double end_time = get_time_ms();

            cublas_total_time += (end_time - start_time);
        }

        // Benchmark Simple Strassen
        double strassen_simple_total_time = 0.0;
        for (int i = 0; i < benchmark_runs; i++) {
            checkCudaErrors(cudaMemcpy(d_C_strassen_simple,
                                       h_C_strassen_simple,
                                       M * ldc * sizeof(float),
                                       cudaMemcpyHostToDevice));

            double start_time = get_time_ms();
            sgemm_strassen_fused_single(M,
                                        N,
                                        K,
                                        alpha,
                                        d_A,
                                        lda,
                                        d_B,
                                        ldb,
                                        beta,
                                        d_C_strassen_simple,
                                        ldc,
                                        cublas_handle);
            checkCudaErrors(cudaDeviceSynchronize());
            double end_time = get_time_ms();

            strassen_simple_total_time += (end_time - start_time);
        }

        // Benchmark Optimized Strassen
        double strassen_optimized_total_time = 0.0;
        for (int i = 0; i < benchmark_runs; i++) {
            checkCudaErrors(cudaMemcpy(d_C_strassen_optimized,
                                       h_C_strassen_optimized,
                                       M * ldc * sizeof(float),
                                       cudaMemcpyHostToDevice));

            double start_time = get_time_ms();
            sgemm_strassen_fused_single_optimized(M,
                                                  N,
                                                  K,
                                                  alpha,
                                                  d_A,
                                                  lda,
                                                  d_B,
                                                  ldb,
                                                  beta,
                                                  d_C_strassen_optimized,
                                                  ldc,
                                                  cublas_handle);
            checkCudaErrors(cudaDeviceSynchronize());
            double end_time = get_time_ms();

            strassen_optimized_total_time += (end_time - start_time);
        }

        // Calculate averages
        double cublas_avg_time = cublas_total_time / benchmark_runs;
        double strassen_simple_avg_time = strassen_simple_total_time / benchmark_runs;
        double strassen_optimized_avg_time = strassen_optimized_total_time / benchmark_runs;

        // Calculate GFLOPs
        double operations = 2.0 * M * N * K; // FMA count
        double cublas_gflops = (operations / (cublas_avg_time * 1e-3)) / 1e9;
        double strassen_simple_gflops = (operations / (strassen_simple_avg_time * 1e-3)) / 1e9;
        double strassen_optimized_gflops = (operations / (strassen_optimized_avg_time * 1e-3))
                                           / 1e9;

        // Calculate speedups
        double simple_speedup = cublas_avg_time / strassen_simple_avg_time;
        double optimized_speedup = cublas_avg_time / strassen_optimized_avg_time;
        double opt_vs_simple_speedup = strassen_simple_avg_time / strassen_optimized_avg_time;

        printf("cuBLAS:            %.3f ms, %.1f GFLOPs\n", cublas_avg_time, cublas_gflops);
        printf("Strassen Simple:   %.3f ms, %.1f GFLOPs\n",
               strassen_simple_avg_time,
               strassen_simple_gflops);
        printf("Strassen Optimized:%.3f ms, %.1f GFLOPs\n",
               strassen_optimized_avg_time,
               strassen_optimized_gflops);
        printf("Simple vs cuBLAS:  %.2fx %s\n",
               simple_speedup > 1.0 ? simple_speedup : -1.0 / simple_speedup,
               simple_speedup > 1.0 ? "(Simple faster)" : "(cuBLAS faster)");
        printf("Optimized vs cuBLAS: %.2fx %s\n",
               optimized_speedup > 1.0 ? optimized_speedup : -1.0 / optimized_speedup,
               optimized_speedup > 1.0 ? "(Optimized faster)" : "(cuBLAS faster)");
        printf("Optimized vs Simple: %.2fx %s\n",
               opt_vs_simple_speedup > 1.0 ? opt_vs_simple_speedup : -1.0 / opt_vs_simple_speedup,
               opt_vs_simple_speedup > 1.0 ? "(Optimized faster)" : "(Simple faster)");

        // Verify correctness
        checkCudaErrors(
            cudaMemcpy(h_C_cublas, d_C_cublas, M * ldc * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_C_strassen_simple,
                                   d_C_strassen_simple,
                                   M * ldc * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_C_strassen_optimized,
                                   d_C_strassen_optimized,
                                   M * ldc * sizeof(float),
                                   cudaMemcpyDeviceToHost));

        cmp_result result_simple =
            compare_mats(h_C_cublas, h_C_strassen_simple, M * N, 1e-3f, false, false);
        cmp_result result_optimized =
            compare_mats(h_C_cublas, h_C_strassen_optimized, M * N, 1e-3f, false, false);
        printf("Simple Correctness:    %s\n", result_simple.equal ? "PASS" : "FAIL");
        printf("Optimized Correctness: %s\n", result_optimized.equal ? "PASS" : "FAIL");

        // Cleanup
        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C_cublas));
        checkCudaErrors(cudaFreeHost(h_C_strassen_simple));
        checkCudaErrors(cudaFreeHost(h_C_strassen_optimized));
        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C_cublas));
        checkCudaErrors(cudaFree(d_C_strassen_simple));
        checkCudaErrors(cudaFree(d_C_strassen_optimized));
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}