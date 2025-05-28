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

void
print_matrix(const char* title, const float* matrix, int rows, int cols, int ld, int max_print) {
    printf("\n%s (showing up to %dx%d):\n", title, max_print, max_print);
    for (int i = 0; i < rows && i < max_print; ++i) {
        for (int j = 0; j < cols && j < max_print; ++j) {
            printf("%8.3f ", matrix[i * ld + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int
main() {
    printf("=========================================================\n");
    printf("Testing Optimized Strassen Fused Single Kernel\n");
    printf("=========================================================\n");

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Test configurations
    int test_sizes[] = {64, 128, 256, 512};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int passed = 0;
    int failed = 0;

    for (int test_idx = 0; test_idx < num_tests; test_idx++) {
        int M = test_sizes[test_idx];
        int N = test_sizes[test_idx];
        int K = test_sizes[test_idx];

        printf("\n--- Test %d: M=%d, N=%d, K=%d ---\n", test_idx + 1, M, N, K);

        // Basic SGEMM parameters
        float alpha = 1.0f;
        float beta = 0.0f;
        size_t lda = K;
        size_t ldb = N;
        size_t ldc = N;

        // Allocate host memory
        float* h_A = alloc_mat_host(M * lda * sizeof(float));
        float* h_B = alloc_mat_host(K * ldb * sizeof(float));
        float* h_C_ref = alloc_mat_host(M * ldc * sizeof(float));
        float* h_C_optimized = alloc_mat_host(M * ldc * sizeof(float));
        float* h_C_simple = alloc_mat_host(M * ldc * sizeof(float));

        // Initialize matrices
        init_random(h_A, M * K);
        init_random(h_B, K * N);
        init_const(h_C_ref, M * N, 0.0f);
        init_const(h_C_optimized, M * N, 0.0f);
        init_const(h_C_simple, M * N, 0.0f);

        // Allocate device memory
        float* d_A = alloc_mat_device(M * lda * sizeof(float));
        float* d_B = alloc_mat_device(K * ldb * sizeof(float));
        float* d_C_ref = alloc_mat_device(M * ldc * sizeof(float));
        float* d_C_optimized = alloc_mat_device(M * ldc * sizeof(float));
        float* d_C_simple = alloc_mat_device(M * ldc * sizeof(float));

        // Copy to device
        checkCudaErrors(cudaMemcpy(d_A, h_A, M * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, K * ldb * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(d_C_ref, h_C_ref, M * ldc * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_C_optimized,
                                   h_C_optimized,
                                   M * ldc * sizeof(float),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(d_C_simple, h_C_simple, M * ldc * sizeof(float), cudaMemcpyHostToDevice));

        // Run cuBLAS reference
        printf("  Running cuBLAS reference...\n");
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
                                 d_C_ref,
                                 ldc));

        // Run optimized Strassen kernel
        printf("  Running optimized Strassen kernel...\n");
        sgemm_strassen_fused_single_optimized(M,
                                              N,
                                              K,
                                              alpha,
                                              d_A,
                                              lda,
                                              d_B,
                                              ldb,
                                              beta,
                                              d_C_optimized,
                                              ldc,
                                              cublas_handle);

        // Run simple Strassen kernel
        printf("  Running simple Strassen kernel...\n");
        sgemm_strassen_fused_single(M,
                                    N,
                                    K,
                                    alpha,
                                    d_A,
                                    lda,
                                    d_B,
                                    ldb,
                                    beta,
                                    d_C_simple,
                                    ldc,
                                    cublas_handle);

        checkCudaErrors(cudaDeviceSynchronize());

        // Copy results back
        checkCudaErrors(
            cudaMemcpy(h_C_ref, d_C_ref, M * ldc * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_C_optimized,
                                   d_C_optimized,
                                   M * ldc * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(
            cudaMemcpy(h_C_simple, d_C_simple, M * ldc * sizeof(float), cudaMemcpyDeviceToHost));

        // Debug print for small matrices
        if (M <= 8) {
            print_matrix("Input A", h_A, M, K, lda, 8);
            print_matrix("Input B", h_B, K, N, ldb, 8);
            print_matrix("cuBLAS Reference", h_C_ref, M, N, ldc, 8);
            print_matrix("Optimized Strassen", h_C_optimized, M, N, ldc, 8);
            print_matrix("Simple Strassen", h_C_simple, M, N, ldc, 8);
        }

        // Compare results
        const float tolerance = 1e-3f;
        cmp_result result_opt = compare_mats(h_C_ref, h_C_optimized, M * N, tolerance, false, true);
        cmp_result result_simple = compare_mats(h_C_ref, h_C_simple, M * N, tolerance, false, true);

        bool test_passed = result_opt.equal && result_simple.equal;
        if (test_passed) {
            passed++;
            printf("  RESULT: PASSED\n");
        } else {
            failed++;
            printf("  RESULT: FAILED\n");
            if (!result_opt.equal) {
                printf("    Optimized kernel failed: %s\n", result_opt.debug_info.c_str());
            }
            if (!result_simple.equal) {
                printf("    Simple kernel failed: %s\n", result_simple.debug_info.c_str());
            }
        }

        // Cleanup
        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C_ref));
        checkCudaErrors(cudaFreeHost(h_C_optimized));
        checkCudaErrors(cudaFreeHost(h_C_simple));
        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C_ref));
        checkCudaErrors(cudaFree(d_C_optimized));
        checkCudaErrors(cudaFree(d_C_simple));
    }

    printf("\n=============== FINAL RESULTS ===============\n");
    printf("PASSED: %d / %d\n", passed, num_tests);
    printf("FAILED: %d / %d\n", failed, num_tests);

    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    return (failed > 0) ? 1 : 0;
}