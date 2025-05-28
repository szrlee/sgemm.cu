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

int
main() {
    printf("Testing Simple Strassen Kernel Only\n");

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Test with small size first
    int M = 64, N = 64, K = 64;
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t lda = K;
    size_t ldb = N;
    size_t ldc = N;

    // Allocate host memory
    float* h_A = alloc_mat_host(M * lda * sizeof(float));
    float* h_B = alloc_mat_host(K * ldb * sizeof(float));
    float* h_C_ref = alloc_mat_host(M * ldc * sizeof(float));
    float* h_C_simple = alloc_mat_host(M * ldc * sizeof(float));

    // Initialize with simple values for debugging
    init_const(h_A, M * K, 1.0f);
    init_const(h_B, K * N, 2.0f);
    init_const(h_C_ref, M * N, 0.0f);
    init_const(h_C_simple, M * N, 0.0f);

    // Allocate device memory
    float* d_A = alloc_mat_device(M * lda * sizeof(float));
    float* d_B = alloc_mat_device(K * ldb * sizeof(float));
    float* d_C_ref = alloc_mat_device(M * ldc * sizeof(float));
    float* d_C_simple = alloc_mat_device(M * ldc * sizeof(float));

    // Copy to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, M * lda * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, K * ldb * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C_ref, h_C_ref, M * ldc * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_C_simple, h_C_simple, M * ldc * sizeof(float), cudaMemcpyHostToDevice));

    // Run cuBLAS reference
    printf("Running cuBLAS reference...\n");
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

    // Run simple Strassen kernel
    printf("Running simple Strassen kernel...\n");
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
    checkCudaErrors(cudaMemcpy(h_C_ref, d_C_ref, M * ldc * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy(h_C_simple, d_C_simple, M * ldc * sizeof(float), cudaMemcpyDeviceToHost));

    // Print a few values for debugging
    printf("Expected result (cuBLAS): %f, %f, %f, %f\n",
           h_C_ref[0],
           h_C_ref[1],
           h_C_ref[N],
           h_C_ref[N + 1]);
    printf("Simple Strassen result: %f, %f, %f, %f\n",
           h_C_simple[0],
           h_C_simple[1],
           h_C_simple[N],
           h_C_simple[N + 1]);

    // Compare results
    const float tolerance = 1e-3f;
    cmp_result result = compare_mats(h_C_ref, h_C_simple, M * N, tolerance, false, true);

    if (result.equal) {
        printf("SUCCESS: Simple Strassen kernel works correctly!\n");
    } else {
        printf("FAILED: Simple Strassen kernel has errors: %s\n", result.debug_info.c_str());
    }

    // Cleanup
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C_ref));
    checkCudaErrors(cudaFreeHost(h_C_simple));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C_ref));
    checkCudaErrors(cudaFree(d_C_simple));

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return result.equal ? 0 : 1;
}