#ifndef SGEMM_STRASSEN_FUSED_LAUNCHER_ADAPTED_CUH_
#define SGEMM_STRASSEN_FUSED_LAUNCHER_ADAPTED_CUH_

#include "kernels/strassen_fused_128x128x8.cuh" // Include the new kernel
#include <cstdio>                               // For fprintf
#include <cublas_v2.h> // For cublasHandle_t, though not used by this kernel
#include <cuda_runtime.h>
#include <stdexcept>   // For std::runtime_error if desired for error handling
#include <stdio.h>     // Added for printf/fprintf, just in case for linter
#include <vector>      // For std::vector if used for any host logic

// Helper for grid dimensions (if not in a common place)
static inline int
div_ceil_strassen_adapted(int a, int b) {
    return (a + b - 1) / b;
}

// Adapted to work like the old sgemm_strassen_1level_fused_vP (device pointers)
void
sgemm_strassen_1level_fused_vP( // Renamed
    int M,
    int N,
    int K,
    float
        alpha_for_product, // Alpha for the X*V product (typically 1.0 for M_i) - passed as host_alpha to kernel
    const float* A_dev,
    int lda,               // Device pointer
    const float* B_dev,
    int ldb,               // Device pointer
    float beta_for_C, // Beta for C_dev_inout = alpha_for_product*M_i + beta_for_C*C_initial
                      // if 0, C_dev_inout is zeroed. if 1, C_dev_inout is used as is.
                      // if other, C_dev_inout should be pre-scaled by caller.
    float* C_dev_inout,
    int ldc,          // Device pointer for C (output)
    cublasHandle_t cublas_handle, // Unused, for API compatibility
    cudaStream_t stream = 0,
    bool is_first_kernel_launch_for_debug = false) {
    if (M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
        fprintf(stderr, "sgemm_strassen_1level_fused_vP: M, N, K must be divisible by 2.\\n");
        return;
    }

    // Minimal check, kernel guards should handle finer details
    if (M < STRASSEN_TILE_M || N < STRASSEN_TILE_N) {
        fprintf(stderr,
                "Warning (sgemm_strassen_1level_fused_vP): Matrix dimensions smaller than one tile "
                "size.\\n");
    }

    int m2 = M / 2;
    int n2 = N / 2;
    int k2 = K / 2;

    cudaError_t err;

    // Handle C_dev_inout based on beta_for_C
    // The strassen_fused_128x128x8_kernel performs C_target += gamma * M_i.
    // So C_dev_inout must be correctly initialized *before* kernel calls if beta_for_C is involved.
    if (beta_for_C == 0.0f) {
        // If beta is 0, the final C matrix should be alpha_for_product * sum(Strassen terms).
        // The individual M_i terms are added to C_dev_inout. So C_dev_inout must start at 0.
        err = cudaMemsetAsync(C_dev_inout, 0, (size_t)M * ldc * sizeof(float), stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemsetAsync C_dev_inout failed: %s\\n", cudaGetErrorString(err));
            return;
        }
    } else if (beta_for_C != 1.0f) {
        // If beta is not 0 and not 1, the caller (e.g., hybrid Strassen or test.cu)
        // is responsible for pre-scaling the initial C matrix by beta_for_C
        // and placing it in C_dev_inout. This function will then accumulate M_i terms.
        // No action needed here for C_dev_inout itself, assuming caller handled it.
        // fprintf(stderr, "Info (sgemm_strassen_1level_fused_vP): beta_for_C is %f. C_dev_inout assumed pre-scaled.\\n", beta_for_C);
    }
    // If beta_for_C == 1.0f, C_dev_inout is used as is for accumulation.

    // Submatrix device pointers (derived from input device pointers)
    const float* A00 = A_dev;
    const float* A01 = A_dev + k2;
    const float* A10 = A_dev + (size_t)m2 * lda;
    const float* A11 = A_dev + (size_t)m2 * lda + k2;

    const float* B00 = B_dev;
    const float* B01 = B_dev + n2;
    const float* B10 = B_dev + (size_t)k2 * ldb;
    const float* B11 = B_dev + (size_t)k2 * ldb + n2;

    float* C00 = C_dev_inout;
    float* C01 = C_dev_inout + n2;
    float* C10 = C_dev_inout + (size_t)m2 * ldc;
    float* C11 = C_dev_inout + (size_t)m2 * ldc + n2;

    // printf("[STRASSEN LAUNCHER DEBUG] M=%d, N=%d, K=%d -> m2=%d, n2=%d, k2=%d\n",
    //        M,
    //        N,
    //        K,
    //        m2,
    //        n2,
    //        k2);
    // printf("[STRASSEN LAUNCHER DEBUG] A=%p, lda=%d, B=%p, ldb=%d, C=%p, ldc=%d\n",
    //        (void*)A_dev,
    //        lda,
    //        (void*)B_dev,
    //        ldb,
    //        (void*)C_dev_inout,
    //        ldc);
    // printf("[STRASSEN LAUNCHER DEBUG] Sub A: A00=%p, A01=%p, A10=%p, A11=%p\n",
    //        (void*)A00,
    //        (void*)A01,
    //        (void*)A10,
    //        (void*)A11);
    // printf("[STRASSEN LAUNCHER DEBUG] Sub B: B00=%p, B01=%p, B10=%p, B11=%p\n",
    //        (void*)B00,
    //        (void*)B01,
    //        (void*)B10,
    //        (void*)B11);
    // printf("[STRASSEN LAUNCHER DEBUG] Sub C: C00=%p, C01=%p, C10=%p, C11=%p\n",
    //        (void*)C00,
    //        (void*)C01,
    //        (void*)C10,
    //        (void*)C11);

    dim3 threads(256);
    dim3 blocks(div_ceil_strassen_adapted(n2, STRASSEN_TILE_N),
                div_ceil_strassen_adapted(m2, STRASSEN_TILE_M));

    size_t smem_per_block = 2
                            * (SMEM_A_SINGLE_BUFFER_SIZE_FLOATS + SMEM_B_SINGLE_BUFFER_SIZE_FLOATS)
                            * sizeof(float);

    // Note: alpha_for_product is passed as 'host_alpha' to the kernel.
    // The kernel's effective_gamma includes this host_alpha.
    // Example M0=(A00+A11)(B00+B11); C00+=M0; C11+=M0;
    // Kernel computes M = (X+dY)(V+eW). Then D += eff_g0*M, E += eff_g1*M
    // Here, M = M0. D=C00, E=C11.
    // eff_g0 = alpha_for_product * (+1 for C00). eff_g1 = alpha_for_product * (+1 for C11).
    // M0=(A00+A11)(B00+B11); C00+=M0; C11+=M0;
    // printf("[STRASSEN LAUNCHER] M0 kernel launch\\n"); // DISABLED FOR PERFORMANCE
    strassen_fused_128x128x8_kernel<1, 1, 1, 1><<<blocks, threads, smem_per_block, stream>>>(
        m2,
        n2,
        k2,
        alpha_for_product,
        A00,
        lda,
        A11,
        lda, // X_gm, ldx, Y_gm, ldy
        B00,
        ldb,
        B11,
        ldb, // V_gm, ldv, W_gm, ldw
        C00,
        ldc,
        C11,
        ldc,
        is_first_kernel_launch_for_debug); // enable_debug_prints for M0
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after M0 kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }
    // cudaDeviceSynchronize(); // Sync after M0 // DISABLED FOR PERFORMANCE

    // M1=(A10+A11)B00; C10+=M1; C11-=M1;
    // printf("[STRASSEN LAUNCHER] M1 kernel launch\\n"); // DISABLED FOR PERFORMANCE
    strassen_fused_128x128x8_kernel<1, 0, 1, -1><<<blocks, threads, smem_per_block, stream>>>(
        m2,
        n2,
        k2,
        alpha_for_product,
        A10,
        lda,
        A11,
        lda, // X_base=A10, X_off=A11, delta_X=1 (A10+A11)
        B00,
        ldb,
        B00,
        ldb, // V_base=B00, V_off=B00, delta_V=0 (B00)
        C10,
        ldc,
        C11,
        ldc,
        false); // Disable debug for M1
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after M1 kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }
    // cudaDeviceSynchronize(); // Sync after M1 // DISABLED FOR PERFORMANCE

    // M2=A00(B01-B11); C01+=M2; C11+=M2;
    // printf("[STRASSEN LAUNCHER] M2 kernel launch\\n"); // DISABLED FOR PERFORMANCE
    strassen_fused_128x128x8_kernel<0, -1, 1, 1><<<blocks, threads, smem_per_block, stream>>>(
        m2,
        n2,
        k2,
        alpha_for_product,
        A00,
        lda,
        A00,
        lda, // X_base=A00, X_off=A00, delta_X=0 (A00)
        B01,
        ldb,
        B11,
        ldb, // V_base=B01, V_off=B11, delta_V=-1 (B01-B11)
        C01,
        ldc,
        C11,
        ldc,
        is_first_kernel_launch_for_debug); // Enable debug for M2 too
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after M2 kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }
    // cudaDeviceSynchronize(); // Sync after M2 // DISABLED FOR PERFORMANCE

    // M3=A11(B10-B00); C00+=M3; C10+=M3;
    // printf("[STRASSEN LAUNCHER] M3 kernel launch\\n"); // DISABLED FOR PERFORMANCE
    strassen_fused_128x128x8_kernel<0, -1, 1, 1><<<blocks, threads, smem_per_block, stream>>>(
        m2,
        n2,
        k2,
        alpha_for_product,
        A11,
        lda,
        A11,
        lda, // X_base=A11, X_off=A11, delta_X=0 (A11)
        B10,
        ldb,
        B00,
        ldb, // V_base=B10, V_off=B00, delta_V=-1 (B10-B00)
        C00,
        ldc,
        C10,
        ldc,
        is_first_kernel_launch_for_debug); // Enable debug for M3 too
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after M3 kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }
    // cudaDeviceSynchronize(); // Sync after M3 // DISABLED FOR PERFORMANCE

    // M4=(A00+A01)B11; C01+=M4; C00-=M4;
    // printf("[STRASSEN LAUNCHER] M4 kernel launch\\n"); // DISABLED FOR PERFORMANCE
    strassen_fused_128x128x8_kernel<1, 0, -1, 1><<<blocks, threads, smem_per_block, stream>>>(
        m2,
        n2,
        k2,
        alpha_for_product,
        A00,
        lda,
        A01,
        lda, // X_base=A00, X_off=A01, delta_X=1 (A00+A01)
        B11,
        ldb,
        B11,
        ldb, // V_base=B11, V_off=B11, delta_V=0 (B11)
        C00,
        ldc,
        C01,
        ldc,
        is_first_kernel_launch_for_debug); // Enable debug for M4 too
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after M4 kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }
    // cudaDeviceSynchronize(); // Sync after M4 // DISABLED FOR PERFORMANCE

    // M5=(A10-A00)(B00+B01); C11+=M5;
    // printf("[STRASSEN LAUNCHER] M5 kernel launch\\n"); // DISABLED FOR PERFORMANCE
    strassen_fused_128x128x8_kernel<-1, 1, 1, 0><<<blocks, threads, smem_per_block, stream>>>(
        m2,
        n2,
        k2,
        alpha_for_product,
        A10,
        lda,
        A00,
        lda, // X_base=A10, X_off=A00, delta_X=-1 (A10-A00)
        B00,
        ldb,
        B01,
        ldb, // V_base=B00, V_off=B01, delta_V=1 (B00+B01)
        C11,
        ldc,
        C11,
        ldc,
        is_first_kernel_launch_for_debug); // Enable debug for M5 too
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after M5 kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }
    // cudaDeviceSynchronize(); // Sync after M5 // DISABLED FOR PERFORMANCE

    // M6=(A01-A11)(B10+B11); C00+=M6;
    // printf("[STRASSEN LAUNCHER] M6 kernel launch\\n"); // DISABLED FOR PERFORMANCE
    strassen_fused_128x128x8_kernel<-1, 1, 1, 0><<<blocks, threads, smem_per_block, stream>>>(
        m2,
        n2,
        k2,
        alpha_for_product,
        A01,
        lda,
        A11,
        lda, // X_base=A01, X_off=A11, delta_X=-1 (A01-A11)
        B10,
        ldb,
        B11,
        ldb, // V_base=B10, V_off=B11, delta_V=1 (B10+B11)
        C00,
        ldc,
        C00,
        ldc,
        is_first_kernel_launch_for_debug); // Enable debug for M6 too
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after M6 kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }
    // cudaDeviceSynchronize(); // Sync after M6 // DISABLED FOR PERFORMANCE

    // Synchronization is expected to be handled by the caller if necessary,
    // especially if this function is part of a larger sequence of operations on the same stream.
    // For example, sgemm_strassen_hybrid_2level_vC will synchronize the stream after all its M_i calculations.
}

#endif // SGEMM_STRASSEN_FUSED_LAUNCHER_ADAPTED_CUH_
