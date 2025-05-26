#ifndef SGEMM_STRASSEN_FUSED_LAUNCHER_VP_CUH_
#define SGEMM_STRASSEN_FUSED_LAUNCHER_VP_CUH_

#include "common/helper_cuda.h"
#include "kernels/strassen_fused_128x128x8.cuh" // Should point to the new vP kernel
#include <iostream>
#include <vector>

// Helper function for calculating grid dimensions (can be kept if not part of helper_cuda.h)
static inline int div_ceil_strassen_vP(int a, int b) {
    return (a + b - 1) / b;
}

// Refactored Host function for 1-Level Fused Strassen (operates on device pointers)
void sgemm_strassen_1level_fused_vP(
    int M, int N, int K,
    float alpha_sub, // Alpha for this specific A*B operation (effectively 1.0 for M_i calculation)
    const float* A_dev, int lda_dev, // Device pointer for A and its LD
    const float* B_dev, int ldb_dev, // Device pointer for B and its LD
    float beta_sub,  // Beta for this specific C = alpha*A*B + beta*C operation
    float* C_dev, int ldc_dev,       // Device pointer for C and its LD
    cublasHandle_t cublas_handle_for_kernels, // Unused by Version P kernels, but for API consistency
    cudaStream_t stream = 0
) {
    // This function computes C_dev = A_dev * B_dev (when beta_sub=0)
    // or C_dev = C_dev_initial + A_dev * B_dev (when beta_sub=1).
    // alpha_sub is assumed to be 1.0 for the A_dev*B_dev part, as scaling is handled by
    // the GAMMA template parameters in the kernel or by the caller of this function.

    if (M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
        std::cerr << "sgemm_strassen_1level_fused_vP (refactored): M, N, K must be divisible by 2." << std::endl;
        // Or throw std::runtime_error or return cudaError_t
        return;
    }
    if (alpha_sub != 1.0f) {
         std::cerr << "sgemm_strassen_1level_fused_vP (refactored): alpha_sub must be 1.0 for this version, "
                   << "as scaling is handled by kernel's GAMMA or caller." << std::endl;
        // This function implicitly computes alpha_sub=1.0 * (A_dev * B_dev) part
    }


    const int m2 = M / 2;
    const int n2 = N / 2;
    const int k2 = K / 2;

    // Handle C_dev based on beta_sub
    if (beta_sub == 0.0f) {
        checkCudaErrors(cudaMemsetAsync(C_dev, 0, (size_t)M * ldc_dev * sizeof(float), stream));
    } else if (beta_sub != 1.0f) {
        // This simplified version expects C_dev to be pre-scaled by beta_sub by the caller,
        // or beta_sub is 0.0 or 1.0.
        // For the 2-level hybrid, the C_dev (which is an M_i workspace) is always zeroed (beta_sub=0).
        // If this were a general purpose func, C_dev = beta_sub * C_dev would be done here.
        std::cerr << "sgemm_strassen_1level_fused_vP (refactored): beta_sub other than 0.0 or 1.0 means C_dev should be pre-scaled by caller." << std::endl;
    }
    // If beta_sub == 1.0f, C_dev is used as is (accumulates).
    
    // Submatrix device pointers derived from input device pointers
    const float* A00 = A_dev;
    const float* A01 = A_dev + k2;
    const float* A10 = A_dev + (size_t)m2 * lda_dev;
    const float* A11 = A_dev + (size_t)m2 * lda_dev + k2;

    const float* B00 = B_dev;
    const float* B01 = B_dev + n2;
    const float* B10 = B_dev + (size_t)k2 * ldb_dev;
    const float* B11 = B_dev + (size_t)k2 * ldb_dev + n2;

    float* C00 = C_dev;
    float* C01 = C_dev + n2;
    float* C10 = C_dev + (size_t)m2 * ldc_dev;
    float* C11 = C_dev + (size_t)m2 * ldc_dev + n2;

    dim3 threads(256);
    dim3 grid(div_ceil_strassen_vP(n2, 128), div_ceil_strassen_vP(m2, 128)); // N-dim first for grid.x

    // Calculate dynamic shared memory size based on constants in the kernel file
    // These constants are from src/kernels/strassen_fused_128x128x8.cuh
    // constexpr int M_TILE_SIZE = 128; // Not directly used for smem calc here but defines tile
    // constexpr int K_TILE_SIZE = 8;
    // constexpr int SMEM_A_ROWS = M_TILE_SIZE;
    // constexpr int SMEM_A_COLS = K_TILE_SIZE;
    // constexpr int SMEM_A_LD_PADDING = 4;
    // constexpr int SMEM_A_ELEMENTS_PER_TILE = SMEM_A_ROWS * (SMEM_A_COLS + SMEM_A_LD_PADDING);
    // constexpr int SMEM_B_ROWS = K_TILE_SIZE;
    // constexpr int SMEM_B_COLS = N_TILE_SIZE; // N_TILE_SIZE = 128
    // constexpr int SMEM_B_LD_PADDING = 0;
    // constexpr int SMEM_B_ELEMENTS_PER_TILE = SMEM_B_ROWS * (SMEM_B_COLS + SMEM_B_LD_PADDING);
    // The kernel uses these symbolic constants which must match what's in its own header.
    dim3 grid(div_ceil_strassen_vP(n2, 128), div_ceil_strassen_vP(m2, 128));

    // Shared memory calculation (must match kernel's expectations for its constants)
    const int smem_a_eff_cols_kernel = 8 + 4; // K_TILE_SIZE + SMEM_A_LD_PADDING from kernel header
    const int smem_a_tile_floats_kernel = 128 * smem_a_eff_cols_kernel;
    const int smem_b_eff_cols_kernel = 128 + 0; // N_TILE_SIZE + SMEM_B_LD_PADDING from kernel header
    const int smem_b_tile_floats_kernel = 8 * smem_b_eff_cols_kernel;
    size_t smem_per_block_bytes = 2 * (smem_a_tile_floats_kernel + smem_b_tile_floats_kernel) * sizeof(float);

    // The 7 Strassen kernel calls.
    // These compute M_i = S_i * T_i and accumulate into C_dev submatrices (C00, C01, C10, C11).
    // The GAMMA template parameters in the kernel handle the +/- accumulation.
    // alpha_sub is effectively 1.0 for these operations, as the result of this function
    // (which is an M_i block for the 2-level hybrid) will be scaled by the outer alpha later.

    // M0 = (A00+A11)(B00+B11); C00+=M0, C11+=M0
    strassen_fused_kernel_128x128x8_vP<1, 1, 1, 1><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A00,lda_dev,A11,lda_dev, B00,ldb_dev,B11,ldb_dev, C00,ldc_dev,C11,ldc_dev);

    // M1 = (A10+A11)B00; C10+=M1, C11-=M1
    strassen_fused_kernel_128x128x8_vP<1, 0, 1, -1><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A10,lda_dev,A11,lda_dev, B00,ldb_dev,B00,ldb_dev, C10,ldc_dev,C11,ldc_dev);

    // M2 = A00(B01-B11); C01+=M2, C00-=M2
    strassen_fused_kernel_128x128x8_vP<0, -1, 1, 0><<<grid, threads, smem_per_block_bytes, stream>>>( // C01 += A00(B01-B11)
        m2,n2,k2, A00,lda_dev,A00,lda_dev, B01,ldb_dev,B11,ldb_dev, C01,ldc_dev,C01,ldc_dev);
    strassen_fused_kernel_128x128x8_vP<0, -1, 1, 0><<<grid, threads, smem_per_block_bytes, stream>>>( // C00 += A00(B11-B01) means C00 -= A00(B01-B11)
        m2,n2,k2, A00,lda_dev,A00,lda_dev, B11,ldb_dev,B01,ldb_dev, C00,ldc_dev,C00,ldc_dev);
        
    // M3 = A11(B10-B00); C11+=M3, C10+=M3
    strassen_fused_kernel_128x128x8_vP<0, -1, 1, 1><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A11,lda_dev,A11,lda_dev, B10,ldb_dev,B00,ldb_dev, C11,ldc_dev,C10,ldc_dev);

    // M4 = (A00+A01)B11; C01+=M4, C11-=M4
    strassen_fused_kernel_128x128x8_vP<1, 0, 1, -1><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A00,lda_dev,A01,lda_dev, B11,ldb_dev,B11,ldb_dev, C01,ldc_dev,C11,ldc_dev);

    // M5 = (A10-A00)(B00+B01); C11+=M5
    strassen_fused_kernel_128x128x8_vP<-1, 1, 1, 0><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A10,lda_dev,A00,lda_dev, B00,ldb_dev,B01,ldb_dev, C11,ldc_dev,C11,ldc_dev);

    // M6 = (A01-A11)(B10+B11); C00+=M6
    strassen_fused_kernel_128x128x8_vP<-1, 1, 1, 0><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A01,lda_dev,A11,lda_dev, B10,ldb_dev,B11,ldb_dev, C00,ldc_dev,C00,ldc_dev);

    // Synchronization is handled by the caller (sgemm_strassen_hybrid_2level)
}

#endif // SGEMM_STRASSEN_FUSED_LAUNCHER_VP_CUH_
