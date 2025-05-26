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

// Host function to launch the Strassen fused kernel (Version P)
void sgemm_strassen_1level_fused_vP(
    int M, int N, int K,
    const float host_alpha, // Standard SGEMM alpha (assumed 1.0f by Strassen part)
    const float* host_A, int lda,
    const float* host_B, int ldb,
    const float host_beta,  // Standard SGEMM beta
    float* host_C, int ldc,
    cudaStream_t stream = 0
) {
    if (host_alpha != 1.0f) {
        std::cerr << "sgemm_strassen_1level_fused_vP: host_alpha must be 1.0f for this version." << std::endl;
        // Potentially fall back to standard GEMM or error out
        return;
    }
    // Strassen's 1-level algorithm requires M, N, K to be divisible by 2.
    // Padding/peeling for odd dimensions would be needed for a general solution.
    if (M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
        std::cerr << "sgemm_strassen_1level_fused_vP: M, N, K must be divisible by 2 for this version." << std::endl;
        // Potentially fall back to standard GEMM or error out
        return;
    }

    const int m2 = M / 2;
    const int n2 = N / 2;
    const int k2 = K / 2;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // Allocate device memory
    checkCudaErrors(cudaMallocAsync((void**)&d_A, (size_t)M * lda * sizeof(float), stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_B, (size_t)K * ldb * sizeof(float), stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_C, (size_t)M * ldc * sizeof(float), stream));

    // Copy A and B from Host to Device
    checkCudaErrors(cudaMemcpy2DAsync(d_A, lda * sizeof(float), host_A, lda * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpy2DAsync(d_B, ldb * sizeof(float), host_B, ldb * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice, stream));

    // Handle C matrix (beta scaling)
    if (host_beta == 0.0f) {
        checkCudaErrors(cudaMemsetAsync(d_C, 0, (size_t)M * ldc * sizeof(float), stream));
    } else {
        // If beta is not 0, copy C from host and scale if beta is not 1.
        // Note: If C is very large, this host-side scaling can be a bottleneck.
        // A device-side scaling kernel might be preferred for performance.
        // Version P snippet implies host-side scaling for beta != 1 before copy, or copy then scale on device.
        // For simplicity here, if beta is not 1.0, we'll copy and then assume a device-side scale (though not implemented in this snippet directly for C).
        // The provided snippet's kernel itself handles accumulation (D += M, E += gamma*M).
        // So, if host_C is the source for beta part, it needs to be scaled then copied, or copied then scaled by a separate kernel.

        if (host_beta != 1.0f) {
            // Version P implies host-side scaling or a separate kernel.
            // Let's allocate temporary host memory for scaled C if beta != 1.0
            std::vector<float> h_C_scaled_temp( (size_t)M * ldc );
            for(int r=0; r<M; ++r) {
                for(int c=0; c<N; ++c) {
                    h_C_scaled_temp[r*ldc + c] = host_C[r*ldc + c] * host_beta;
                }
            }
            checkCudaErrors(cudaMemcpy2DAsync(d_C, ldc * sizeof(float), h_C_scaled_temp.data(), ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice, stream));
        } else {
             checkCudaErrors(cudaMemcpy2DAsync(d_C, ldc * sizeof(float), host_C, ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice, stream));
        }
    }
    
    // Submatrix device pointers
    const float* A00 = d_A;
    const float* A01 = d_A + k2;
    const float* A10 = d_A + (size_t)m2 * lda;
    const float* A11 = d_A + (size_t)m2 * lda + k2;

    const float* B00 = d_B;
    const float* B01 = d_B + n2;
    const float* B10 = d_B + (size_t)k2 * ldb;
    const float* B11 = d_B + (size_t)k2 * ldb + n2;

    float* C00 = d_C;
    float* C01 = d_C + n2;
    float* C10 = d_C + (size_t)m2 * ldc;
    float* C11 = d_C + (size_t)m2 * ldc + n2;

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
    // If we directly use the values from the kernel header:
    const int smem_a_eff_cols = 8 + 4; // K_TILE_SIZE + SMEM_A_LD_PADDING
    const int smem_a_tile_floats = 128 * smem_a_eff_cols; // M_TILE_SIZE * effective_cols
    const int smem_b_eff_cols = 128 + 0; // N_TILE_SIZE + SMEM_B_LD_PADDING
    const int smem_b_tile_floats = 8 * smem_b_eff_cols; // K_TILE_SIZE * effective_cols
    
    size_t smem_per_block_bytes = 2 * (smem_a_tile_floats + smem_b_tile_floats) * sizeof(float);
    // Additional shared memory is used for output staging in the kernel (512 floats * num_warps)
    // The kernel's output staging reuses part of the smem_storage.
    // The current kernel uses `smem_storage + sts_output_offset_comp` where `sts_output_offset_comp`
    // is `512 * warp_id + ...`. This implies the `smem_storage` must be large enough for
    // input tiles AND this output staging area.
    // The kernel's `sts_output_offset_comp` was `512 * warp_id + ...` which is fine if it points
    // within the 2*(A+B) tile area.
    // If `smem_storage + sts_output_offset_comp` means `smem_storage[sts_output_offset_comp]`
    // and `sts_output_offset_comp` is calculated based on `warp_id`, then it could map to anywhere.
    // The kernel I wrote previously for vP used:
    // `CVTA_TO_SHARED_PTX(sts_output_smem_addr_base, smem_storage + sts_output_offset_comp);`
    // where `sts_output_offset_comp = 512 * warp_id + 4 * 32 * lane_id_mapped_y_lds + 4 * lane_id_mapped_x_lds;`
    // This means `sts_output_offset_comp` can be up to `512*7 + small_val`, which is around 3584 floats.
    // Total for A+B tiles: `2 * (128*(8+4) + 8*128) = 2 * (1536 + 1024) = 2 * 2560 = 5120 floats`.
    // This requires the output staging to fit within this 5120 floats.
    // The original sgemm output logic uses `smem_ptr` which is the base of all shared memory.
    // The kernel code for vP uses `smem_storage + sts_output_offset_comp` for output.
    // This means the output staging area is part of the `smem_storage`.
    // The size `smem_per_block_bytes` calculated above is for the double buffered input tiles A and B.
    // The output staging reuses the *beginning* of this same shared memory:
    // `CVTA_TO_SHARED_PTX(sts_output_smem_addr_base, smem_storage + sts_output_offset_comp);`
    // where `sts_output_offset_comp` is `512 * warp_id + ...`. This area must be within the total allocated shared memory.
    // The largest `sts_output_offset_comp` is for `warp_id=7`, so `512*7 = 3584`.
    // This is within the `5120` floats allocated for A and B tiles. This is fine, means output overwrites input tiles after use.

    // Strassen's 7 products (M0-M6 from Version P mapping)
    // Kernel: strassen_fused_kernel_128x128x8_vP<DELTA_XY, EPSILON_VW, GAMMA0_DEST, GAMMA1_DEST>
    // (m, n, k, X, ldx, Y, ldy, V, ldv, W, ldw, D, ldd, E, lde)

    // M0 = (A00+A11)(B00+B11); C00+=M0, C11+=M0
    strassen_fused_kernel_128x128x8_vP<1, 1, 1, 1><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A00,lda,A11,lda, B00,ldb,B11,ldb, C00,ldc,C11,ldc);

    // M1 = (A10+A11)B00; C10+=M1, C11-=M1
    strassen_fused_kernel_128x128x8_vP<1, 0, 1, -1><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A10,lda,A11,lda, B00,ldb,B00,ldb, C10,ldc,C11,ldc);

    // M2 = A00(B01-B11); C01+=M2, C00-=M2
    // For C01+=M2: D=C01, M_comp=A00(B01-B11). DELTA=0, EPSILON=-1, GAMMA0=1, GAMMA1=0(dummy E)
    strassen_fused_kernel_128x128x8_vP<0, -1, 1, 0><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A00,lda,A00,lda, B01,ldb,B11,ldb, C01,ldc,C01,ldc);
    // For C00-=M2 (i.e. C00 += -M2): D=C00, M_comp=A00(B11-B01). DELTA=0, EPSILON=-1 (V=B11, W=B01), GAMMA0=1
    strassen_fused_kernel_128x128x8_vP<0, -1, 1, 0><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A00,lda,A00,lda, B11,ldb,B01,ldb, C00,ldc,C00,ldc);
        
    // M3 = A11(B10-B00); C11+=M3, C10+=M3 (Note: original standard Strassen C10 = P3+P4, C11 = P5+P1-P3-P7)
    // Version P mapping: C11+=M3, C10+=M3
    strassen_fused_kernel_128x128x8_vP<0, -1, 1, 1><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A11,lda,A11,lda, B10,ldb,B00,ldb, C11,ldc,C10,ldc); // D=C11, E=C10

    // M4 = (A00+A01)B11; C01+=M4, C11-=M4
    strassen_fused_kernel_128x128x8_vP<1, 0, 1, -1><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A00,lda,A01,lda, B11,ldb,B11,ldb, C01,ldc,C11,ldc);

    // M5 = (A10-A00)(B00+B01); C11+=M5
    strassen_fused_kernel_128x128x8_vP<-1, 1, 1, 0><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A10,lda,A00,lda, B00,ldb,B01,ldb, C11,ldc,C11,ldc);

    // M6 = (A01-A11)(B10+B11); C00+=M6
    strassen_fused_kernel_128x128x8_vP<-1, 1, 1, 0><<<grid, threads, smem_per_block_bytes, stream>>>(
        m2,n2,k2, A01,lda,A11,lda, B10,ldb,B11,ldb, C00,ldc,C00,ldc);

    // Copy result from Device to Host
    checkCudaErrors(cudaMemcpy2DAsync(host_C, ldc * sizeof(float), d_C, ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream)); // Ensure all ops including final copy are done

    // Free device memory
    checkCudaErrors(cudaFreeAsync(d_A, stream));
    checkCudaErrors(cudaFreeAsync(d_B, stream));
    checkCudaErrors(cudaFreeAsync(d_C, stream));
    // A final sync might be needed here if caller expects memory to be freed upon return
    // but typically not required if stream management is done by caller or higher level.
    // For this standalone function, it's good practice.
    checkCudaErrors(cudaStreamSynchronize(stream));
}

#endif // SGEMM_STRASSEN_FUSED_LAUNCHER_VP_CUH_
