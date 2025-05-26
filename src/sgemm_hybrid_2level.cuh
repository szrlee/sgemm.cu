#ifndef SGEMM_HYBRID_2LEVEL_CUH_
#define SGEMM_HYBRID_2LEVEL_CUH_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common/helper_cuda.h"
#include "src/cublas_helpers.cuh"
#include "src/sgemm_strassen_fused.cuh" // For sgemm_strassen_1level_fused_vP

#include <vector>
#include <iostream>
#include <stdexcept> // For std::runtime_error

// Host function for Hybrid 2-Level Strassen SGEMM
void sgemm_strassen_hybrid_2level(
    int M, int N, int K,
    const float host_alpha,
    const float* host_A, int lda,
    const float* host_B, int ldb,
    const float host_beta,
    float* host_C, int ldc,
    cublasHandle_t& cublas_handle,
    cudaStream_t stream = 0 // Allow user-provided stream, default to 0
) {
    // --- Initial Setup ---
    // Dimension Checks
    if (M % 4 != 0 || N % 4 != 0 || K % 4 != 0) {
        std::cerr << "sgemm_strassen_hybrid_2level: M, N, K must be divisible by 4 for this version." << std::endl;
        // Consider throwing an error or returning a specific status
        throw std::runtime_error("M, N, K must be divisible by 4 for sgemm_strassen_hybrid_2level.");
    }
    if (host_alpha == 0.0f) { // If alpha is 0, C = beta*C. Handle separately for simplicity or let it pass.
        // For now, this implementation assumes alpha is not zero.
        // If alpha is zero, the M_i terms become zero.
        // The final C combination C_ij = beta*C_ij_initial.
        // This is already handled by the beta scaling logic if M_i terms are zero.
        // However, Strassen calls with alpha=0 might not be optimized.
        // For simplicity, we'll proceed assuming alpha != 0. The scaling by host_alpha later will handle it.
    }


    const int m2 = M / 2;
    const int n2 = N / 2;
    const int k2 = K / 2;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C_temp_for_beta = nullptr;

    // Device memory allocation
    checkCudaErrors(cudaMallocAsync((void**)&d_A, (size_t)M * lda * sizeof(float), stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_B, (size_t)K * ldb * sizeof(float), stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_C, (size_t)M * ldc * sizeof(float), stream));

    // Copy host_A to d_A, host_B to d_B
    checkCudaErrors(cudaMemcpy2DAsync(d_A, lda * sizeof(float), host_A, lda * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpy2DAsync(d_B, ldb * sizeof(float), host_B, ldb * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice, stream));

    // Initialize d_C based on host_beta
    if (host_beta == 0.0f) {
        checkCudaErrors(cudaMemsetAsync(d_C, 0, (size_t)M * ldc * sizeof(float), stream));
    } else if (host_beta == 1.0f) {
        checkCudaErrors(cudaMemcpy2DAsync(d_C, ldc * sizeof(float), host_C, ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice, stream));
    } else {
        // Need temporary device memory to hold host_C before scaling
        checkCudaErrors(cudaMallocAsync((void**)&d_C_temp_for_beta, (size_t)M * ldc * sizeof(float), stream));
        checkCudaErrors(cudaMemcpy2DAsync(d_C_temp_for_beta, ldc * sizeof(float), host_C, ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice, stream));
        // d_C = host_beta * d_C_temp_for_beta
        matrix_out_of_place_set_and_scale_gpu(cublas_handle, M, N, host_beta, d_C_temp_for_beta, ldc, d_C, ldc);
        checkCudaErrors(cudaFreeAsync(d_C_temp_for_beta, stream)); // Free temporary once d_C is scaled
    }
    // At this point, d_C contains beta * C_initial (or is zero if beta was 0)
    // All subsequent operations will add alpha * (component products) to d_C

    // --- Submatrix Device Pointers ---
    // For A (M x K) -> submatrices m2 x k2
    const float* A00_dev = d_A;
    const float* A01_dev = d_A + k2; // k2 columns over
    const float* A10_dev = d_A + (size_t)m2 * lda;
    const float* A11_dev = d_A + (size_t)m2 * lda + k2;

    // For B (K x N) -> submatrices k2 x n2
    const float* B00_dev = d_B;
    const float* B01_dev = d_B + n2; // n2 columns over
    const float* B10_dev = d_B + (size_t)k2 * ldb;
    const float* B11_dev = d_B + (size_t)k2 * ldb + n2;

    // For C (M x N) -> submatrices m2 x n2
    float* C00_dev = d_C;
    float* C01_dev = d_C + n2; // n2 columns over
    float* C10_dev = d_C + (size_t)m2 * ldc;
    float* C11_dev = d_C + (size_t)m2 * ldc + n2;

    // --- Workspace Allocation ---
    // S matrices: m2 x k2 (5 of them)
    // T matrices: k2 x n2 (5 of them)
    // M matrices: m2 x n2 (7 of them)
    size_t s_matrix_size_floats = (size_t)m2 * k2;
    size_t t_matrix_size_floats = (size_t)k2 * n2;
    size_t m_matrix_size_floats = (size_t)m2 * n2;

    size_t total_workspace_floats = (5 * s_matrix_size_floats) + (5 * t_matrix_size_floats) + (7 * m_matrix_size_floats);
    float* workspace_dev = nullptr;
    checkCudaErrors(cudaMallocAsync((void**)&workspace_dev, total_workspace_floats * sizeof(float), stream));

    // Assign pointers for S, T, M matrices within workspace_dev
    float* S0_dev = workspace_dev;
    float* S1_dev = S0_dev + s_matrix_size_floats;
    float* S4_dev = S1_dev + s_matrix_size_floats; // S2, S3 are direct pointers
    float* S5_dev = S4_dev + s_matrix_size_floats;
    float* S6_dev = S5_dev + s_matrix_size_floats;
    float* current_S_ptr_end = S6_dev + s_matrix_size_floats;

    float* T0_dev = current_S_ptr_end;
    float* T2_dev = T0_dev + t_matrix_size_floats; // T1, T4 are direct pointers
    float* T3_dev = T2_dev + t_matrix_size_floats;
    float* T5_dev = T3_dev + t_matrix_size_floats;
    float* T6_dev = T5_dev + t_matrix_size_floats;
    float* current_T_ptr_end = T6_dev + t_matrix_size_floats;

    float* M0_dev = current_T_ptr_end;
    float* M1_dev = M0_dev + m_matrix_size_floats;
    float* M2_dev = M1_dev + m_matrix_size_floats;
    float* M3_dev = M2_dev + m_matrix_size_floats;
    float* M4_dev = M3_dev + m_matrix_size_floats;
    float* M5_dev = M4_dev + m_matrix_size_floats;
    float* M6_dev = M5_dev + m_matrix_size_floats;

    // --- Compute S_i and T_i terms ---
    // S0 = A00 + A11
    matrix_add_gpu(cublas_handle, m2, k2, A00_dev, lda, A11_dev, lda, S0_dev, k2);
    // S1 = A10 + A11
    matrix_add_gpu(cublas_handle, m2, k2, A10_dev, lda, A11_dev, lda, S1_dev, k2);
    // S2 = A00 (direct pointer, no computation)
    const float* S2_ptr = A00_dev;
    // S3 = A11 (direct pointer, no computation)
    const float* S3_ptr = A11_dev;
    // S4 = A00 + A01
    matrix_add_gpu(cublas_handle, m2, k2, A00_dev, lda, A01_dev, lda, S4_dev, k2);
    // S5 = A10 - A00
    matrix_subtract_gpu(cublas_handle, m2, k2, A10_dev, lda, A00_dev, lda, S5_dev, k2);
    // S6 = A01 - A11
    matrix_subtract_gpu(cublas_handle, m2, k2, A01_dev, lda, A11_dev, lda, S6_dev, k2);

    // T0 = B00 + B11
    matrix_add_gpu(cublas_handle, k2, n2, B00_dev, ldb, B11_dev, ldb, T0_dev, n2);
    // T1 = B00 (direct pointer)
    const float* T1_ptr = B00_dev;
    // T2 = B01 - B11
    matrix_subtract_gpu(cublas_handle, k2, n2, B01_dev, ldb, B11_dev, ldb, T2_dev, n2);
    // T3 = B10 - B00
    matrix_subtract_gpu(cublas_handle, k2, n2, B10_dev, ldb, B00_dev, ldb, T3_dev, n2);
    // T4 = B11 (direct pointer)
    const float* T4_ptr = B11_dev;
    // T5 = B00 + B01
    matrix_add_gpu(cublas_handle, k2, n2, B00_dev, ldb, B01_dev, ldb, T5_dev, n2);
    // T6 = B10 + B11
    matrix_add_gpu(cublas_handle, k2, n2, B10_dev, ldb, B11_dev, ldb, T6_dev, n2);

    checkCudaErrors(cudaStreamSynchronize(stream)); // Ensure S and T computations are complete

    // --- Compute Inner 7 Products (M_i) using 1-Level Fused Strassen ---
    const float alpha_inner = 1.0f;
    const float beta_inner = 0.0f; // M_i = S_a * T_b (no accumulation)

    // M0 = S0 * T0 = (A00+A11)(B00+B11)
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S0_dev, k2, T0_dev, n2, beta_inner, M0_dev, n2, stream);
    // M1 = S1 * T1 = (A10+A11)B00
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S1_dev, k2, T1_ptr, ldb, beta_inner, M1_dev, n2, stream);
    // M2 = S2 * T2 = A00(B01-B11)
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S2_ptr, lda, T2_dev, n2, beta_inner, M2_dev, n2, stream);
    // M3 = S3 * T3 = A11(B10-B00)
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S3_ptr, lda, T3_dev, n2, beta_inner, M3_dev, n2, stream);
    // M4 = S4 * T4 = (A00+A01)B11
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S4_dev, k2, T4_ptr, ldb, beta_inner, M4_dev, n2, stream);
    // M5 = S5 * T5 = (A10-A00)(B00+B01)
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S5_dev, k2, T5_dev, n2, beta_inner, M5_dev, n2, stream);
    // M6 = S6 * T6 = (A01-A11)(B10+B11)
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S6_dev, k2, T6_dev, n2, beta_inner, M6_dev, n2, stream);
    
    // sgemm_strassen_1level_fused_vP calls cudaStreamSynchronize internally for each call.
    // If it didn't, a single cudaStreamSynchronize(stream) would be needed here.

    // --- Combine M_i to form final C submatrices ---
    // C00 = M0+M3-M4+M6 (all scaled by host_alpha, accumulated onto existing C00_dev)
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M0_dev, n2, C00_dev, ldc);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M3_dev, n2, C00_dev, ldc);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, -host_alpha, M4_dev, n2, C00_dev, ldc); // -M4
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M6_dev, n2, C00_dev, ldc);

    // C01 = M2+M4
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M2_dev, n2, C01_dev, ldc);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M4_dev, n2, C01_dev, ldc);

    // C10 = M1+M3
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M1_dev, n2, C10_dev, ldc);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M3_dev, n2, C10_dev, ldc);

    // C11 = M0-M1+M2+M5
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M0_dev, n2, C11_dev, ldc);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, -host_alpha, M1_dev, n2, C11_dev, ldc); // -M1
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M2_dev, n2, C11_dev, ldc);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M5_dev, n2, C11_dev, ldc);

    checkCudaErrors(cudaStreamSynchronize(stream)); // Ensure C combinations are complete

    // --- Copy Result to Host ---
    checkCudaErrors(cudaMemcpy2DAsync(host_C, ldc * sizeof(float), d_C, ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream)); // Ensure final D2H copy is complete

    // --- Cleanup ---
    checkCudaErrors(cudaFreeAsync(workspace_dev, stream));
    checkCudaErrors(cudaFreeAsync(d_A, stream));
    checkCudaErrors(cudaFreeAsync(d_B, stream));
    checkCudaErrors(cudaFreeAsync(d_C, stream));
    // A final sync to ensure frees complete before function returns if stream is not default
    if (stream != 0) {
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
}

#endif // SGEMM_HYBRID_2LEVEL_CUH_
