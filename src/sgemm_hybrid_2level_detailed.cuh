#ifndef SGEMM_HYBRID_2LEVEL_DETAILED_CUH_
#define SGEMM_HYBRID_2LEVEL_DETAILED_CUH_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common/helper_cuda.h" // For CUDA_CHECK, CUBLAS_CHECK
#include "src/cublas_helpers.cuh"     // For stream-aware matrix_add_gpu etc.
#include "src/sgemm_strassen_fused.cuh" // For sgemm_strassen_1level_fused_vP

#include <vector>
#include <cstdio>    // For fprintf, stderr, snprintf
#include <stdexcept> // For std::runtime_error

// Host function for Hybrid 2-Level Strassen SGEMM (Version D detailed)
void sgemm_strassen_hybrid_2level_detailed(
    int M, int N, int K,
    const float host_alpha_overall,
    const float* A_host_full, int lda_host_full,
    const float* B_host_full, int ldb_host_full,
    const float host_beta_overall,
    float* C_host_result, int ldc_host_full,
    cublasHandle_t& cublas_handle 
) {
    // --- Initial Setup ---
    if (M % 4 != 0 || N % 4 != 0 || K % 4 != 0) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "sgemm_strassen_hybrid_2level_detailed: M, N, K must be divisible by 4. Got M=%d, N=%d, K=%d", M, N, K);
        fprintf(stderr, "%s\n", err_msg);
        throw std::runtime_error(err_msg);
    }
    if (M < 4 || N < 4 || K < 4) { // Practical minimum for 2-level
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "sgemm_strassen_hybrid_2level_detailed: M, N, K must be >= 4. Got M=%d, N=%d, K=%d", M, N, K);
        fprintf(stderr, "%s\n", err_msg);
        throw std::runtime_error(err_msg);
    }

    const int m2 = M / 2;
    const int n2 = N / 2;
    const int k2 = K / 2;

    // Leading dimensions for submatrices in workspace
    const int ld_S_m2k2 = k2;
    const int ld_T_k2n2 = n2;
    const int ld_M_m2n2 = n2;

    // CUDA Streams
    const int num_streams_for_M_products = 7;
    cudaStream_t streams[num_streams_for_M_products];
    for (int i = 0; i < num_streams_for_M_products; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
    cudaStream_t main_stream = streams[0]; // Use streams[0] for S/T, C setup, D2H, and some M products.

    // Associate cuBLAS with the main stream for initial S/T ops and C accumulations
    CUBLAS_CHECK(cublasSetStream(cublas_handle, main_stream));

    // Device memory
    float *d_A_full = nullptr, *d_B_full = nullptr, *d_C_full = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&d_A_full, (size_t)M * lda_host_full * sizeof(float), main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_B_full, (size_t)K * ldb_host_full * sizeof(float), main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_C_full, (size_t)M * ldc_host_full * sizeof(float), main_stream));

    // Copy A_host_full to d_A_full, B_host_full to d_B_full
    CUDA_CHECK(cudaMemcpy2DAsync(d_A_full, lda_host_full * sizeof(float), A_host_full, lda_host_full * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice, main_stream));
    CUDA_CHECK(cudaMemcpy2DAsync(d_B_full, ldb_host_full * sizeof(float), B_host_full, ldb_host_full * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice, main_stream));

    // Initialize d_C_full based on host_beta_overall
    if (host_beta_overall == 0.0f) {
        CUDA_CHECK(cudaMemsetAsync(d_C_full, 0, (size_t)M * ldc_host_full * sizeof(float), main_stream));
    } else {
        // For beta != 0, scale C_host_result by beta on host, then copy to d_C_full
        std::vector<float> C_temp_scaled_host((size_t)M * ldc_host_full);
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                C_temp_scaled_host[r * ldc_host_full + c] = C_host_result[r * ldc_host_full + c] * host_beta_overall;
            }
        }
        CUDA_CHECK(cudaMemcpy2DAsync(d_C_full, ldc_host_full * sizeof(float), C_temp_scaled_host.data(), ldc_host_full * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice, main_stream));
    }

    // Submatrix Device Pointers
    const float* A00_d = d_A_full;
    const float* A01_d = d_A_full + k2;
    const float* A10_d = d_A_full + (size_t)m2 * lda_host_full;
    const float* A11_d = d_A_full + (size_t)m2 * lda_host_full + k2;

    const float* B00_d = d_B_full;
    const float* B01_d = d_B_full + n2;
    const float* B10_d = d_B_full + (size_t)k2 * ldb_host_full;
    const float* B11_d = d_B_full + (size_t)k2 * ldb_host_full + n2;

    float* C00_d = d_C_full;
    float* C01_d = d_C_full + n2;
    float* C10_d = d_C_full + (size_t)m2 * ldc_host_full;
    float* C11_d = d_C_full + (size_t)m2 * ldc_host_full + n2;

    // Workspace Allocation (Individual cudaMallocAsync on main_stream)
    size_t s_matrix_bytes = (size_t)m2 * k2 * sizeof(float); // ld_S_m2k2 = k2
    size_t t_matrix_bytes = (size_t)k2 * n2 * sizeof(float); // ld_T_k2n2 = n2
    size_t m_matrix_bytes = (size_t)m2 * n2 * sizeof(float); // ld_M_m2n2 = n2

    float *S0_d = nullptr, *S1_d = nullptr, *S4_d = nullptr, *S5_d = nullptr, *S6_d = nullptr;
    float *T0_d = nullptr, *T2_d = nullptr, *T3_d = nullptr, *T5_d = nullptr, *T6_d = nullptr;
    float *M0_d = nullptr, *M1_d = nullptr, *M2_d = nullptr, *M3_d = nullptr, *M4_d = nullptr, *M5_d = nullptr, *M6_d = nullptr;

    CUDA_CHECK(cudaMallocAsync((void**)&S0_d, s_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&S1_d, s_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&S4_d, s_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&S5_d, s_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&S6_d, s_matrix_bytes, main_stream));

    CUDA_CHECK(cudaMallocAsync((void**)&T0_d, t_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&T2_d, t_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&T3_d, t_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&T5_d, t_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&T6_d, t_matrix_bytes, main_stream));

    CUDA_CHECK(cudaMallocAsync((void**)&M0_d, m_matrix_bytes, main_stream)); // M_i can also use specific streams if alloc is bottleneck
    CUDA_CHECK(cudaMallocAsync((void**)&M1_d, m_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M2_d, m_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M3_d, m_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M4_d, m_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M5_d, m_matrix_bytes, main_stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M6_d, m_matrix_bytes, main_stream));

    const float* S2_ptr_d = A00_d; const int ld_S2 = lda_host_full;
    const float* S3_ptr_d = A11_d; const int ld_S3 = lda_host_full;
    const float* T1_ptr_d = B00_d; const int ld_T1 = ldb_host_full;
    const float* T4_ptr_d = B11_d; const int ld_T4 = ldb_host_full;

    // Compute S_i and T_i terms (on main_stream)
    matrix_add_gpu(cublas_handle, m2, k2, A00_d, lda_host_full, A11_d, lda_host_full, S0_d, ld_S_m2k2, main_stream);
    matrix_add_gpu(cublas_handle, m2, k2, A10_d, lda_host_full, A11_d, lda_host_full, S1_d, ld_S_m2k2, main_stream);
    matrix_add_gpu(cublas_handle, m2, k2, A00_d, lda_host_full, A01_d, lda_host_full, S4_d, ld_S_m2k2, main_stream);
    matrix_subtract_gpu(cublas_handle, m2, k2, A10_d, lda_host_full, A00_d, lda_host_full, S5_d, ld_S_m2k2, main_stream);
    matrix_subtract_gpu(cublas_handle, m2, k2, A01_d, lda_host_full, A11_d, lda_host_full, S6_d, ld_S_m2k2, main_stream);

    matrix_add_gpu(cublas_handle, k2, n2, B00_d, ldb_host_full, B11_d, ldb_host_full, T0_d, ld_T_k2n2, main_stream);
    matrix_subtract_gpu(cublas_handle, k2, n2, B01_d, ldb_host_full, B11_d, ldb_host_full, T2_d, ld_T_k2n2, main_stream);
    matrix_subtract_gpu(cublas_handle, k2, n2, B10_d, ldb_host_full, B00_d, ldb_host_full, T3_d, ld_T_k2n2, main_stream);
    matrix_add_gpu(cublas_handle, k2, n2, B00_d, ldb_host_full, B01_d, ldb_host_full, T5_d, ld_T_k2n2, main_stream);
    matrix_add_gpu(cublas_handle, k2, n2, B10_d, ldb_host_full, B11_d, ldb_host_full, T6_d, ld_T_k2n2, main_stream);

    CUDA_CHECK(cudaStreamSynchronize(main_stream)); // Ensure S and T computations are complete before M computations start using them

    // Compute Inner 7 Products (M_i) using 1-Level Fused Strassen on different streams
    const float alpha_inner = 1.0f; // M_i = 1.0 * (S_a * T_b)
    const float beta_inner = 0.0f;  // M_i are results, not accumulations

    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S0_d, ld_S_m2k2, T0_d, ld_T_k2n2, beta_inner, M0_d, ld_M_m2n2, cublas_handle, streams[0]);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S1_d, ld_S_m2k2, T1_ptr_d, ld_T1, beta_inner, M1_d, ld_M_m2n2, cublas_handle, streams[1]);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S2_ptr_d, ld_S2, T2_d, ld_T_k2n2, beta_inner, M2_d, ld_M_m2n2, cublas_handle, streams[2]);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S3_ptr_d, ld_S3, T3_d, ld_T_k2n2, beta_inner, M3_d, ld_M_m2n2, cublas_handle, streams[3]);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S4_d, ld_S_m2k2, T4_ptr_d, ld_T4, beta_inner, M4_d, ld_M_m2n2, cublas_handle, streams[4]);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S5_d, ld_S_m2k2, T5_d, ld_T_k2n2, beta_inner, M5_d, ld_M_m2n2, cublas_handle, streams[5]);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S6_d, ld_S_m2k2, T6_d, ld_T_k2n2, beta_inner, M6_d, ld_M_m2n2, cublas_handle, streams[6]);

    // Synchronize all streams for M products before C accumulation
    for (int i = 0; i < num_streams_for_M_products; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // Combine M_i to form final C submatrices (on main_stream)
    // C00 = M0+M3-M4+M6
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M0_d, ld_M_m2n2, C00_d, ldc_host_full, main_stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M3_d, ld_M_m2n2, C00_d, ldc_host_full, main_stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, -host_alpha_overall,M4_d, ld_M_m2n2, C00_d, ldc_host_full, main_stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M6_d, ld_M_m2n2, C00_d, ldc_host_full, main_stream);

    // C01 = M2+M4
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M2_d, ld_M_m2n2, C01_d, ldc_host_full, main_stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M4_d, ld_M_m2n2, C01_d, ldc_host_full, main_stream);

    // C10 = M1+M3
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M1_d, ld_M_m2n2, C10_d, ldc_host_full, main_stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M3_d, ld_M_m2n2, C10_d, ldc_host_full, main_stream);

    // C11 = M0-M1+M2+M5
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M0_d, ld_M_m2n2, C11_d, ldc_host_full, main_stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, -host_alpha_overall,M1_d, ld_M_m2n2, C11_d, ldc_host_full, main_stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M2_d, ld_M_m2n2, C11_d, ldc_host_full, main_stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha_overall, M5_d, ld_M_m2n2, C11_d, ldc_host_full, main_stream);

    CUDA_CHECK(cudaStreamSynchronize(main_stream)); // Ensure C combinations are complete

    // Copy Result to Host
    CUDA_CHECK(cudaMemcpy2DAsync(C_host_result, ldc_host_full * sizeof(float), d_C_full, ldc_host_full * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost, main_stream));
    CUDA_CHECK(cudaStreamSynchronize(main_stream)); // Ensure final D2H copy is complete

    // Cleanup
    CUDA_CHECK(cudaFreeAsync(S0_d, main_stream)); CUDA_CHECK(cudaFreeAsync(S1_d, main_stream));
    CUDA_CHECK(cudaFreeAsync(S4_d, main_stream)); CUDA_CHECK(cudaFreeAsync(S5_d, main_stream));
    CUDA_CHECK(cudaFreeAsync(S6_d, main_stream));
    CUDA_CHECK(cudaFreeAsync(T0_d, main_stream)); CUDA_CHECK(cudaFreeAsync(T2_d, main_stream));
    CUDA_CHECK(cudaFreeAsync(T3_d, main_stream)); CUDA_CHECK(cudaFreeAsync(T5_d, main_stream));
    CUDA_CHECK(cudaFreeAsync(T6_d, main_stream));
    CUDA_CHECK(cudaFreeAsync(M0_d, main_stream)); CUDA_CHECK(cudaFreeAsync(M1_d, main_stream));
    CUDA_CHECK(cudaFreeAsync(M2_d, main_stream)); CUDA_CHECK(cudaFreeAsync(M3_d, main_stream));
    CUDA_CHECK(cudaFreeAsync(M4_d, main_stream)); CUDA_CHECK(cudaFreeAsync(M5_d, main_stream));
    CUDA_CHECK(cudaFreeAsync(M6_d, main_stream));
    
    CUDA_CHECK(cudaFreeAsync(d_A_full, main_stream));
    CUDA_CHECK(cudaFreeAsync(d_B_full, main_stream));
    CUDA_CHECK(cudaFreeAsync(d_C_full, main_stream));
    
    for (int i = 0; i < num_streams_for_M_products; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    // Note: cublas_handle is managed by the caller
}

#endif // SGEMM_HYBRID_2LEVEL_DETAILED_CUH_
