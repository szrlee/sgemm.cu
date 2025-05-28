#ifndef SGEMM_HYBRID_2LEVEL_VC_CUH_
#define SGEMM_HYBRID_2LEVEL_VC_CUH_

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Explicit forward declaration to match sgemm_strassen_fused.cuh
void sgemm_strassen_1level_fused_vP(int m,
                                    int n,
                                    int k,
                                    float alpha_sub,
                                    const float* d_A_sub,
                                    int ldA_sub,
                                    const float* d_B_sub,
                                    int ldB_sub,
                                    float beta_sub,
                                    float* d_C_sub,
                                    int ldC_sub,
                                    cublasHandle_t cublas_handle,
                                    cudaStream_t stream,
                                    bool is_first_kernel_launch_for_debug = false);

#include "cublas_helpers.cuh"
#include "helper_cuda.h"            // Assumed to have checkCudaErrors
#include "sgemm_strassen_fused.cuh" // Moved earlier, provides sgemm_strassen_1level_fused_vP

#include <cstdio>                   // For printf or stderr
#include <stdexcept>                // For std::runtime_error
#include <vector>

// Basic error checking macros (if not already in helper_cuda.h or to be sure)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t status = call;                                \
        if (status != cudaSuccess) {                              \
            fprintf(stderr,                                       \
                    "CUDA error at %s:%d: %s\n",                  \
                    __FILE__,                                     \
                    __LINE__,                                     \
                    cudaGetErrorString(status));                  \
            throw std::runtime_error(cudaGetErrorString(status)); \
        }                                                         \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                                  \
    do {                                                                                    \
        cublasStatus_t status = call;                                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuBLAS error at %s:%d code=%d\n", __FILE__, __LINE__, status); \
            throw std::runtime_error("cuBLAS error");                                       \
        }                                                                                   \
    } while (0)
#endif


// Host function for Hybrid 2-Level Strassen SGEMM (Version C)
void
sgemm_strassen_hybrid_2level_vC(int M,
                                int N,
                                int K,
                                const float host_alpha,
                                const float* A_host_full,
                                int lda_host_full,
                                const float* B_host_full,
                                int ldb_host_full,
                                const float host_beta,
                                float* C_host_result,
                                int ldc_host_full,
                                cublasHandle_t& cublas_handle,
                                cudaStream_t stream = 0) {
    // --- Initial Setup ---
    if (M % 4 != 0 || N % 4 != 0 || K % 4 != 0) {
        char err_msg[200];
        snprintf(
            err_msg,
            sizeof(err_msg),
            "sgemm_strassen_hybrid_2level_vC: M, N, K must be divisible by 4. Got M=%d, N=%d, K=%d",
            M,
            N,
            K);
        fprintf(stderr, "%s\n", err_msg);
        throw std::runtime_error(err_msg);
    }

    const int m2 = M / 2;
    const int n2 = N / 2;
    const int k2 = K / 2;

    float *d_A_full = nullptr, *d_B_full = nullptr, *d_C_full = nullptr;

    // Device memory allocation
    CUDA_CHECK(
        cudaMallocAsync((void**)&d_A_full, (size_t)M * lda_host_full * sizeof(float), stream));
    CUDA_CHECK(
        cudaMallocAsync((void**)&d_B_full, (size_t)K * ldb_host_full * sizeof(float), stream));

    CUDA_CHECK(
        cudaMallocAsync((void**)&d_C_full, (size_t)M * ldc_host_full * sizeof(float), stream));

    // Copy host_A to d_A_full, host_B to d_B_full
    CUDA_CHECK(cudaMemcpy2DAsync(d_A_full,
                                 lda_host_full * sizeof(float),
                                 A_host_full,
                                 lda_host_full * sizeof(float),
                                 K * sizeof(float),
                                 M,
                                 cudaMemcpyHostToDevice,
                                 stream));
    CUDA_CHECK(cudaMemcpy2DAsync(d_B_full,
                                 ldb_host_full * sizeof(float),
                                 B_host_full,
                                 ldb_host_full * sizeof(float),
                                 N * sizeof(float),
                                 K,
                                 cudaMemcpyHostToDevice,
                                 stream));

    // Initialize d_C_full based on host_beta
    if (host_beta == 0.0f) {
        CUDA_CHECK(cudaMemsetAsync(d_C_full, 0, (size_t)M * ldc_host_full * sizeof(float), stream));
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(d_C_full,
                                     ldc_host_full * sizeof(float),
                                     C_host_result,
                                     ldc_host_full * sizeof(float),
                                     N * sizeof(float),
                                     M,
                                     cudaMemcpyHostToDevice,
                                     stream));
        if (host_beta != 1.0f) {
            const float beta_zero_for_sgeam = 0.0f;
            CUBLAS_CHECK(cublasSgeam(cublas_handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     M,
                                     N,
                                     &host_beta,
                                     d_C_full,
                                     ldc_host_full,
                                     &beta_zero_for_sgeam,
                                     d_C_full,
                                     ldc_host_full,
                                     d_C_full,
                                     ldc_host_full));
        }
    }

    const float* A00_dev = d_A_full;
    const float* A01_dev = d_A_full + k2;
    const float* A10_dev = d_A_full + (size_t)m2 * lda_host_full;
    const float* A11_dev = d_A_full + (size_t)m2 * lda_host_full + k2;

    const float* B00_dev = d_B_full;
    const float* B01_dev = d_B_full + n2;
    const float* B10_dev = d_B_full + (size_t)k2 * ldb_host_full;
    const float* B11_dev = d_B_full + (size_t)k2 * ldb_host_full + n2;

    float* C00_dev = d_C_full;
    float* C01_dev = d_C_full + n2;
    float* C10_dev = d_C_full + (size_t)m2 * ldc_host_full;
    float* C11_dev = d_C_full + (size_t)m2 * ldc_host_full + n2;

    size_t s_matrix_size_bytes = (size_t)m2 * k2 * sizeof(float);
    size_t t_matrix_size_bytes = (size_t)k2 * n2 * sizeof(float);
    size_t m_matrix_size_bytes = (size_t)m2 * n2 * sizeof(float);

    float *S0_dev = nullptr, *S1_dev = nullptr, *S4_dev = nullptr, *S5_dev = nullptr,
          *S6_dev = nullptr;
    float *T0_dev = nullptr, *T2_dev = nullptr, *T3_dev = nullptr, *T5_dev = nullptr,
          *T6_dev = nullptr;
    float *M0_dev = nullptr, *M1_dev = nullptr, *M2_dev = nullptr, *M3_dev = nullptr,
          *M4_dev = nullptr, *M5_dev = nullptr, *M6_dev = nullptr;

    CUDA_CHECK(cudaMallocAsync((void**)&S0_dev, s_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&S1_dev, s_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&S4_dev, s_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&S5_dev, s_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&S6_dev, s_matrix_size_bytes, stream));

    CUDA_CHECK(cudaMallocAsync((void**)&T0_dev, t_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&T2_dev, t_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&T3_dev, t_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&T5_dev, t_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&T6_dev, t_matrix_size_bytes, stream));

    CUDA_CHECK(cudaMallocAsync((void**)&M0_dev, m_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M1_dev, m_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M2_dev, m_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M3_dev, m_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M4_dev, m_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M5_dev, m_matrix_size_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&M6_dev, m_matrix_size_bytes, stream));

    const float* S2_ptr = A00_dev;
    const float* S3_ptr = A11_dev;
    const float* T1_ptr = B00_dev;
    const float* T4_ptr = B11_dev;

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    matrix_add_gpu(cublas_handle,
                   m2,
                   k2,
                   A00_dev,
                   lda_host_full,
                   A11_dev,
                   lda_host_full,
                   S0_dev,
                   m2,
                   stream);
    matrix_add_gpu(cublas_handle,
                   m2,
                   k2,
                   A10_dev,
                   lda_host_full,
                   A11_dev,
                   lda_host_full,
                   S1_dev,
                   m2,
                   stream);
    matrix_add_gpu(cublas_handle,
                   m2,
                   k2,
                   A00_dev,
                   lda_host_full,
                   A01_dev,
                   lda_host_full,
                   S4_dev,
                   m2,
                   stream);
    matrix_subtract_gpu(cublas_handle,
                        m2,
                        k2,
                        A10_dev,
                        lda_host_full,
                        A00_dev,
                        lda_host_full,
                        S5_dev,
                        m2,
                        stream);
    matrix_subtract_gpu(cublas_handle,
                        m2,
                        k2,
                        A01_dev,
                        lda_host_full,
                        A11_dev,
                        lda_host_full,
                        S6_dev,
                        m2,
                        stream);

    matrix_add_gpu(cublas_handle,
                   k2,
                   n2,
                   B00_dev,
                   ldb_host_full,
                   B11_dev,
                   ldb_host_full,
                   T0_dev,
                   k2,
                   stream);
    matrix_subtract_gpu(cublas_handle,
                        k2,
                        n2,
                        B01_dev,
                        ldb_host_full,
                        B11_dev,
                        ldb_host_full,
                        T2_dev,
                        k2,
                        stream);
    matrix_subtract_gpu(cublas_handle,
                        k2,
                        n2,
                        B10_dev,
                        ldb_host_full,
                        B00_dev,
                        ldb_host_full,
                        T3_dev,
                        k2,
                        stream);
    matrix_add_gpu(cublas_handle,
                   k2,
                   n2,
                   B00_dev,
                   ldb_host_full,
                   B01_dev,
                   ldb_host_full,
                   T5_dev,
                   k2,
                   stream);
    matrix_add_gpu(cublas_handle,
                   k2,
                   n2,
                   B10_dev,
                   ldb_host_full,
                   B11_dev,
                   ldb_host_full,
                   T6_dev,
                   k2,
                   stream);

    printf("[HYBRID DEBUG] M0 computation/accumulation SKIPPED. M3 (using cublasSgemm) ENABLED. "
           "Others disabled.\n");
    fflush(stdout);

    const float const_float_one = 1.0f;
    const float const_float_zero = 0.0f;

    // M0 = host_alpha * S0 * T0
    /* CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             m2, n2, k2, 
                             &host_alpha, 
                             S0_dev, m2,       
                             T0_dev, k2,       
                             &const_float_zero, 
                             M0_dev, m2        
                             )); */
    printf("[HYBRID DEBUG] M0 computation skipped.\n");
    fflush(stdout);

    // M1 = host_alpha * S1 * T1
    /* CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m2, n2, k2,
                             &host_alpha,
                             S1_dev, m2,
                             T1_ptr, ldb_host_full, 
                             &const_float_zero,
                             M1_dev, m2
                             )); */

    // M2 = host_alpha * S2 * T2
    /* CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m2, n2, k2,
                             &host_alpha,
                             S2_ptr, lda_host_full, 
                             T2_dev, k2,
                             &const_float_zero,
                             M2_dev, m2
                             )); */

    // M3 = host_alpha * S3 * T3
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m2,
                             n2,
                             k2,
                             &host_alpha,
                             S3_ptr,
                             lda_host_full,
                             T3_dev,
                             k2,
                             &const_float_zero,
                             M3_dev,
                             m2));
    printf("[HYBRID DEBUG] M3 computation completed.\n");
    fflush(stdout);


    // M4 = host_alpha * S4 * T4
    /* CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m2, n2, k2,
                             &host_alpha,
                             S4_dev, m2,
                             T4_ptr, ldb_host_full, 
                             &const_float_zero,
                             M4_dev, m2
                             )); */

    // M5 = host_alpha * S5 * T5
    /* CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m2, n2, k2,
                             &host_alpha,
                             S5_dev, m2,
                             T5_dev, k2,
                             &const_float_zero,
                             M5_dev, m2
                             )); */


    // M6 = host_alpha * S6 * T6
    /* CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m2, n2, k2,
                             &host_alpha,
                             S6_dev, m2,
                             T6_dev, k2,
                             &const_float_zero,
                             M6_dev, m2
                             )); */

    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("[HYBRID DEBUG PRE-SYNC] About to synchronize device before C00_M0 accumulation.\n");
    fflush(stdout);

    // C00 = M0+M3-M4+M6
    printf("[HYBRID DEBUG C00_M0 HELPER] M=%d,N=%d,K=%d. m2=%d,n2=%d. M0_dev=%p, ldM0(n2)=%d. "
           "C00_dev=%p, ldC00(ldc_host_full)=%d. host_alpha=%f\n",
           M,
           N,
           K,
           m2,
           n2,
           (void*)M0_dev,
           n2,
           (void*)C00_dev,
           ldc_host_full,
           host_alpha);
    fflush(stdout);
    /* matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,    
                                     M0_dev,        
                                     n2,            
                                     C00_dev,       
                                     ldc_host_full, 
                                     stream); */
    printf("[HYBRID DEBUG C00_M0 HELPER] Call SKIPPED.\n");
    fflush(stdout);

    printf("[HYBRID HELPER C00_M3] m2=%d, n2=%d, host_alpha=%f, M3_dev=%p, ldM3=%d, C00_dev=%p, "
           "ldC00=%d\n",
           m2,
           n2,
           host_alpha,
           (void*)M3_dev,
           m2,
           (void*)C00_dev,
           ldc_host_full);
    fflush(stdout);
    matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,
                                     M3_dev,
                                     m2,
                                     C00_dev,
                                     ldc_host_full,
                                     stream);
    printf("[HYBRID HELPER C00_M3] Call completed.\n");
    fflush(stdout);

    printf("[HYBRID HELPER C00_M4] Accumulation for M4 on C00 (using -host_alpha) currently "
           "disabled for testing M0/M3.\n");
    fflush(stdout);
    /* matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     -host_alpha,   
                                     M4_dev,        
                                     m2,            // ldM4 should be m2
                                     C00_dev,       
                                     ldc_host_full, 
                                     stream); */

    printf("[HYBRID DEBUG] Accumulation for M6 on C00 currently disabled for testing M0/M3.\n");
    fflush(stdout);
    /* matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,   
                                     M6_dev,        
                                     m2,            // ldM6 should be m2
                                     C00_dev,       
                                     ldc_host_full, 
                                     stream); */


    printf("[HYBRID DEBUG] Accumulations for C01, C10, C11 currently disabled.\n");
    fflush(stdout);
    /* matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,
                                     M2_dev,
                                     m2, // ldM2
                                     C01_dev,
                                     ldc_host_full,
                                     stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,
                                     M4_dev,
                                     m2, // ldM4
                                     C01_dev,
                                     ldc_host_full,
                                     stream);

    matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,
                                     M1_dev,
                                     m2, // ldM1
                                     C10_dev,
                                     ldc_host_full,
                                     stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,
                                     M3_dev,
                                     m2, // ldM3
                                     C10_dev,
                                     ldc_host_full,
                                     stream);

    matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,
                                     M0_dev,
                                     m2, // ldM0
                                     C11_dev,
                                     ldc_host_full,
                                     stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     -host_alpha,
                                     M1_dev,
                                     m2, // ldM1
                                     C11_dev,
                                     ldc_host_full,
                                     stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,
                                     M2_dev,
                                     m2, // ldM2
                                     C11_dev,
                                     ldc_host_full,
                                     stream);
    matrix_accumulate_scaled_add_gpu(cublas_handle,
                                     m2,
                                     n2,
                                     host_alpha,
                                     M5_dev,
                                     m2, // ldM5
                                     C11_dev,
                                     ldc_host_full,
                                     stream); */

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy2DAsync(C_host_result,
                                 ldc_host_full * sizeof(float),
                                 d_C_full,
                                 ldc_host_full * sizeof(float),
                                 N * sizeof(float),
                                 M,
                                 cudaMemcpyDeviceToHost,
                                 stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(S0_dev, stream));
    CUDA_CHECK(cudaFreeAsync(S1_dev, stream));
    CUDA_CHECK(cudaFreeAsync(S4_dev, stream));
    CUDA_CHECK(cudaFreeAsync(S5_dev, stream));
    CUDA_CHECK(cudaFreeAsync(S6_dev, stream));
    CUDA_CHECK(cudaFreeAsync(T0_dev, stream));
    CUDA_CHECK(cudaFreeAsync(T2_dev, stream));
    CUDA_CHECK(cudaFreeAsync(T3_dev, stream));
    CUDA_CHECK(cudaFreeAsync(T5_dev, stream));
    CUDA_CHECK(cudaFreeAsync(T6_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M0_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M1_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M2_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M3_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M4_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M5_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M6_dev, stream));

    CUDA_CHECK(cudaFreeAsync(d_A_full, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_full, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_full, stream));

    if (stream != 0) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
}

#endif // SGEMM_HYBRID_2LEVEL_VC_CUH_
