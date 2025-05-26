#ifndef SGEMM_HYBRID_2LEVEL_VC_CUH_
#define SGEMM_HYBRID_2LEVEL_VC_CUH_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common/helper_cuda.h" // Assumed to have checkCudaErrors
#include "src/cublas_helpers.cuh"
#include "src/sgemm_strassen_fused.cuh" // For sgemm_strassen_1level_fused_vP

#include <vector>
#include <cstdio> // For printf or stderr
#include <stdexcept> // For std::runtime_error

// Basic error checking macros (if not already in helper_cuda.h or to be sure)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = call;                                              \
        if (status != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(status));                                \
            throw std::runtime_error(cudaGetErrorString(status));               \
        }                                                                       \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d code=%d\n", __FILE__, __LINE__, status);\
            throw std::runtime_error("cuBLAS error");                           \
        }                                                                       \
    } while (0)
#endif


// Host function for Hybrid 2-Level Strassen SGEMM (Version C)
void sgemm_strassen_hybrid_2level_vC(
    int M, int N, int K,
    const float host_alpha,
    const float* A_host_full, int lda_host_full,
    const float* B_host_full, int ldb_host_full,
    const float host_beta,
    float* C_host_result, int ldc_host_full,
    cublasHandle_t& cublas_handle, 
    cudaStream_t stream = 0
) {
    // --- Initial Setup ---
    if (M % 4 != 0 || N % 4 != 0 || K % 4 != 0) {
        char err_msg[200];
        snprintf(err_msg, sizeof(err_msg), "sgemm_strassen_hybrid_2level_vC: M, N, K must be divisible by 4. Got M=%d, N=%d, K=%d", M,N,K);
        fprintf(stderr, "%s\n", err_msg);
        throw std::runtime_error(err_msg);
    }

    const int m2 = M / 2;
    const int n2 = N / 2;
    const int k2 = K / 2;

    float *d_A_full = nullptr, *d_B_full = nullptr, *d_C_full = nullptr;
    float *d_C_temp_for_beta_scaling = nullptr; // For beta scaling if needed

    // Device memory allocation
    CUDA_CHECK(cudaMallocAsync((void**)&d_A_full, (size_t)M * lda_host_full * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_B_full, (size_t)K * ldb_host_full * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_C_full, (size_t)M * ldc_host_full * sizeof(float), stream));

    // Copy host_A to d_A_full, host_B to d_B_full
    CUDA_CHECK(cudaMemcpy2DAsync(d_A_full, lda_host_full * sizeof(float), A_host_full, lda_host_full * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(d_B_full, ldb_host_full * sizeof(float), B_host_full, ldb_host_full * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice, stream));

    // Initialize d_C_full based on host_beta
    if (host_beta == 0.0f) {
        CUDA_CHECK(cudaMemsetAsync(d_C_full, 0, (size_t)M * ldc_host_full * sizeof(float), stream));
    } else {
        // Copy C_host_result to d_C_full first (if beta is not 0)
        CUDA_CHECK(cudaMemcpy2DAsync(d_C_full, ldc_host_full * sizeof(float), C_host_result, ldc_host_full * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice, stream));
        if (host_beta != 1.0f) {
            // Scale d_C_full by host_beta in-place. cublasSscal scales a vector.
            // For a matrix, we can treat it as a long vector M*N elements.
            // However, ldc might mean it's not contiguous.
            // Using cublasSgeam C = beta*C + 0*C (effectively C = beta*C, in-place if B is C)
            // This requires C to be both input B and output C.
            const float beta_zero_for_sgeam = 0.0f;
             // C_full = host_beta * C_full (copied from host) + 0.0 * C_full
            CUBLAS_CHECK(cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     M, N,
                                     &host_beta, d_C_full, ldc_host_full,
                                     &beta_zero_for_sgeam, d_C_full, ldc_host_full, // Dummy B, can be same as C if beta_op(B) = 0
                                     d_C_full, ldc_host_full));
        }
    }
    // At this point, d_C_full contains beta * C_initial_on_host (or is zero if beta was 0)

    // --- Submatrix Device Pointers ---
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

    // --- Workspace Allocation (Individual Allocations for Version C) ---
    size_t s_matrix_size_bytes = (size_t)m2 * k2 * sizeof(float);
    size_t t_matrix_size_bytes = (size_t)k2 * n2 * sizeof(float);
    size_t m_matrix_size_bytes = (size_t)m2 * n2 * sizeof(float);

    float *S0_dev = nullptr, *S1_dev = nullptr, *S4_dev = nullptr, *S5_dev = nullptr, *S6_dev = nullptr;
    float *T0_dev = nullptr, *T2_dev = nullptr, *T3_dev = nullptr, *T5_dev = nullptr, *T6_dev = nullptr;
    float *M0_dev = nullptr, *M1_dev = nullptr, *M2_dev = nullptr, *M3_dev = nullptr, *M4_dev = nullptr, *M5_dev = nullptr, *M6_dev = nullptr;

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

    // S2 and S3 are direct pointers to Aij submatrices
    const float* S2_ptr = A00_dev;
    const float* S3_ptr = A11_dev;
    // T1 and T4 are direct pointers to Bij submatrices
    const float* T1_ptr = B00_dev;
    const float* T4_ptr = B11_dev;

    // --- Compute S_i and T_i terms using cublas_helpers ---
    // Assuming cublas_helpers functions use the default stream or need update for stream param
    // For now, let's assume they are okay or a sync is needed if they don't take streams.
    // If matrix_add_gpu and matrix_subtract_gpu are synchronous or use default stream,
    // and this function uses a non-default stream, synchronization might be needed.
    // For this implementation, we assume they will operate correctly on the given stream
    // if cublasSetStream(handle, stream) is called before these operations.
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    matrix_add_gpu(cublas_handle, m2, k2, A00_dev, lda_host_full, A11_dev, lda_host_full, S0_dev, k2);
    matrix_add_gpu(cublas_handle, m2, k2, A10_dev, lda_host_full, A11_dev, lda_host_full, S1_dev, k2);
    matrix_add_gpu(cublas_handle, m2, k2, A00_dev, lda_host_full, A01_dev, lda_host_full, S4_dev, k2);
    matrix_subtract_gpu(cublas_handle, m2, k2, A10_dev, lda_host_full, A00_dev, lda_host_full, S5_dev, k2);
    matrix_subtract_gpu(cublas_handle, m2, k2, A01_dev, lda_host_full, A11_dev, lda_host_full, S6_dev, k2);

    matrix_add_gpu(cublas_handle, k2, n2, B00_dev, ldb_host_full, B11_dev, ldb_host_full, T0_dev, n2);
    matrix_subtract_gpu(cublas_handle, k2, n2, B01_dev, ldb_host_full, B11_dev, ldb_host_full, T2_dev, n2);
    matrix_subtract_gpu(cublas_handle, k2, n2, B10_dev, ldb_host_full, B00_dev, ldb_host_full, T3_dev, n2);
    matrix_add_gpu(cublas_handle, k2, n2, B00_dev, ldb_host_full, B01_dev, ldb_host_full, T5_dev, n2);
    matrix_add_gpu(cublas_handle, k2, n2, B10_dev, ldb_host_full, B11_dev, ldb_host_full, T6_dev, n2);

    // --- Compute Inner 7 Products (M_i) using 1-Level Fused Strassen ---
    const float alpha_inner = 1.0f;
    const float beta_inner = 0.0f; // M_i = S_a * T_b (no accumulation into M_i itself)

    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S0_dev, k2, T0_dev, n2, beta_inner, M0_dev, n2, cublas_handle, stream);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S1_dev, k2, T1_ptr, ldb_host_full, beta_inner, M1_dev, n2, cublas_handle, stream);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S2_ptr, lda_host_full, T2_dev, n2, beta_inner, M2_dev, n2, cublas_handle, stream);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S3_ptr, lda_host_full, T3_dev, n2, beta_inner, M3_dev, n2, cublas_handle, stream);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S4_dev, k2, T4_ptr, ldb_host_full, beta_inner, M4_dev, n2, cublas_handle, stream);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S5_dev, k2, T5_dev, n2, beta_inner, M5_dev, n2, cublas_handle, stream);
    sgemm_strassen_1level_fused_vP(m2, n2, k2, alpha_inner, S6_dev, k2, T6_dev, n2, beta_inner, M6_dev, n2, cublas_handle, stream);
    
    // Note: sgemm_strassen_1level_fused_vP is refactored to not sync internally.
    // A single sync after all 7 M_i products might be possible if there are no cross-dependencies for workspace.
    // However, each M_i is independent. For safety and clarity of stages, can sync after all are launched.
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Ensure all M_i computations are complete

    // --- Combine M_i to form final C submatrices ---
    // C = alpha * (sum of M_i terms) + beta * C_initial (already in d_C_full)
    // So we do: C_sub = C_sub + host_alpha * M_i_sub (or -host_alpha * M_i_sub)
    // This uses matrix_accumulate_scaled_add_gpu: C_io = scale_A * A + C_io (beta_for_C_io=1.0f)

    // C00 = M0+M3-M4+M6
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M0_dev, n2, C00_dev, ldc_host_full);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M3_dev, n2, C00_dev, ldc_host_full);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, -host_alpha, M4_dev, n2, C00_dev, ldc_host_full);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M6_dev, n2, C00_dev, ldc_host_full);

    // C01 = M2+M4
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M2_dev, n2, C01_dev, ldc_host_full);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M4_dev, n2, C01_dev, ldc_host_full);

    // C10 = M1+M3
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M1_dev, n2, C10_dev, ldc_host_full);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M3_dev, n2, C10_dev, ldc_host_full);

    // C11 = M0-M1+M2+M5
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M0_dev, n2, C11_dev, ldc_host_full);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, -host_alpha, M1_dev, n2, C11_dev, ldc_host_full);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M2_dev, n2, C11_dev, ldc_host_full);
    matrix_accumulate_scaled_add_gpu(cublas_handle, m2, n2, host_alpha, M5_dev, n2, C11_dev, ldc_host_full);

    CUDA_CHECK(cudaStreamSynchronize(stream)); // Ensure C combinations are complete

    // --- Copy Result to Host ---
    CUDA_CHECK(cudaMemcpy2DAsync(C_host_result, ldc_host_full * sizeof(float), d_C_full, ldc_host_full * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Ensure final D2H copy is complete

    // --- Cleanup ---
    CUDA_CHECK(cudaFreeAsync(S0_dev, stream)); CUDA_CHECK(cudaFreeAsync(S1_dev, stream));
    CUDA_CHECK(cudaFreeAsync(S4_dev, stream)); CUDA_CHECK(cudaFreeAsync(S5_dev, stream));
    CUDA_CHECK(cudaFreeAsync(S6_dev, stream));
    CUDA_CHECK(cudaFreeAsync(T0_dev, stream)); CUDA_CHECK(cudaFreeAsync(T2_dev, stream));
    CUDA_CHECK(cudaFreeAsync(T3_dev, stream)); CUDA_CHECK(cudaFreeAsync(T5_dev, stream));
    CUDA_CHECK(cudaFreeAsync(T6_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M0_dev, stream)); CUDA_CHECK(cudaFreeAsync(M1_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M2_dev, stream)); CUDA_CHECK(cudaFreeAsync(M3_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M4_dev, stream)); CUDA_CHECK(cudaFreeAsync(M5_dev, stream));
    CUDA_CHECK(cudaFreeAsync(M6_dev, stream));
    
    CUDA_CHECK(cudaFreeAsync(d_A_full, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_full, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_full, stream));
    
    // A final sync to ensure frees complete if stream is not default,
    // though technically the stream belongs to the caller.
    // For robust standalone behavior of this function if it were the top level:
    if (stream != 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

#endif // SGEMM_HYBRID_2LEVEL_VC_CUH_
