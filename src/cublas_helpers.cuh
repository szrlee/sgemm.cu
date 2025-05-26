#ifndef CUBLAS_HELPERS_CUH_
#define CUBLAS_HELPERS_CUH_

#include <cublas_v2.h>
#include "common/helper_cuda.h" // For checkCudaErrors

// Wrapper Function 1: matrix_add_gpu
// Operation: C = A + B
inline void matrix_add_gpu(cublasHandle_t handle, int m, int n,
                           const float* A_dev, int lda,
                           const float* B_dev, int ldb,
                           float* C_dev, int ldc) {
    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasStatus_t status = cublasSgeam(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        m, n,
                                        &alpha,
                                        A_dev, lda,
                                        &beta,
                                        B_dev, ldb,
                                        C_dev, ldc);
    checkCudaErrors(status);
}

// Wrapper Function 2: matrix_subtract_gpu
// Operation: C = A - B
inline void matrix_subtract_gpu(cublasHandle_t handle, int m, int n,
                                const float* A_dev, int lda, // Matrix from which to subtract
                                const float* B_dev, int ldb, // Matrix to subtract
                                float* C_dev, int ldc) {    // Result C = A - B
    const float alpha = 1.0f;
    const float beta = -1.0f;
    cublasStatus_t status = cublasSgeam(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        m, n,
                                        &alpha,
                                        A_dev, lda,
                                        &beta,
                                        B_dev, ldb,
                                        C_dev, ldc);
    checkCudaErrors(status);
}

// Wrapper Function 3: matrix_accumulate_scaled_add_gpu
// Operation: C_io = scale_factor_for_A * A + 1.0 * C_io
// Precondition: A_dev must not be the same device memory pointer as C_io_dev.
// If A_dev == C_io_dev, the operation becomes C = (scale_factor_for_A + 1.0) * C,
// which might not be the intended accumulation if A was supposed to be a different matrix.
inline void matrix_accumulate_scaled_add_gpu(cublasHandle_t handle, int m, int n,
                                             float scale_factor_for_A, const float* A_dev, int lda,
                                             float* C_io_dev, int ldc) { // C_io = C_io + scale_factor_for_A * A
    const float beta_for_C_io = 1.0f;
    // cublasSgeam: C = alpha*op(A) + beta*op(B)
    // Here: C_io_dev = scale_factor_for_A * A_dev + beta_for_C_io * C_io_dev
    // So, A argument is A_dev, B argument is C_io_dev, C argument is C_io_dev
    cublasStatus_t status = cublasSgeam(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N, // op(A), op(B)
                                        m, n,
                                        &scale_factor_for_A,
                                        A_dev, lda,
                                        &beta_for_C_io,
                                        C_io_dev, ldc, // C_io_dev is matrix B
                                        C_io_dev, ldc); // C_io_dev is matrix C (output)
    checkCudaErrors(status);
}

// Wrapper Function 4: matrix_out_of_place_set_and_scale_gpu
// Operation: C_out = scale_factor * A_in + 0.0 * (dummy B)
// This function performs an out-of-place scaling: C_out = scale_factor * A_in.
// Precondition: For true out-of-place C_out = s*A_in, A_in_dev should generally not be the same as C_out_dev
// unless scale_factor is 1.0 (direct copy). If A_in_dev == C_out_dev and scale_factor != 1.0,
// this performs an in-place scale, which is C_out = scale_factor * C_out (original).
// If A_in_dev is different from C_out_dev, this works as C_out = scale_factor * A_in.
inline void matrix_out_of_place_set_and_scale_gpu(
                                cublasHandle_t handle, int m, int n,
                                float scale_factor, const float* A_in_dev, int lda,
                                float* C_out_dev, int ldc) { // C_out = scale_factor * A_in
    const float beta_zero = 0.0f;
    // cublasSgeam: C = alpha*op(A) + beta*op(B)
    // Here: C_out_dev = scale_factor * A_in_dev + 0.0 * B_dummy
    // B_dummy can be any valid device pointer of appropriate dimensions; its content is not accessed.
    // Using A_in_dev as the dummy B matrix pointer is safe as per cuBLAS docs for beta=0.
    cublasStatus_t status = cublasSgeam(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        m, n,
                                        &scale_factor,
                                        A_in_dev, lda,
                                        &beta_zero,
                                        A_in_dev, lda, // Dummy B matrix, its ldb (lda here)
                                        C_out_dev, ldc);
    checkCudaErrors(status);
}

#endif // CUBLAS_HELPERS_CUH_
