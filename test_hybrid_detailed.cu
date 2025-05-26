#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath> // For fabs
#include <stdexcept> // For std::runtime_error
#include <numeric>   // For std::iota if needed
#include <algorithm> // For std::max_element if needed
#include <iostream>  // For std::cout, std::cerr
#include <iomanip>   // For std::fixed, std::setprecision

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Assuming these are the correct paths relative to where this test file will be compiled from
#include "common/helper_cuda.h"       // Contains CUDA_CHECK, CUBLAS_CHECK
#include "src/sgemm.cuh"              // For reference GPU sgemm
#include "src/sgemm_hybrid_2level_detailed.cuh" // The function to test

// CUDA_CHECK and CUBLAS_CHECK are now expected to be defined in common/helper_cuda.h
// and throw std::runtime_error.

// Helper function to initialize a host matrix with random values
void init_matrix_host_random(std::vector<float>& matrix, int rows, int cols, int ld) {
    if (matrix.size() != (size_t)rows * ld) {
        matrix.resize((size_t)rows * ld);
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            matrix[r * ld + c] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 10.0f) - 5.0f; // Random between -5 and 5
        }
    }
}

// CPU GEMM for reference
void gemm_cpu_reference(
    int M, int N, int K,
    float alpha,
    const std::vector<float>& A, int lda,
    const std::vector<float>& B, int ldb,
    float beta,
    const std::vector<float>& C_initial, int ldc_initial, // C used for beta term
    std::vector<float>& C_ref, int ldc_ref             // Output C
) {
    if (C_ref.size() != (size_t)M * ldc_ref) {
        C_ref.resize((size_t)M * ldc_ref);
    }

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[r * lda + i] * B[i * ldb + c];
            }
            float initial_c_val = (C_initial.empty() || (r * ldc_initial + c) >= C_initial.size()) ? 0.0f : C_initial[r * ldc_initial + c];
            if (beta == 0.0f) {
                C_ref[r * ldc_ref + c] = alpha * sum;
            } else {
                C_ref[r * ldc_ref + c] = alpha * sum + beta * initial_c_val;
            }
        }
    }
}

// Detailed matrix comparison function
struct ComparisonResult {
    bool passed;
    double max_abs_diff;
    double avg_rel_diff;
    int num_errors;
    int total_elements;
};

ComparisonResult compare_matrices_detailed(
    int M, int N,
    const std::vector<float>& ref_matrix, int ld_ref,
    const std::vector<float>& test_matrix, int ld_test,
    float epsilon
) {
    ComparisonResult res = {true, 0.0, 0.0, 0, M*N};
    double total_relative_error = 0.0;
    int non_zero_refs = 0;

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float ref_val = ref_matrix[r * ld_ref + c];
            float test_val = test_matrix[r * ld_test + c];
            float abs_diff = std::fabs(ref_val - test_val);

            if (abs_diff > res.max_abs_diff) {
                res.max_abs_diff = abs_diff;
            }

            if (ref_val != 0.0f) {
                total_relative_error += abs_diff / std::fabs(ref_val);
                non_zero_refs++;
            }
            
            if (abs_diff > epsilon) {
                if (res.passed && res.num_errors < 10) { // Print first few errors
                    fprintf(stderr, "Error at [%d,%d]: Ref=%.6f, Test=%.6f, Diff=%.6f\n", r, c, ref_val, test_val, abs_diff);
                }
                res.passed = false;
                res.num_errors++;
            }
        }
    }
    if (non_zero_refs > 0) {
        res.avg_rel_diff = total_relative_error / non_zero_refs;
    } else {
        res.avg_rel_diff = 0.0; // Avoid division by zero if all refs are zero
    }
    return res;
}


struct TestConfig {
    int M, N, K;
    float alpha, beta;
    std::string id;
};

int main(int argc, char** argv) {
    srand(time(NULL));

    // Command-line parsing (simplified for brevity, or use helper_string.h if available/integrated)
    int M_default = 256, N_default = 256, K_default = 256;
    if (argc > 1) M_default = atoi(argv[1]);
    if (argc > 2) N_default = atoi(argv[2]);
    if (argc > 3) K_default = atoi(argv[3]);
    if (M_default % 4 != 0 || N_default % 4 != 0 || K_default % 4 != 0 || M_default <4 || N_default < 4 || K_default < 4) {
        fprintf(stderr, "M, N, K must be >= 4 and divisible by 4. Using default 256x256x256.\n");
        M_default = 256; N_default = 256; K_default = 256;
    }
    

    std::vector<TestConfig> test_configs = {
        {M_default, N_default, K_default, 1.0f, 0.0f, "Default_beta0"},
        {M_default, N_default, K_default, 1.0f, 0.75f, "Default_beta0.75"},
        {64, 64, 64, 1.0f, 0.0f, "64_beta0"},
        {64, 64, 64, 1.0f, 1.0f, "64_beta1"},
        {128, 128, 128, 1.0f, 0.0f, "128_beta0"},
        {128, 128, 128, 1.0f, 0.5f, "128_beta0.5"},
        {256, 128, 512, 1.0f, 0.0f, "256x128x512_beta0"},
        {256, 128, 512, 1.0f, 1.0f, "256x128x512_beta1"},
    };

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    cudaStream_t main_stream_for_ref_gpu; // Stream for reference GPU SGEMM
    CUDA_CHECK(cudaStreamCreateWithFlags(&main_stream_for_ref_gpu, cudaStreamNonBlocking));


    bool all_tests_passed = true;
    std::cout << std::fixed << std::setprecision(6);

    for (const auto& cfg : test_configs) {
        int M = cfg.M, N = cfg.N, K = cfg.K;
        float alpha = cfg.alpha, beta = cfg.beta;
        int lda = K, ldb = N, ldc = N; // Row-major

        printf("\nRunning Test: ID=%s, M=%d, N=%d, K=%d, alpha=%.2f, beta=%.2f\n", 
               cfg.id.c_str(), M, N, K, alpha, beta);

        std::vector<float> h_A(M * lda);
        std::vector<float> h_B(K * ldb);
        std::vector<float> h_C_initial(M * ldc);
        std::vector<float> h_C_hybrid_result(M * ldc);
        std::vector<float> h_C_gpu_ref_result(M * ldc);
        std::vector<float> h_C_cpu_ref(M * ldc);

        init_matrix_host_random(h_A, M, K, lda);
        init_matrix_host_random(h_B, K, N, ldb);
        if (beta != 0.0f) {
            init_matrix_host_random(h_C_initial, M, N, ldc);
        } else {
            std::fill(h_C_initial.begin(), h_C_initial.end(), 0.0f);
        }
        
        // Prepare C for hybrid (it expects C_host_result to have initial values for beta scaling)
        h_C_hybrid_result = h_C_initial; 


        // --- Hybrid Strassen Call ---
        try {
            printf("Executing Hybrid Strassen (Detailed Version D)...\n");
            sgemm_strassen_hybrid_2level_detailed(M, N, K, alpha, 
                                                  h_A.data(), lda, 
                                                  h_B.data(), ldb, 
                                                  beta, 
                                                  h_C_hybrid_result.data(), ldc, 
                                                  cublas_handle); // Uses its own streams internally
            printf("Hybrid Strassen execution completed.\n");

            // --- GPU Reference SGEMM Call ---
            printf("Executing GPU Reference SGEMM...\n");
            float *d_A_ref = nullptr, *d_B_ref = nullptr, *d_C_ref = nullptr;
            CUDA_CHECK(cudaMallocAsync((void**)&d_A_ref, (size_t)M * lda * sizeof(float), main_stream_for_ref_gpu));
            CUDA_CHECK(cudaMallocAsync((void**)&d_B_ref, (size_t)K * ldb * sizeof(float), main_stream_for_ref_gpu));
            CUDA_CHECK(cudaMallocAsync((void**)&d_C_ref, (size_t)M * ldc * sizeof(float), main_stream_for_ref_gpu));

            CUDA_CHECK(cudaMemcpy2DAsync(d_A_ref, lda * sizeof(float), h_A.data(), lda * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice, main_stream_for_ref_gpu));
            CUDA_CHECK(cudaMemcpy2DAsync(d_B_ref, ldb * sizeof(float), h_B.data(), ldb * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice, main_stream_for_ref_gpu));
            
            if (beta == 0.0f) {
                CUDA_CHECK(cudaMemsetAsync(d_C_ref, 0, (size_t)M * ldc * sizeof(float), main_stream_for_ref_gpu));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(d_C_ref, ldc * sizeof(float), h_C_initial.data(), ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice, main_stream_for_ref_gpu));
            }
            
            // Assuming sgemm from sgemm.cuh is a global function, does not take handle/stream explicitly
            // It will use the default stream unless modified. For safety, sync its operations if it's not stream-aware.
            // The reference sgemm is expected to be a GPU kernel.
            // float alpha_ptr = alpha; float beta_ptr = beta; // sgemm might take pointers
            // sgemm(M, N, K, &alpha_ptr, d_A_ref, lda, d_B_ref, ldb, &beta_ptr, d_C_ref, ldc);
            // For now, using a simple sgemm call, assuming it's compatible.
            // The sgemm in src/sgemm.cuh needs alpha/beta by ptr.
            float const_alpha = alpha; float const_beta = beta; // For pointer passing
            sgemm(M, N, K, &const_alpha, d_A_ref, lda, d_B_ref, ldb, &const_beta, d_C_ref, ldc); // Assuming sgemm uses default stream or its own
            CUDA_CHECK(cudaStreamSynchronize(main_stream_for_ref_gpu)); // Sync after sgemm if it used main_stream_for_ref_gpu or default

            CUDA_CHECK(cudaMemcpy2DAsync(h_C_gpu_ref_result.data(), ldc * sizeof(float), d_C_ref, ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost, main_stream_for_ref_gpu));
            CUDA_CHECK(cudaStreamSynchronize(main_stream_for_ref_gpu));

            CUDA_CHECK(cudaFreeAsync(d_A_ref, main_stream_for_ref_gpu));
            CUDA_CHECK(cudaFreeAsync(d_B_ref, main_stream_for_ref_gpu));
            CUDA_CHECK(cudaFreeAsync(d_C_ref, main_stream_for_ref_gpu));
            printf("GPU Reference SGEMM execution completed.\n");

            // --- CPU Reference GEMM Calculation ---
            printf("Executing CPU Reference GEMM...\n");
            gemm_cpu_reference(M, N, K, alpha, h_A, lda, h_B, ldb, beta, h_C_initial, ldc, h_C_cpu_ref, ldc);
            printf("CPU Reference GEMM execution completed.\n");

            // --- Verification ---
            printf("Comparing Hybrid Strassen vs CPU Reference...\n");
            ComparisonResult res_vs_cpu = compare_matrices_detailed(M, N, h_C_cpu_ref, ldc, h_C_hybrid_result, ldc, 1e-3f);
            printf("Hybrid vs CPU: %s (Errors: %d/%d, MaxAbsDiff: %e, AvgRelDiff: %e)\n",
                   res_vs_cpu.passed ? "PASSED" : "FAILED", res_vs_cpu.num_errors, res_vs_cpu.total_elements, res_vs_cpu.max_abs_diff, res_vs_cpu.avg_rel_diff);
            if (!res_vs_cpu.passed) all_tests_passed = false;

            printf("Comparing Hybrid Strassen vs GPU Reference SGEMM...\n");
            ComparisonResult res_vs_gpu = compare_matrices_detailed(M, N, h_C_gpu_ref_result, ldc, h_C_hybrid_result, ldc, 1e-3f);
            printf("Hybrid vs GPU Ref: %s (Errors: %d/%d, MaxAbsDiff: %e, AvgRelDiff: %e)\n",
                   res_vs_gpu.passed ? "PASSED" : "FAILED", res_vs_gpu.num_errors, res_vs_gpu.total_elements, res_vs_gpu.max_abs_diff, res_vs_gpu.avg_rel_diff);
            if (!res_vs_gpu.passed) all_tests_passed = false;

        } catch (const std::runtime_error& e) {
            fprintf(stderr, "Test ID %s CRITICAL ERROR: %s\n", cfg.id.c_str(), e.what());
            all_tests_passed = false;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(main_stream_for_ref_gpu));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    printf("\n=============== OVERALL TEST SUMMARY ===============\n");
    if (all_tests_passed) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("ONE OR MORE TESTS FAILED\n");
    }
    printf("==================================================\n");

    return all_tests_passed ? 0 : 1;
}
