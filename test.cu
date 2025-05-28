#include <algorithm>  // For std::min, std::max
#include <cstdio>     // For printf, fprintf, fflush
#include <cstdlib>    // For std::getenv, srand, rand
#include <ctime>      // For time()
#include <filesystem> // For std::filesystem
#include <fstream>    // For std::ofstream
#include <iomanip>    // For std::fixed, std::setprecision
#include <iostream>   // For std::cout, std::cerr, std::endl
#include <numeric>    // For std::iota or other utilities
#include <stdexcept>  // For std::runtime_error
#include <string>     // For std::string
#include <vector>     // For std::vector

#include "helper_matrix.h"
#include "helper_string.h"
#include "sgemm.cuh"
#include "sgemm_hybrid_2level_detailed.cuh"
#include "sgemm_strassen_fused.cuh"

// CUDA Headers
#include <cublas_v2.h>    // For CUBLAS APIs
#include <cuda_runtime.h> // For CUDA runtime APIs

#define MATSIZE_MAX_DEFAULT  512
#define MATSIZE_MIN_DEFAULT  2
#define MATSIZE_STEP_DEFAULT 1
#define SAVEDIR_DEFAULT      "test_results"

// Helper function to print a sub-matrix
auto print_sub_matrix = [](const char* title,
                           const float* matrix,
                           int rows,
                           int cols,
                           int ld,
                           int print_rows,
                           int print_cols)
{
    printf("\n%s (up to %dx%d of %dx%d, ld=%d):\n", title, print_rows, print_cols, rows, cols, ld);
    for (int i = 0; i < std::min(rows, print_rows); ++i) {
        for (int j = 0; j < std::min(cols, print_cols); ++j) {
            printf("%8.3f ", matrix[i * ld + j]);
        }
        printf("\n");
    }
    printf("\n");
};

int
main(int argc, char** argv) {
    srand(time(NULL));

    cublasHandle_t cublas_handle_main; // Create main handle early
    CUBLAS_CHECK(cublasCreate(&cublas_handle_main));

    std::vector<std::string> args = {};
    for (int i = 1; i < argc; i++) {
        args.push_back(std::string{argv[i]});
    }

    int matsize_max = get_cmd_line_arg_int(args, "mmax", MATSIZE_MAX_DEFAULT);
    int matsize_min = get_cmd_line_arg_int(args, "mmin", MATSIZE_MIN_DEFAULT);
    int matsize_step = get_cmd_line_arg_int(args, "mstep", MATSIZE_STEP_DEFAULT);
    std::string save_dir = get_cmd_line_arg_string(args, "savedir", SAVEDIR_DEFAULT);

    int sep_len = 25;
    printf("%.*s\n", sep_len, "===================================================");
    printf("Testing...\n");
    printf("%.*s\n", sep_len, "===================================================");

    std::string test_summary{};
    std::string test_full_info{};
    std::string failed_tests{};
    const int n_tests = (matsize_max - matsize_min) / matsize_step + 1;
    std::vector<cmp_result> test_results(n_tests, cmp_result{});
    int n_failed = 0;

    // Matrix<float> host_C_cublas(M, N); // Removed as M, N are not defined here and var is unused
    // Matrix<float> host_C_custom(M, N); // Removed as M, N are not defined here and var is unused

    for (int i = 0; i < n_tests; i += 1) {
        size_t matsize = matsize_min + i * matsize_step;
        size_t m = matsize, n = matsize, k = matsize;
        size_t lda = k;
        size_t ldb = n;
        size_t ldc = n;

        float* A_host = alloc_mat_host(m * lda * sizeof(float));
        float* B_host = alloc_mat_host(k * ldb * sizeof(float));
        float* C_host = alloc_mat_host(m * ldc * sizeof(float));
        float* C_ref_host = alloc_mat_host(m * ldc * sizeof(float));

        init_random(A_host, m * lda);
        init_random(B_host, k * ldb);
        init_random(C_host, m * ldc);
        checkCudaErrors(
            cudaMemcpy(C_ref_host, C_host, m * ldc * sizeof(float), cudaMemcpyHostToDevice));

        float* A_device = alloc_mat_device(m * lda * sizeof(float));
        float* B_device = alloc_mat_device(k * ldb * sizeof(float));
        float* C_device = alloc_mat_device(m * ldc * sizeof(float));
        float* C_ref_device = alloc_mat_device(m * ldc * sizeof(float));

        float alpha = 1.5;
        float beta = 0.5;

        checkCudaErrors(
            cudaMemcpy(A_device, A_host, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(B_device, B_host, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(C_device, C_host, m * ldc * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(C_ref_device, C_ref_host, m * ldc * sizeof(float), cudaMemcpyHostToDevice));

        sgemm(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc);
        sgemm_basic(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_ref_device, ldc);
        cudaMemcpy(C_host, C_device, m * ldc * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref_host, C_ref_device, m * ldc * sizeof(float), cudaMemcpyDeviceToHost);

        cmp_result result = compare_mats(C_ref_host, C_host, m * ldc, 1e-4, false, true);
        std::string result_string{};
        if (!result.equal) {
            n_failed += 1;
            failed_tests += std::to_string(i);
            failed_tests += " ";
            result_string = "FAILED";
        } else {
            result_string = "PASSED";
        }
        printf("Test #%i | matsize = %lu | %s\n", i, matsize, result_string.c_str());

        test_full_info += "Test #" + std::to_string(i);
        test_full_info += " Matsize = " + std::to_string(matsize) + ": " + result.debug_info;
        test_results[i] = result;

        checkCudaErrors(cudaFreeHost(A_host));
        checkCudaErrors(cudaFreeHost(B_host));
        checkCudaErrors(cudaFreeHost(C_host));
        checkCudaErrors(cudaFreeHost(C_ref_host));
        checkCudaErrors(cudaFree(A_device));
        checkCudaErrors(cudaFree(B_device));
        checkCudaErrors(cudaFree(C_device));
        checkCudaErrors(cudaFree(C_ref_device));
        checkCudaErrors(cudaGetLastError());
    }
    test_summary += "\n=============== SUMMARY ===============\n";
    test_summary += "PASSED: " + std::to_string(n_tests - n_failed) + " / "
                    + std::to_string(n_tests) + "\n";
    test_summary += "FAILED: " + ((failed_tests.size() > 0) ? failed_tests : "0") + "\n";
    test_full_info += test_summary;
    printf("%s", test_summary.c_str());

    std::filesystem::path work_dir_path = std::filesystem::current_path();
    std::filesystem::path store_path = work_dir_path / save_dir / "sgemm.cu.txt";
    std::ofstream test_results_file(store_path);
    test_results_file << test_full_info.c_str();
    // test_results_file.close(); // Keep open to add Strassen results
    // printf("Test results stored in %s\n", store_path.c_str());


    // --- Test Strassen Fused SGEMM ---
    printf("%.*s\n", sep_len, "===================================================");
    printf("Testing Strassen Fused SGEMM...\n");
    printf("%.*s\n", sep_len, "===================================================");

    struct StrassenTestConfig {
        int M, N, K;
        float beta;
        std::string id;
    };

    std::vector<StrassenTestConfig> strassen_configs = {
        {64, 64, 64, 0.0f, "64x64x64_beta0_DEBUG_ONLY"},
        {128, 128, 128, 0.0f, "128x128x128_beta0"},
        {256, 256, 256, 0.0f, "256x256x256_beta0"},
        {512, 512, 512, 0.0f, "512x512x512_beta0"},
        {1024, 1024, 1024, 0.0f, "1024x1024x1024_beta0"},
        // Add more configurations as needed, e.g., non-square, different K
        // {2048, 2048, 2048, 0.0f, "2048x2048x2048_beta0"}, // May be too large for some GPUs
    };

    std::string strassen_test_summary =
        "\n=============== STRASSEN FUSED SUMMARY ===============\n";
    std::string strassen_test_full_info = "";
    int strassen_n_tests = strassen_configs.size();
    int strassen_n_failed = 0;
    int strassen_n_passed_global = 0;

    // Get environment variable to single out a test for compute-sanitizer
    const char* specific_test_env = std::getenv("SGEMM_SPECIFIC_STRASSEN_TEST_ID");
    std::string specific_test_id_str = specific_test_env ? specific_test_env : "";

    for (const auto& config : strassen_configs) {
        int m = config.M;
        int n = config.N;
        int k = config.K;
        float beta = config.beta;
        std::string test_id = config.id;

        // if (specific_test_env && test_id != specific_test_env) {
        //     printf("Skipping Strassen test: %s (specific_test_env is set to %s)\\n",
        //            test_id.c_str(),
        //            specific_test_env);
        //     strassen_n_tests--; // Adjust total tests count
        //     continue;
        // }

        printf("--- Strassen Test ID: %s (M=%d, N=%d, K=%d, beta=%.2f) ---\
",
               test_id.c_str(),
               m,
               n,
               k,
               beta);

        size_t lda = k;
        size_t ldb = n;
        size_t ldc = n;
        float alpha = 1.0f; // Using alpha=1.0 for Strassen tests for simplicity

        float* h_A = alloc_mat_host(m * lda * sizeof(float));
        float* h_B = alloc_mat_host(k * ldb * sizeof(float));
        float* h_C_ref_for_strassen = alloc_mat_host(m * ldc * sizeof(float));
        float* h_C_strassen_1L_vP = alloc_mat_host(m * ldc * sizeof(float));

        bool current_test_is_debug_target = false; // (test_id == "64x64x64_beta0_DEBUG_ONLY");

        if (current_test_is_debug_target) {
            printf("[DEBUG TEST] Initializing h_A with 1.0, h_B with 2.0\n");
            init_const(h_A, (size_t)m * k, 1.0f);
            init_const(h_B, (size_t)k * n, 2.0f);

            // Print specific input values for M0 debugging
            // M0=(A00+A11)(B00+B11). m2=M/2, n2=N/2, k2=K/2.
            // For M=N=K=64, m2=n2=k2=32.
            // A00 is at h_A[0], A11 is at h_A[m2*lda + k2]
            // B00 is at h_B[0], B11 is at h_B[k2*ldb + n2]
            if (m == 64 && k == 64 && n == 64) {
                printf("[DEBUG TEST INPUTS FOR M0 (Host)]\n");
                printf("h_A[0] (A00_00): %f\n", h_A[0]);
                printf("h_A[%d] (A11_00): %f\n", 32 * (int)lda + 32, h_A[32 * lda + 32]);
                printf("h_B[0] (B00_00): %f\n", h_B[0]);
                printf("h_B[%d] (B11_00): %f\n", 32 * (int)ldb + 32, h_B[32 * ldb + 32]);
                fflush(stdout);
            }
        } else {
            init_random(h_A, (size_t)m * k);
            init_random(h_B, (size_t)k * n);
        }

        // Initialize C matrices based on beta
        if (beta == 0.0f) {
            init_const(h_C_ref_for_strassen, (size_t)m * n, 0.0f);
            init_const(h_C_strassen_1L_vP,
                       (size_t)m * n,
                       0.0f); // Strassen should handle zeroing if beta=0, but good practice
        } else {
            // For non-zero beta, initialize C with some values to test beta blending
            if (current_test_is_debug_target) {
                printf("[DEBUG TEST] Initializing h_C_ref_for_strassen and h_C_strassen_1L_vP with "
                       "3.0 for beta testing\n");
                init_const(h_C_ref_for_strassen, (size_t)m * n, 3.0f);
                init_const(h_C_strassen_1L_vP, (size_t)m * n, 3.0f);
            } else {
                init_random(h_C_ref_for_strassen, (size_t)m * n);
                init_random(h_C_strassen_1L_vP, (size_t)m * n); // Copy for Strassen input
            }
        }

        float* d_A = alloc_mat_device(m * lda * sizeof(float));
        float* d_B = alloc_mat_device(k * ldb * sizeof(float));
        float* d_C_ref_for_strassen = alloc_mat_device(m * ldc * sizeof(float));
        float* d_C_strassen_1L_vP = alloc_mat_device(m * ldc * sizeof(float));

        checkCudaErrors(cudaMemcpy(d_A, h_A, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_C_ref_for_strassen,
                                   h_C_ref_for_strassen,
                                   m * ldc * sizeof(float),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_C_strassen_1L_vP,
                                   h_C_strassen_1L_vP,
                                   m * ldc * sizeof(float),
                                   cudaMemcpyHostToDevice));

        // Reference sgemm using cuBLAS
        CUBLAS_CHECK(cublasSgemm(cublas_handle_main, // Use main handle
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 n,
                                 m,
                                 k,
                                 &alpha,
                                 d_B, // B is first for cuBLAS when A is not transposed
                                 ldb,
                                 d_A,
                                 lda,
                                 &beta,
                                 d_C_ref_for_strassen,
                                 ldc));

        // 1-Level Strassen Fused vP (GPU-centric)
        sgemm_strassen_1level_fused_vP(
            m,
            n,
            k,
            alpha, // This is alpha_for_product
            d_A,
            lda,
            d_B,
            ldb,
            beta,                                 // This is beta_for_C
            d_C_strassen_1L_vP,
            ldc,
            cublas_handle_main,                   // Pass the main handle
            nullptr,                              // stream (0)
            current_test_is_debug_target);        // Pass true only for the debug target test case

        checkCudaErrors(cudaDeviceSynchronize()); // Synchronize after kernel launches

        checkCudaErrors(cudaMemcpy(h_C_ref_for_strassen,
                                   d_C_ref_for_strassen,
                                   m * ldc * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_C_strassen_1L_vP,
                                   d_C_strassen_1L_vP,
                                   m * ldc * sizeof(float),
                                   cudaMemcpyDeviceToHost));

        // Print sub-matrices for the debug target case
        if (current_test_is_debug_target) {
            print_sub_matrix("A (Host)", h_A, m, k, lda, 8, 8);
            print_sub_matrix("B (Host)", h_B, k, n, ldb, 8, 8);
            print_sub_matrix("C_ref (cuBLAS) (Host)", h_C_ref_for_strassen, m, n, ldc, 8, 8);
            print_sub_matrix("C_strassen_1L (Host)", h_C_strassen_1L_vP, m, n, ldc, 8, 8);
        }

        // --- Comparison ---
        cmp_result result_strassen =
            compare_mats(h_C_ref_for_strassen, h_C_strassen_1L_vP, m * ldc, 1e-1f, false, true);

        std::string result_strassen_string{};
        if (!result_strassen.equal) {
            strassen_n_failed += 1;
            result_strassen_string = "FAILED";
        } else {
            strassen_n_passed_global += 1;
            result_strassen_string = "PASSED";
        }
        printf("Strassen Test ID=%s | %s\n", test_id.c_str(), result_strassen_string.c_str());

        strassen_test_full_info += "Strassen Test ID=" + test_id;
        strassen_test_full_info += " M=" + std::to_string(m) + " N=" + std::to_string(n)
                                   + " K=" + std::to_string(k);
        strassen_test_full_info += " beta=" + std::to_string(beta)
                                   + "; Result: " + result_strassen_string;
        strassen_test_full_info += "; Details: " + result_strassen.debug_info + "\n";

        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C_ref_for_strassen));
        checkCudaErrors(cudaFreeHost(h_C_strassen_1L_vP));
    }

    strassen_test_summary += "PASSED: " + std::to_string(strassen_n_passed_global) + " / "
                             + std::to_string(strassen_n_tests) + "\n";
    if (strassen_n_failed > 0) { strassen_test_summary += "SOME STRASSEN TESTS FAILED.\n"; }
    strassen_test_full_info += strassen_test_summary;
    printf("%s", strassen_test_summary.c_str());
    test_results_file << strassen_test_full_info.c_str();


    // --- Test Hybrid 2-Level Strassen SGEMM ---
    /*
    printf("%.*s\n", sep_len, "===================================================");
    printf("Testing Hybrid 2-Level Strassen SGEMM...\n");
    printf("%.*s\n", sep_len, "===================================================");

    struct HybridStrassenTestConfig {
        int M, N, K;
        float beta;
        std::string id;
    };

    std::vector<HybridStrassenTestConfig> hybrid_configs = {
        {64, 64, 64, 0.0f, "Hybrid_64x64x64_beta0"},
        {64, 64, 64, 0.75f, "Hybrid_64x64x64_beta0.75"},
        {128, 128, 128, 0.0f, "Hybrid_128x128x128_beta0"},
        {128, 128, 128, 0.75f, "Hybrid_128x128x128_beta0.75"},
        {256, 256, 128, 0.0f, "Hybrid_256x256x128_beta0"}, // M,N,K div by 4
        {256, 256, 128, 0.75f, "Hybrid_256x256x128_beta0.75"},
        {128, 256, 256, 0.0f, "Hybrid_128x256x256_beta0"},
        {128, 256, 256, 0.75f, "Hybrid_128x256x256_beta0.75"},
        {512, 512, 512, 0.0f, "Hybrid_512x512x512_beta0"},
        {512, 512, 512, 0.75f, "Hybrid_512x512x512_beta0.75"}};

    std::string hybrid_test_summary = "\n=============== HYBRID STRASSEN SUMMARY ===============\n";
    std::string hybrid_test_full_info = "";
    int hybrid_n_tests = hybrid_configs.size();
    int hybrid_n_failed = 0;

    for (int i = 0; i < hybrid_n_tests; ++i) {
        const auto& cfg = hybrid_configs[i];
        size_t m = cfg.M, n = cfg.N, k = cfg.K;
        float beta_val = cfg.beta;
        float alpha_val = 1.0f;

        size_t lda = k;
        size_t ldb = n;
        size_t ldc = n;

        printf("Running Hybrid Strassen Test: ID=%s, M=%zu, N=%zu, K=%zu, beta=%.2f, alpha=%.2f\n",
               cfg.id.c_str(),
               m,
               n,
               k,
               beta_val,
               alpha_val);

        float* h_A = alloc_mat_host(m * lda * sizeof(float));
        float* h_B = alloc_mat_host(k * ldb * sizeof(float));
        float* h_C_initial = alloc_mat_host(m * ldc * sizeof(float));
        float* h_C_ref_gpu_out = alloc_mat_host(m * ldc * sizeof(float));
        float* h_C_hybrid_out = alloc_mat_host(m * ldc * sizeof(float));

        init_random(h_A, m * lda);
        init_random(h_B, k * ldb);
        init_random(h_C_initial, m * ldc);

        // --- Reference Calculation (using GPU sgemm from sgemm.cuh) ---
        float* d_A_ref = alloc_mat_device(m * lda * sizeof(float));
        float* d_B_ref = alloc_mat_device(k * ldb * sizeof(float));
        float* d_C_ref = alloc_mat_device(m * ldc * sizeof(float));

        checkCudaErrors(cudaMemcpy(d_A_ref, h_A, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B_ref, h_B, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(d_C_ref, h_C_initial, m * ldc * sizeof(float), cudaMemcpyHostToDevice));

        sgemm(m, n, k, &alpha_val, d_A_ref, lda, d_B_ref, ldb, &beta_val, d_C_ref, ldc);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(
            cudaMemcpy(h_C_ref_gpu_out, d_C_ref, m * ldc * sizeof(float), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_A_ref));
        checkCudaErrors(cudaFree(d_B_ref));
        checkCudaErrors(cudaFree(d_C_ref));

        // --- Hybrid Strassen Calculation ---
        checkCudaErrors(
            cudaMemcpy(h_C_hybrid_out, h_C_initial, m * ldc * sizeof(float), cudaMemcpyHostToHost));
        cudaStream_t stream_hybrid_vc =
            0; // Using default stream as per original test structure for hybrid
        // cudaStreamCreate(&stream_hybrid_vc); // Or create a new stream
        sgemm_strassen_hybrid_2level_vC(m,
                                        n,
                                        k,
                                        alpha_val,
                                        h_A,
                                        lda,
                                        h_B,
                                        ldb,
                                        beta_val,
                                        h_C_hybrid_out,
                                        ldc,
                                        cublas_handle_main, // Use main handle
                                        stream_hybrid_vc);
        // if (stream_hybrid_vc != 0) cudaStreamDestroy(stream_hybrid_vc);

        // --- Comparison ---
        // Epsilon might need to be slightly larger for Strassen due to different operation order
        cmp_result result =
            compare_mats(h_C_ref_gpu_out, h_C_hybrid_out, m * ldc, 1e-3f, false, true);
        std::string result_string{};
        if (!result.equal) {
            hybrid_n_failed += 1;
            result_string = "FAILED";
        } else {
            result_string = "PASSED";
        }
        printf("Hybrid Strassen Test ID=%s | %s\n", cfg.id.c_str(), result_string.c_str());
        hybrid_test_full_info += "Test ID=" + cfg.id + " M=" + std::to_string(m)
                                 + " N=" + std::to_string(n) + " K=" + std::to_string(k)
                                 + " beta=" + std::to_string(beta_val) + " -> " + result_string
                                 + "; Details: " + result.debug_info + "\n";

        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C_initial));
        checkCudaErrors(cudaFreeHost(h_C_ref_gpu_out));
        checkCudaErrors(cudaFreeHost(h_C_hybrid_out));
    }

    hybrid_test_summary += "PASSED: " + std::to_string(hybrid_n_tests - hybrid_n_failed) + " / "
                           + std::to_string(hybrid_n_tests) + "\n";
    if (hybrid_n_failed > 0) { hybrid_test_summary += "SOME HYBRID STRASSEN TESTS FAILED.\n"; }
    printf("%s", hybrid_test_summary.c_str());

    test_full_info += hybrid_test_full_info;
    test_full_info += hybrid_test_summary;
    */

    // Now write all combined results to file
    // test_results_file << test_full_info.c_str(); // This line was writing combined info.
    // For Strassen-only, we'll keep it as is but test_full_info won't have hybrid part.
    test_results_file.close(); // Close file after all tests (sgemm, strassen) are done.
    printf("All test results stored in %s\n", store_path.c_str());

    fflush(stdout);            // Ensure all printf output is flushed
    CUBLAS_CHECK(cublasDestroy(cublas_handle_main)); // Destroy main handle

    return 0;
}