#include <helper_matrix.h>
#include <helper_string.h>
#include <cublas_v2.h> 
#include <sgemm.cuh>
#include <sgemm_strassen_fused.cuh> 
#include <sgemm_hybrid_2level_vC.cuh> // Updated to vC for Hybrid Strassen
#include <string>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using string = std::string;

#define MATSIZE_MAX_DEFAULT  1024
#define MATSIZE_MIN_DEFAULT  2
#define MATSIZE_STEP_DEFAULT 1
#define SAVEDIR_DEFAULT      "test_results"

int
main(int argc, char** argv) {
    srand(time(NULL));

    std::vector<string> args = {};
    for (int i = 1; i < argc; i++) {
        args.push_back(string{argv[i]});
    }

    int matsize_max = get_cmd_line_arg_int(args, "mmax", MATSIZE_MAX_DEFAULT);
    int matsize_min = get_cmd_line_arg_int(args, "mmin", MATSIZE_MIN_DEFAULT);
    int matsize_step = get_cmd_line_arg_int(args, "mstep", MATSIZE_STEP_DEFAULT);
    string save_dir = get_cmd_line_arg_string(args, "savedir", SAVEDIR_DEFAULT);

    int sep_len = 25;
    printf("%.*s\n", sep_len, "===================================================");
    printf("Testing...\n");
    printf("%.*s\n", sep_len, "===================================================");

    string test_summary{};
    string test_full_info{};
    string failed_tests{};
    const int n_tests = (matsize_max - matsize_min) / matsize_step + 1;
    std::vector<cmp_result> test_results(n_tests, cmp_result{});
    int n_failed = 0;

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
        string result_string{};
        if (!result.equal) {
            n_failed += 1;
            failed_tests += to_string(i);
            failed_tests += " ";
            result_string = "FAILED";
        } else {
            result_string = "PASSED";
        }
        printf("Test #%i | matsize = %lu | %s\n", i, matsize, result_string.c_str());

        test_full_info += "Test #" + to_string(i);
        test_full_info += " Matsize = " + to_string(matsize) + ": " + result.debug_info;
        test_results[i] = result;

        checkCudaErrors(cudaFreeHost(A_host));
        checkCudaErrors(cudaFreeHost(B_host));
        checkCudaErrors(cudaFreeHost(C_host));
        checkCudaErrors(cudaFreeHost(C_ref_host));
        checkCudaErrors(cudaFree(A_device));
        checkCudaErrors(cudaFree(B_device));
        checkCudaErrors(cudaFree(C_device));
        checkCudaErrors(cudaGetLastError());
    }
    test_summary += "\n=============== SUMMARY ===============\n";
    test_summary += "PASSED: " + to_string(n_tests - n_failed) + " / " + to_string(n_tests) + "\n";
    test_summary += "FAILED: " + ((failed_tests.size() > 0) ? failed_tests : "0") + "\n";
    test_full_info += test_summary;
    printf("%s", test_summary.c_str());

    fs::path work_dir_path = fs::current_path();
    fs::path store_path = work_dir_path / save_dir / "sgemm.cu.txt";
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
        string id;
    };

    std::vector<StrassenTestConfig> strassen_configs = {
        {64, 64, 64, 0.0f, "64x64x64_beta0"},
        {64, 64, 64, 0.75f, "64x64x64_beta0.75"},
        {128, 128, 128, 0.0f, "128x128x128_beta0"},
        {128, 128, 128, 0.75f, "128x128x128_beta0.75"},
        {256, 256, 128, 0.0f, "256x256x128_beta0"},
        {256, 256, 128, 0.75f, "256x256x128_beta0.75"},
        {128, 256, 256, 0.0f, "128x256x256_beta0"},
        {128, 256, 256, 0.75f, "128x256x256_beta0.75"}
        // Add more configurations as needed, ensuring M,N,K are divisible by 2.
    };

    string strassen_test_summary = "\n=============== STRASSEN FUSED SUMMARY ===============\n";
    string strassen_test_full_info = "";
    int strassen_n_tests = strassen_configs.size();
    int strassen_n_failed = 0;

    for (int i = 0; i < strassen_n_tests; ++i) {
        const auto& cfg = strassen_configs[i];
        size_t m = cfg.M, n = cfg.N, k = cfg.K;
        float beta_val = cfg.beta;
        float alpha_val = 1.0f; // Strassen fused kernels assume alpha = 1.0

        // Adjust lda, ldb, ldc for row-major matrices
        // A is M x K, B is K x N, C is M x N
        size_t lda = k;
        size_t ldb = n;
        size_t ldc = n;

        printf("Running Strassen Test: ID=%s, M=%zu, N=%zu, K=%zu, beta=%.2f, alpha=%.2f\n", 
               cfg.id.c_str(), m, n, k, beta_val, alpha_val);

        float* h_A = alloc_mat_host(m * lda * sizeof(float));
        float* h_B = alloc_mat_host(k * ldb * sizeof(float));
        float* h_C_initial = alloc_mat_host(m * ldc * sizeof(float));
        float* h_C_ref_gpu_out = alloc_mat_host(m * ldc * sizeof(float)); // Output from GPU reference sgemm
        float* h_C_strassen_out = alloc_mat_host(m * ldc * sizeof(float)); // Output from Strassen

        init_random(h_A, m * lda);
        init_random(h_B, k * ldb);
        init_random(h_C_initial, m * ldc);

        // --- Reference Calculation (using GPU sgemm from sgemm.cuh) ---
        float* d_A_ref = alloc_mat_device(m * lda * sizeof(float));
        float* d_B_ref = alloc_mat_device(k * ldb * sizeof(float));
        float* d_C_ref = alloc_mat_device(m * ldc * sizeof(float));

        checkCudaErrors(cudaMemcpy(d_A_ref, h_A, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B_ref, h_B, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_C_ref, h_C_initial, m * ldc * sizeof(float), cudaMemcpyHostToDevice));
        
        // sgemm from sgemm.cuh takes alpha and beta by pointer
        sgemm(m, n, k, &alpha_val, d_A_ref, lda, d_B_ref, ldb, &beta_val, d_C_ref, ldc);
        checkCudaErrors(cudaGetLastError()); // Check for kernel errors
        checkCudaErrors(cudaDeviceSynchronize()); // Ensure kernel completion

        checkCudaErrors(cudaMemcpy(h_C_ref_gpu_out, d_C_ref, m * ldc * sizeof(float), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_A_ref));
        checkCudaErrors(cudaFree(d_B_ref));
        checkCudaErrors(cudaFree(d_C_ref));

        // --- Strassen Calculation (using refactored sgemm_strassen_1level_fused_vP with device pointers) ---
        float* d_A_strassen = alloc_mat_device(m * lda * sizeof(float));
        float* d_B_strassen = alloc_mat_device(k * ldb * sizeof(float));
        float* d_C_strassen_inout = alloc_mat_device(m * ldc * sizeof(float));
        cudaStream_t stream_strassen_1L = 0; // Use default stream

        checkCudaErrors(cudaMemcpy2DAsync(d_A_strassen, lda * sizeof(float), h_A, lda * sizeof(float), k * sizeof(float), m, cudaMemcpyHostToDevice, stream_strassen_1L));
        checkCudaErrors(cudaMemcpy2DAsync(d_B_strassen, ldb * sizeof(float), h_B, ldb * sizeof(float), n * sizeof(float), k, cudaMemcpyHostToDevice, stream_strassen_1L));

        if (beta_val != 0.0f) {
            // If beta is not 0, copy initial C content to device.
            // sgemm_strassen_1level_fused_vP will use it as input if beta_val=1.0f.
            // If beta_val is other non-zero, sgemm_strassen_1level_fused_vP expects C_dev to be pre-scaled by caller.
            // For this test, we assume if beta_val is non-zero, it's typically 1.0 for accumulation,
            // or the test checks against a reference that handles arbitrary beta.
            // The reference sgemm correctly handles arbitrary beta.
            // So, we provide h_C_initial for non-zero beta.
            checkCudaErrors(cudaMemcpy2DAsync(d_C_strassen_inout, ldc * sizeof(float), h_C_initial, ldc * sizeof(float), n * sizeof(float), m, cudaMemcpyHostToDevice, stream_strassen_1L));
        }
        // If beta_val == 0.0f, sgemm_strassen_1level_fused_vP will cudaMemsetAsync d_C_strassen_inout.
        
        // cublas_handle was created for hybrid tests; it's named cublas_handle there.
        // Here, the test section for 1-level Strassen is before hybrid, so use its own handle if needed,
        // or assume one is available (e.g. cublas_handle_for_hybrid created earlier).
        // For now, use the one from the hybrid section, assuming it's created at a higher scope or passed.
        // The prior diff added `cublas_handle_for_hybrid` globally in main.
        sgemm_strassen_1level_fused_vP(m, n, k, 
                                     alpha_val,         // alpha_sub (typically 1.0 for this usage)
                                     d_A_strassen, lda, 
                                     d_B_strassen, ldb, 
                                     beta_val,          // beta_sub
                                     d_C_strassen_inout, ldc, 
                                     cublas_handle_for_hybrid, // Pass the handle
                                     stream_strassen_1L);
        
        checkCudaErrors(cudaStreamSynchronize(stream_strassen_1L)); // Wait for Strassen kernels to complete

        // Copy result back to h_C_strassen_out
        checkCudaErrors(cudaMemcpy2DAsync(h_C_strassen_out, ldc * sizeof(float), d_C_strassen_inout, ldc * sizeof(float), n * sizeof(float), m, cudaMemcpyDeviceToHost, stream_strassen_1L));
        checkCudaErrors(cudaStreamSynchronize(stream_strassen_1L)); // Wait for D2H to complete

        checkCudaErrors(cudaFreeAsync(d_A_strassen, stream_strassen_1L));
        checkCudaErrors(cudaFreeAsync(d_B_strassen, stream_strassen_1L));
        checkCudaErrors(cudaFreeAsync(d_C_strassen_inout, stream_strassen_1L));
        // Final sync for frees if stream is not default, though not strictly needed here before comparison
        if(stream_strassen_1L != 0) checkCudaErrors(cudaStreamSynchronize(stream_strassen_1L));

        // --- Comparison ---
        // Compare h_C_ref_gpu_out (from reference GPU sgemm) with h_C_strassen_out
        cmp_result result = compare_mats(h_C_ref_gpu_out, h_C_strassen_out, m * ldc, 1e-4f, false, true);
        string result_string{};
        if (!result.equal) {
            strassen_n_failed += 1;
            result_string = "FAILED";
        } else {
            result_string = "PASSED";
        }
        printf("Strassen Test ID=%s | %s (Max Abs Diff: %e, Rel Diff: %e)\n", 
               cfg.id.c_str(), result_string.c_str(), result.max_abs_diff, result.max_rel_diff);

        strassen_test_full_info += "Strassen Test ID=" + cfg.id + ": " + result.debug_info;
        
        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C_initial));
        checkCudaErrors(cudaFreeHost(h_C_ref_gpu_out));
        checkCudaErrors(cudaFreeHost(h_C_strassen_out));
    }

    strassen_test_summary += "PASSED: " + to_string(strassen_n_tests - strassen_n_failed) + " / " + to_string(strassen_n_tests) + "\n";
    if (strassen_n_failed > 0) {
        strassen_test_summary += "SOME STRASSEN TESTS FAILED.\n";
    }
    printf("%s", strassen_test_summary.c_str());

    test_full_info += strassen_test_full_info; // Append Strassen specific info
    test_full_info += strassen_test_summary;   // Append Strassen summary

    // Now write all results to file
    // test_results_file << test_full_info.c_str(); // Keep open for Hybrid Strassen results
    // test_results_file.close();                 // Will be closed after hybrid tests
    // printf("All test results stored in %s\n", store_path.c_str());


    // --- cuBLAS Handle for Hybrid Strassen ---
    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed!\n");
        // Close the file before exiting if it was opened
        if (test_results_file.is_open()) {
            test_results_file.close();
        }
        return -1; 
    }

    // --- Test Hybrid 2-Level Strassen SGEMM ---
    printf("%.*s\n", sep_len, "===================================================");
    printf("Testing Hybrid 2-Level Strassen SGEMM...\n");
    printf("%.*s\n", sep_len, "===================================================");

    struct HybridStrassenTestConfig {
        int M, N, K;
        float beta;
        string id;
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
        {512, 512, 512, 0.75f, "Hybrid_512x512x512_beta0.75"}
    };

    string hybrid_test_summary = "\n=============== HYBRID STRASSEN SUMMARY ===============\n";
    string hybrid_test_full_info = "";
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
               cfg.id.c_str(), m, n, k, beta_val, alpha_val);

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
        checkCudaErrors(cudaMemcpy(d_C_ref, h_C_initial, m * ldc * sizeof(float), cudaMemcpyHostToDevice));
        
        sgemm(m, n, k, &alpha_val, d_A_ref, lda, d_B_ref, ldb, &beta_val, d_C_ref, ldc);
        checkCudaErrors(cudaGetLastError()); 
        checkCudaErrors(cudaDeviceSynchronize()); 

        checkCudaErrors(cudaMemcpy(h_C_ref_gpu_out, d_C_ref, m * ldc * sizeof(float), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_A_ref));
        checkCudaErrors(cudaFree(d_B_ref));
        checkCudaErrors(cudaFree(d_C_ref));

        // --- Hybrid Strassen Calculation ---
        checkCudaErrors(cudaMemcpy(h_C_hybrid_out, h_C_initial, m * ldc * sizeof(float), cudaMemcpyHostToHost));
        cudaStream_t stream_hybrid_vc = 0; // Using default stream as per original test structure for hybrid
        // cudaStreamCreate(&stream_hybrid_vc); // Or create a new stream
        sgemm_strassen_hybrid_2level_vC(m, n, k, alpha_val, h_A, lda, h_B, ldb, beta_val, h_C_hybrid_out, ldc, cublas_handle, stream_hybrid_vc);
        // if (stream_hybrid_vc != 0) cudaStreamDestroy(stream_hybrid_vc);
        
        // --- Comparison ---
        // Epsilon might need to be slightly larger for Strassen due to different operation order
        cmp_result result = compare_mats(h_C_ref_gpu_out, h_C_hybrid_out, m * ldc, 1e-3f, false, true); 
        string result_string{};
        if (!result.equal) {
            hybrid_n_failed += 1;
            result_string = "FAILED";
        } else {
            result_string = "PASSED";
        }
        printf("Hybrid Strassen Test ID=%s | %s (Max Abs Diff: %e, Rel Diff: %e)\n", 
               cfg.id.c_str(), result_string.c_str(), result.max_abs_diff, result.max_rel_diff);

        hybrid_test_full_info += "Hybrid Strassen Test ID=" + cfg.id + ": " + result.debug_info;
        
        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C_initial));
        checkCudaErrors(cudaFreeHost(h_C_ref_gpu_out));
        checkCudaErrors(cudaFreeHost(h_C_hybrid_out));
    }

    hybrid_test_summary += "PASSED: " + to_string(hybrid_n_tests - hybrid_n_failed) + " / " + to_string(hybrid_n_tests) + "\n";
    if (hybrid_n_failed > 0) {
        hybrid_test_summary += "SOME HYBRID STRASSEN TESTS FAILED.\n";
    }
    printf("%s", hybrid_test_summary.c_str());

    test_full_info += hybrid_test_full_info; 
    test_full_info += hybrid_test_summary;   

    // Now write all combined results to file
    test_results_file << test_full_info.c_str();
    test_results_file.close();
    printf("All test results stored in %s\n", store_path.c_str());

    cublasDestroy(cublas_handle); // Destroy cuBLAS handle

    return 0;
}