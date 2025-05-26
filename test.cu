#include <helper_matrix.h>
#include <helper_string.h>
#include <sgemm.cuh>
#include <sgemm_strassen_fused.cuh> // Added for Strassen fused
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

        // --- Strassen Calculation ---
        // sgemm_strassen_1level_fused takes alpha and beta by value
        // It needs h_C_initial as input, and writes to h_C_strassen_out
        checkCudaErrors(cudaMemcpy(h_C_strassen_out, h_C_initial, m * ldc * sizeof(float), cudaMemcpyHostToHost)); // Prepare input C for Strassen
        sgemm_strassen_1level_fused(m, n, k, alpha_val, h_A, lda, h_B, ldb, beta_val, h_C_strassen_out, ldc);
        // sgemm_strassen_1level_fused already calls cudaDeviceSynchronize where needed internally

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
    test_results_file << test_full_info.c_str();
    test_results_file.close();
    printf("All test results stored in %s\n", store_path.c_str());

    return 0;
}