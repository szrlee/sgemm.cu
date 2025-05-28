#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <helper_matrix.h>
#include <helper_string.h>
#include <sgemm.cuh>
#include <sgemm_strassen_fused.cuh> // For Strassen vP
#include <string>
#include <vector>

#include <fstream>

#define MATSIZE_MIN_DEFAULT    512  // Changed for Strassen benchmark range
#define MATSIZE_STEP_DEFAULT   256  // Changed for more points
#define NPTS_DEFAULT           15   // (4096-512)/256 + 1 = 15 points to reach 4096
#define WARMUP_MATSIZE_DEFAULT 2048 // Adjusted warmup size
#define WARMUP_NITER_DEFAULT   10   // Reduced warmup iterations for faster testing cycle
#define MATSIZE_MAX_DEFAULT    4096 // Added definition
#define SAVEDIR_DEFAULT        "benchmark_results"
#define FILE_NAME_DEFAULT      "sgemm.cu"

int
main(int argc, char** argv) {

    std::vector<std::string> args = {};
    for (int i = 1; i < argc; i++) {
        args.push_back(std::string{argv[i]});
    }

    // Ensure cuBLAS handle is always created for Hybrid Strassen and refactored 1-Level Strassen
    cublasHandle_t cublas_handle_main; // Renamed for general use
    checkCudaErrors(cublasCreate(&cublas_handle_main));

    int matsize_min = get_cmd_line_arg_int(args, "mmin", MATSIZE_MIN_DEFAULT);
    int matsize_step = get_cmd_line_arg_int(args, "mstep", MATSIZE_STEP_DEFAULT);
    int npts = get_cmd_line_arg_int(args, "npts", NPTS_DEFAULT);
    int warmup_matsize = get_cmd_line_arg_int(args, "wmsize", WARMUP_MATSIZE_DEFAULT);
    int warmup_niter = get_cmd_line_arg_int(args, "wniter", WARMUP_NITER_DEFAULT);
    std::string save_dir = get_cmd_line_arg_string(args, "savedir", SAVEDIR_DEFAULT);
#if CUBLAS == 1
    std::string file_name = "cuBLAS";
#else
    std::string file_name = get_cmd_line_arg_string(args, "fname", FILE_NAME_DEFAULT);
#endif
    int sep_len = 25;

#if CUBLAS == 1
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));
#endif
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (warmup_niter > 0) {
        printf("%.*s\n", sep_len, "===================================================");
        printf("Warm-up\n");
        printf("%.*s\n", sep_len, "===================================================");

        int m = warmup_matsize, n = warmup_matsize, k = warmup_matsize;
        int lda = k, ldb = n, ldc = n;

        float* A_host = alloc_mat_host(m * lda * sizeof(float));
        float* B_host = alloc_mat_host(k * ldb * sizeof(float));
        float* C_host = alloc_mat_host(m * ldc * sizeof(float));

        float* A_device = alloc_mat_device(m * lda * sizeof(float));
        float* B_device = alloc_mat_device(k * ldb * sizeof(float));
        float* C_device = alloc_mat_device(m * ldc * sizeof(float));

        init_random(A_host, m * lda);
        init_random(B_host, k * ldb);
        init_random(C_host, m * ldc);
        checkCudaErrors(
            cudaMemcpy(A_device, A_host, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(B_device, B_host, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(C_device, C_host, m * ldc * sizeof(float), cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup_niter; i++) {
#if CUBLAS == 1
            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        B_device,
                        ldb,
                        A_device,
                        lda,
                        &beta,
                        C_device,
                        ldc);
#else
            sgemm(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta, C_device, ldc);
#endif
            cudaDeviceSynchronize();
            fflush(stdout);
            printf("\r%i / %u", i + 1, warmup_niter);
        }
        printf("\n\n");
        checkCudaErrors(cudaFreeHost(A_host));
        checkCudaErrors(cudaFreeHost(B_host));
        checkCudaErrors(cudaFreeHost(C_host));
        checkCudaErrors(cudaFree(A_device));
        checkCudaErrors(cudaFree(B_device));
        checkCudaErrors(cudaFree(C_device));
    }

    std::vector<double> sgemm_avg_times(npts, 0.0);
    std::vector<double> strassen_avg_times(npts, 0.0);
    std::vector<int> matsizes(npts, 0);
    const int NUM_ITERATIONS = 10; // Number of iterations for timing loop

    // Output file for benchmark results
    std::filesystem::path work_dir_path = std::filesystem::current_path();
    std::filesystem::path store_benchmark_path = work_dir_path / save_dir
                                                 / "all_sgemm_benchmarks_final.txt";
    std::ofstream benchmark_file(store_benchmark_path);
    benchmark_file << "M,N,K,SGEMM_Time_ms,Strassen_1L_vP_GPU_Time_ms,Speedup_1L_GPU\n";


    printf("%.*s\n", sep_len, "===================================================");
    printf("Benchmark: SGEMM vs Strassen 1-Level (vP, GPU-centric)\n");
    printf("Alpha: %.1f, Beta: %.1f, Iterations per size: %d\n", alpha, beta, NUM_ITERATIONS);
    printf("%.*s\n", sep_len, "===================================================");

    for (int i = 0; i < npts; i++) {
        int matsize = matsize_min + i * matsize_step;
        if (matsize > MATSIZE_MAX_DEFAULT
            && npts == NPTS_DEFAULT) { // Cap at default max if using default npts
            matsize = MATSIZE_MAX_DEFAULT;
            if (i > 0 && matsizes[i - 1] == MATSIZE_MAX_DEFAULT) break; // Avoid duplicate max size
        }
        if (matsize == 0) continue; // Should not happen with new defaults

        matsizes[i] = matsize;
        int m = matsize, n = matsize, k = matsize;
        int lda = k, ldb = n, ldc = n;

        printf("Benchmarking Size: M=%d, N=%d, K=%d\n", m, n, k);

        float* h_A = alloc_mat_host(m * lda * sizeof(float));
        float* h_B = alloc_mat_host(k * ldb * sizeof(float));
        float* h_C_for_strassen_1L_dummy = alloc_mat_host(
            m * ldc * sizeof(float)); // Not strictly needed if not verifying output

        float* d_C_ref = alloc_mat_device(m * ldc * sizeof(float));  // For sgemm output
        float* d_A_main = alloc_mat_device(m * lda * sizeof(float)); // Renamed from d_A
        float* d_B_main = alloc_mat_device(k * ldb * sizeof(float)); // Renamed from d_B
        float* d_C_1L_out = alloc_mat_device(m * ldc
                                             * sizeof(float)); // For 1-Level Strassen GPU output

        init_random(h_A, m * lda);
        init_random(h_B, k * ldb);

        // Copy A & B to device once for sgemm and 1-Level Strassen (GPU-centric)
        checkCudaErrors(cudaMemcpy(d_A_main, h_A, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B_main, h_B, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        // For beta=0, d_C_ref and d_C_1L_out do not need pre-initialization.

        cudaEvent_t start_event, stop_event;
        checkCudaErrors(cudaEventCreate(&start_event));
        checkCudaErrors(cudaEventCreate(&stop_event));
        float elapsed_time_ms;

        // --- Benchmark Standard sgemm (from sgemm.cuh) ---
#if CUBLAS == 1
        // Warm-up for cuBLAS
        cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha,
                    d_B_main,
                    ldb,
                    d_A_main,
                    lda,
                    &beta,
                    d_C_ref,
                    ldc);
#else
        // Warm-up for sgemm
        sgemm(m, n, k, &alpha, d_A_main, lda, d_B_main, ldb, &beta, d_C_ref, ldc);
#endif
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaEventRecord(start_event, 0));
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
#if CUBLAS == 1
            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        d_B_main,
                        ldb,
                        d_A_main,
                        lda,
                        &beta,
                        d_C_ref,
                        ldc);
#else
            sgemm(m, n, k, &alpha, d_A_main, lda, d_B_main, ldb, &beta, d_C_ref, ldc);
#endif
        }
        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));
        checkCudaErrors(cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event));
        sgemm_avg_times[i] = elapsed_time_ms / NUM_ITERATIONS;

        // --- Benchmark Strassen 1-Level sgemm_strassen_1level_fused_vP (GPU-centric) ---
        // Warm-up (operates on device pointers d_A_main, d_B_main, d_C_1L_out)
        // beta is 0.0f, so d_C_1L_out will be zeroed by the function.
        sgemm_strassen_1level_fused_vP(m,
                                       n,
                                       k,
                                       alpha,
                                       d_A_main,
                                       lda,
                                       d_B_main,
                                       ldb,
                                       beta,
                                       d_C_1L_out,
                                       ldc,
                                       cublas_handle_main,
                                       0,      // stream 0
                                       false); // Added debug flag
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaEventRecord(start_event, 0));
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            sgemm_strassen_1level_fused_vP(m,
                                           n,
                                           k,
                                           alpha,
                                           d_A_main,
                                           lda,
                                           d_B_main,
                                           ldb,
                                           beta,
                                           d_C_1L_out,
                                           ldc,
                                           cublas_handle_main,
                                           0,      // stream 0
                                           false); // Added debug flag
        }
        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));
        checkCudaErrors(cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event));
        strassen_avg_times[i] = elapsed_time_ms / NUM_ITERATIONS; // This is now Strassen_1L_vP_GPU


        checkCudaErrors(cudaEventDestroy(start_event));
        checkCudaErrors(cudaEventDestroy(stop_event));

        double speedup_1L_gpu = strassen_avg_times[i] > 0 ?
                                    (sgemm_avg_times[i] / strassen_avg_times[i]) :
                                    0;

        printf(
            "Size: %dx%dx%d, SGEMM: %.3f ms, Strassen_1L_vP_GPU: %.3f ms, Speedup_1L_GPU: %.2fx\n",
            m,
            n,
            k,
            sgemm_avg_times[i],
            strassen_avg_times[i],
            speedup_1L_gpu);
        benchmark_file << m << "," << n << "," << k << "," << sgemm_avg_times[i] << ","
                       << strassen_avg_times[i] << "," << speedup_1L_gpu << "\n";

        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C_for_strassen_1L_dummy)); // Was h_C_for_strassen
        checkCudaErrors(cudaFree(d_A_main));                      // Was d_A
        checkCudaErrors(cudaFree(d_B_main));                      // Was d_B
        checkCudaErrors(cudaFree(d_C_ref));
        checkCudaErrors(cudaFree(d_C_1L_out)); // Free new device buffer for 1L Strassen

        if (matsize == MATSIZE_MAX_DEFAULT && npts == NPTS_DEFAULT)
            break;                             // ensure loop terminates if max reached early
    }

    benchmark_file.close();
    printf("Benchmark data stored in %s\n", store_benchmark_path.c_str());
    printf("%.*s\n", sep_len, "===================================================");

#if CUBLAS == 1
    // If the main CUBLAS handle 'handle' was used by sgemm (if CUBLAS==1), destroy it.
    // The cublas_handle_main is used by Strassen versions.
    checkCudaErrors(cublasDestroy(handle));
#endif
    checkCudaErrors(cublasDestroy(cublas_handle_main)); // Destroy the main handle

    return 0;
}