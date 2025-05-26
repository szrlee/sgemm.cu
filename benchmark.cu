#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_matrix.h>
#include <helper_string.h>
#include <sgemm.cuh>
#include <sgemm_strassen_fused.cuh> // For Strassen vP

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using string = std::string;

#define MATSIZE_MIN_DEFAULT    512  // Changed for Strassen benchmark range
#define MATSIZE_STEP_DEFAULT   256  // Changed for more points
#define NPTS_DEFAULT           15   // (4096-512)/256 + 1 = 15 points to reach 4096
#define WARMUP_MATSIZE_DEFAULT 2048 // Adjusted warmup size
#define WARMUP_NITER_DEFAULT   10   // Reduced warmup iterations for faster testing cycle
#define SAVEDIR_DEFAULT        "benchmark_results"
#define FILE_NAME_DEFAULT      "sgemm.cu"

int
main(int argc, char** argv) {

    std::vector<string> args = {};
    for (int i = 1; i < argc; i++) {
        args.push_back(string{argv[i]});
    }

    int matsize_min = get_cmd_line_arg_int(args, "mmin", MATSIZE_MIN_DEFAULT);
    int matsize_step = get_cmd_line_arg_int(args, "mstep", MATSIZE_STEP_DEFAULT);
    int npts = get_cmd_line_arg_int(args, "npts", NPTS_DEFAULT);
    int warmup_matsize = get_cmd_line_arg_int(args, "wmsize", WARMUP_MATSIZE_DEFAULT);
    int warmup_niter = get_cmd_line_arg_int(args, "wniter", WARMUP_NITER_DEFAULT);
    string save_dir = get_cmd_line_arg_string(args, "savedir", SAVEDIR_DEFAULT);
#if CUBLAS == 1
    string file_name = "cuBLAS";
#else
    string file_name = get_cmd_line_arg_string(args, "fname", FILE_NAME_DEFAULT);
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
    fs::path work_dir_path = fs::current_path();
    fs::path store_benchmark_path = work_dir_path / save_dir / "strassen_vs_sgemm_benchmark.txt";
    std::ofstream benchmark_file(store_benchmark_path);
    benchmark_file << "M,N,K,SGEMM_Time_ms,Strassen_Time_ms,Speedup\n";


    printf("%.*s\n", sep_len, "===================================================");
    printf("Benchmark: Standard SGEMM vs Strassen Fused SGEMM\n");
    printf("Alpha: %.1f, Beta: %.1f, Iterations per size: %d\n", alpha, beta, NUM_ITERATIONS);
    printf("%.*s\n", sep_len, "===================================================");

    for (int i = 0; i < npts; i++) {
        int matsize = matsize_min + i * matsize_step;
        if (matsize > MATSIZE_MAX_DEFAULT && npts == NPTS_DEFAULT) { // Cap at default max if using default npts
             matsize = MATSIZE_MAX_DEFAULT;
             if (i > 0 && matsizes[i-1] == MATSIZE_MAX_DEFAULT) break; // Avoid duplicate max size
        }
        if (matsize == 0) continue; // Should not happen with new defaults

        matsizes[i] = matsize;
        int m = matsize, n = matsize, k = matsize;
        int lda = k, ldb = n, ldc = n;

        printf("Benchmarking Size: M=%d, N=%d, K=%d\n", m, n, k);

        float* h_A = alloc_mat_host(m * lda * sizeof(float));
        float* h_B = alloc_mat_host(k * ldb * sizeof(float));
        // h_C_for_strassen is used as output for Strassen. Beta=0, so its initial content doesn't matter.
        float* h_C_for_strassen = alloc_mat_host(m * ldc * sizeof(float)); 
        // d_C_ref is used as output for reference sgemm.
        float* d_C_ref = alloc_mat_device(m * ldc * sizeof(float));


        float* d_A = alloc_mat_device(m * lda * sizeof(float));
        float* d_B = alloc_mat_device(k * ldb * sizeof(float));
        
        init_random(h_A, m * lda);
        init_random(h_B, k * ldb);
        // No need to init h_C_for_strassen as beta=0.0f for Strassen call here.
        // No need to init or copy d_C_ref as beta=0.0f for sgemm call. Kernels should handle output only.

        checkCudaErrors(cudaMemcpy(d_A, h_A, m * lda * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, k * ldb * sizeof(float), cudaMemcpyHostToDevice));
        // For beta=0, d_C_ref does not need to be copied from host. It will be overwritten.
        // For Strassen, h_C_for_strassen is an output parameter. Its input state is handled by sgemm_strassen_1level_fused_vP based on beta.

        cudaEvent_t start_event, stop_event;
        checkCudaErrors(cudaEventCreate(&start_event));
        checkCudaErrors(cudaEventCreate(&stop_event));
        float elapsed_time_ms;

        // --- Benchmark Standard sgemm (from sgemm.cuh) ---
#if CUBLAS == 1
        // Warm-up for cuBLAS
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, ldb, d_A, lda, &beta, d_C_ref, ldc);
#else
        // Warm-up for sgemm
        sgemm(m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C_ref, ldc);
#endif
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaEventRecord(start_event, 0));
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
#if CUBLAS == 1
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, ldb, d_A, lda, &beta, d_C_ref, ldc);
#else
            sgemm(m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C_ref, ldc);
#endif
        }
        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));
        checkCudaErrors(cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event));
        sgemm_avg_times[i] = elapsed_time_ms / NUM_ITERATIONS;

        // --- Benchmark Strassen sgemm_strassen_1level_fused_vP ---
        // Warm-up for Strassen (takes host pointers)
        // h_C_for_strassen will be overwritten. Its input content for beta=0 is handled by the launcher.
        sgemm_strassen_1level_fused_vP(m, n, k, alpha, h_A, lda, h_B, ldb, beta, h_C_for_strassen, ldc, 0);
        checkCudaErrors(cudaDeviceSynchronize()); // Ensure warmup is complete as Strassen vP has its own sync

        checkCudaErrors(cudaEventRecord(start_event, 0));
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            sgemm_strassen_1level_fused_vP(m, n, k, alpha, h_A, lda, h_B, ldb, beta, h_C_for_strassen, ldc, 0);
        }
        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));
        checkCudaErrors(cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event));
        strassen_avg_times[i] = elapsed_time_ms / NUM_ITERATIONS;
        
        checkCudaErrors(cudaEventDestroy(start_event));
        checkCudaErrors(cudaEventDestroy(stop_event));

        printf("Size: %dx%dx%d, SGEMM: %.3f ms, Strassen_vP: %.3f ms, Speedup: %.2fx\n",
               m, n, k, sgemm_avg_times[i], strassen_avg_times[i], sgemm_avg_times[i] / strassen_avg_times[i]);
        benchmark_file << m << "," << n << "," << k << ","
                       << sgemm_avg_times[i] << "," << strassen_avg_times[i] << ","
                       << (strassen_avg_times[i] > 0 ? (sgemm_avg_times[i] / strassen_avg_times[i]) : 0) << "\n";
        
        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C_for_strassen));
        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C_ref));
        
        if (matsize == MATSIZE_MAX_DEFAULT && npts == NPTS_DEFAULT) break; // ensure loop terminates if max reached early
    }
    
    benchmark_file.close();
    printf("Benchmark data stored in %s\n", store_benchmark_path.c_str());
    printf("%.*s\n", sep_len, "===================================================");

#if CUBLAS == 1
    checkCudaErrors(cublasDestroy(handle));
#endif
    return 0;
}