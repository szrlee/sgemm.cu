#ifndef STRASSEN_FUSED_128X128X8_KERNEL_SNIPPET_CUH_
#define STRASSEN_FUSED_128X128X8_KERNEL_SNIPPET_CUH_

#include "../../common/helper_cuda_ptx.h" // Corrected path and filename
#include <cstdint>

constexpr int STRASSEN_TILE_M = 128;
constexpr int STRASSEN_TILE_N = 128;
constexpr int STRASSEN_TILE_K = 8;

constexpr int SMEM_A_LD_FUSED = 132;
constexpr int SMEM_B_LD_FUSED = 128;

constexpr int SMEM_A_SINGLE_BUFFER_SIZE_FLOATS = STRASSEN_TILE_K * SMEM_A_LD_FUSED;
constexpr int SMEM_B_SINGLE_BUFFER_SIZE_FLOATS = STRASSEN_TILE_K * SMEM_B_LD_FUSED;

__device__ __forceinline__ float
get_strassen_coeff_val(int val) {
    if (val == 1) return 1.0f;
    if (val == -1) return -1.0f;
    return 0.0f;
}

template <int DELTA_VAL, int EPSILON_VAL, int GAMMA0_VAL, int GAMMA1_VAL>
__global__
__launch_bounds__(256, 2) void strassen_fused_128x128x8_kernel(int M_sub,
                                                               int N_sub,
                                                               int K_sub,
                                                               const float host_alpha,
                                                               const float* __restrict__ X_gm,
                                                               int ldx,
                                                               const float* __restrict__ Y_gm,
                                                               int ldy,
                                                               const float* __restrict__ V_gm,
                                                               int ldv,
                                                               const float* __restrict__ W_gm,
                                                               int ldw,
                                                               float* D_gm,
                                                               int ldd,
                                                               float* E_gm,
                                                               int lde,
                                                               bool enable_debug_prints = false) {
    // Debug prints disabled for performance
    // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("[KERNEL ENTRY DEBUG] Kernel launched!\n");
    // }

    extern __shared__ float smem_storage[];
    // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     // Attempt to print the address of smem_storage or its value if it's treated as a pointer
    //     printf("[KERNEL DEBUG SMEM_ADDR_RAW] &smem_storage=%p, (void*)smem_storage=%p\n",
    //            (void*)&smem_storage,
    //            (void*)smem_storage);
    // }

    // Always print which Strassen kernel is running (disabled for performance)
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("[STRASSEN KERNEL] Running with D=%d,E=%d,G0=%d,G1=%d (should identify which M_i)\\n",
    //            DELTA_VAL, EPSILON_VAL, GAMMA0_VAL, GAMMA1_VAL);
    // }

    // Get base shared memory address for smem_storage
    uint64_t smem_base_addr_u64 = 0; // Initialize to 0
    void* smem_ptr_void = (void*)smem_storage;
    smem_base_addr_u64 = (uint64_t)smem_ptr_void;

    // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("[KERNEL DEBUG SMEM_POST_CAST] smem_base_addr_u64 = 0x%llx\n",
    //            (unsigned long long)smem_base_addr_u64);
    // }

    // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("[KERNEL DEBUG SMEM_BASE_CVTA] smem_base_addr_u64 = 0x%llx\n",
    //            (unsigned long long)smem_base_addr_u64);
    // }

    // Define shared memory buffer addresses as uint64_t byte offsets from smem_base_addr_u64
    const uint64_t sm_a_ping_addr_u64 = smem_base_addr_u64;
    const uint64_t sm_a_pong_addr_u64 = sm_a_ping_addr_u64
                                        + SMEM_A_SINGLE_BUFFER_SIZE_FLOATS * sizeof(float);
    const uint64_t sm_b_ping_addr_u64 =
        sm_a_pong_addr_u64
        + SMEM_A_SINGLE_BUFFER_SIZE_FLOATS * sizeof(float); // B_ping starts after A_ping and A_pong
    const uint64_t sm_b_pong_addr_u64 = sm_b_ping_addr_u64
                                        + SMEM_B_SINGLE_BUFFER_SIZE_FLOATS * sizeof(float);
    // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("[KERNEL DEBUG SMEM_CUR_WRITE_INIT] sm_a_ping_addr_u64=0x%llx, "
    //            "current_sm_a_write_addr_u64=0x%llx\n",
    //            (unsigned long long)sm_a_ping_addr_u64,
    //            (unsigned long long)sm_a_pong_addr_u64);
    // }

    float accumulator[8][8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            accumulator[r][c] = 0.0f;
        }
    }

    float ldg_reg_buffer_x[4];
    float ldg_reg_buffer_y[4];

    const bool X_ptr_is_valid = (X_gm != nullptr);
    const bool Y_ptr_is_valid = (Y_gm != nullptr);
    const bool V_ptr_is_valid = (V_gm != nullptr);
    const bool W_ptr_is_valid = (W_gm != nullptr);

    const float delta_coeff = get_strassen_coeff_val(DELTA_VAL);
    const float epsilon_coeff = get_strassen_coeff_val(EPSILON_VAL);
    const float effective_gamma0 = host_alpha * get_strassen_coeff_val(GAMMA0_VAL);
    const float effective_gamma1 = host_alpha * get_strassen_coeff_val(GAMMA1_VAL);

    uint64_t current_sm_a_write_addr_u64 = sm_a_ping_addr_u64;
    uint64_t current_sm_b_write_addr_u64 = sm_b_ping_addr_u64;
    uint64_t current_sm_a_read_addr_u64 =
        sm_a_ping_addr_u64; // Start reading from same buffer initially
    uint64_t current_sm_b_read_addr_u64 =
        sm_b_ping_addr_u64; // Start reading from same buffer initially
    // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("[KERNEL DEBUG SMEM_CUR_WRITE_INIT] sm_a_ping_addr_u64=0x%llx, "
    //            "current_sm_a_write_addr_u64=0x%llx\n",
    //            (unsigned long long)sm_a_ping_addr_u64,
    //            (unsigned long long)current_sm_a_write_addr_u64);
    // }

    // --- X and Y Global Load (Initial Pipe Stage) ---
    int ldg_xy_k_offset_in_tile = threadIdx.x
                                  % STRASSEN_TILE_K; // k-index within the 8-element k-strip
    int ldg_xy_m_strip_id =
        threadIdx.x
        / STRASSEN_TILE_K; // Identifies which 4-m strip this thread handles (0-31 for 128m)
    int ldg_xy_m_base_global = blockIdx.y * STRASSEN_TILE_M + ldg_xy_m_strip_id * 4;


    // --- V and W Global Load (Initial Pipe Stage) ---
    int ldg_vw_k_global_idx =
        threadIdx.x
        / 32; // Each group of 32 threads handles one k-plane for V/W. (0-7 for K_TILE=8)
    int ldg_vw_n_strip_id_within_k_plane = threadIdx.x
                                           % 32; // Within a k-plane, which 4-N strip (0-31)
    // Corrected n_base_global for V/W. Each of the 32 threads loads 4 N-values that are contiguous.
    int ldg_vw_n_base_global = blockIdx.x * STRASSEN_TILE_N + ldg_vw_n_strip_id_within_k_plane * 4;

    // Removed old V/W base pointers and offsets global, they are replaced by direct calculation.
    // const float* ldg_v_ptr_base = V_gm + ldg_vw_k_offset_in_tile * ldv + (blockIdx.x * STRASSEN_TILE_N + (threadIdx.x % 32));
    // const float* ldg_w_ptr_base = W_gm + ldg_vw_k_offset_in_tile * ldw + (blockIdx.x * STRASSEN_TILE_N + (threadIdx.x % 32));
    // int ldg_vw_n_offsets_global[4]; // Removed

    // int sts_a_m_sm_base = (threadIdx.x / STRASSEN_TILE_K) * 4; // Removed, was unused
    // int sts_a_k_sm_offset = threadIdx.x % STRASSEN_TILE_K; // Removed, was unused
    // float* sts_a_ptr_sm = current_sm_a_write_ptr + sts_a_m_sm_base * SMEM_A_LD_FUSED
    //                       + sts_a_k_sm_offset; // Removed, was unused

    // uint64_t sts_a_addr_sm; // Removed
    // CVTA_TO_SHARED_PTX(sts_a_addr_sm, sts_a_ptr_sm); // This line was commented out, so removing the variable is fine

    // Shared memory B (V/W) store parameters - these are from paper's direct mapping, not 8x8 tile output mapping for STS stage.
    // A thread (tidx.x) is responsible for a k_sm index and an n_sm_base index for its 4 N-values.
    // int sts_b_k_sm_idx_for_thread = threadIdx.x / 32; // Removed, was unused // k index in shared B (0-7)
    int sts_b_n_sm_base_for_thread_strip =
        (threadIdx.x % 32)
        * 4; // Base N index in shared B for this thread's 4 N-values (0, 4, ..., 124)

    // uint64_t sts_b_addr_sm; // Defined below before use.
    // CVTA_TO_SHARED_PTX(sts_b_addr_sm, sts_b_ptr_sm); // sts_b_ptr_sm is base of current_sm_b_write_ptr + k_offset * lds + n_offset
    // This will be set up more directly for each STS.
    // int sts_b_n_sm_offsets[4]; // Removed, direct calculation now

    // --- SIMPLIFICATION: PROCESS ALL K ELEMENTS ---
    int elements_this_pass = K_sub; // Process all K elements, not just first tile
    // --- END SIMPLIFICATION ---

    // Initialize shared memory pointers for simplified access
    float* sm_a_ping_ptr = (float*)sm_a_ping_addr_u64;
    float* sm_b_ping_ptr = (float*)sm_b_ping_addr_u64;

    // Load all data for all K elements into shared memory
    for (int k_inner = 0; k_inner < elements_this_pass; ++k_inner) {
        // --- Load X and Y for current k_inner slice ---
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int current_m_logical = ldg_xy_m_base_global + i;
            int current_k_logical_for_xy_load = k_inner;

            float x_val = 0.0f;
            if (X_ptr_is_valid && current_m_logical < M_sub
                && current_k_logical_for_xy_load < K_sub) {
                x_val = X_gm[current_m_logical * ldx + current_k_logical_for_xy_load];
            }

            float y_val_s = 0.0f;
            if constexpr (DELTA_VAL != 0) {
                if (Y_ptr_is_valid && current_m_logical < M_sub
                    && current_k_logical_for_xy_load < K_sub) {
                    y_val_s = delta_coeff
                              * Y_gm[current_m_logical * ldy + current_k_logical_for_xy_load];
                }
            }
            ldg_reg_buffer_x[i] = x_val + y_val_s;

            // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0
            //     && k_inner == 0 && i == 0) {
            //     printf(
            //         "[KERNEL DEBUG LOAD] ldg_reg_buffer_x[0] = %.3f (x_val=%.3f + y_val_s=%.3f)\n",
            //         ldg_reg_buffer_x[i],
            //         x_val,
            //         y_val_s);
            // }
        }

        // --- Load V and W for current k_inner slice ---
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int current_n_logical = ldg_vw_n_base_global + i;
            int current_k_logical_for_vw_load = k_inner;

            float v_val = 0.0f;
            if (V_ptr_is_valid && current_n_logical < N_sub
                && current_k_logical_for_vw_load < K_sub) {
                v_val = V_gm[current_k_logical_for_vw_load * ldv + current_n_logical];
            }

            float w_val_s = 0.0f;
            if constexpr (EPSILON_VAL != 0) {
                if (W_ptr_is_valid && current_n_logical < N_sub
                    && current_k_logical_for_vw_load < K_sub) {
                    w_val_s = epsilon_coeff
                              * W_gm[current_k_logical_for_vw_load * ldw + current_n_logical];
                }
            }
            ldg_reg_buffer_y[i] = v_val + w_val_s;

            // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0
            //     && k_inner == 0 && i == 0) {
            //     printf(
            //         "[KERNEL DEBUG LOAD] ldg_reg_buffer_y[0] = %.3f (v_val=%.3f + w_val_s=%.3f)\n",
            //         ldg_reg_buffer_y[i],
            //         v_val,
            //         w_val_s);
            // }
        }
    }

    __syncthreads(); // Synchronize after all data is loaded

    // Perform matrix multiplication using register data (bypass shared memory for K > 8)
    // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("[KERNEL DEBUG FMA] Starting FMA computation, m_base=%d, n_base=%d\n",
    //            ldg_xy_m_base_global,
    //            ldg_vw_n_base_global);
    // }

#pragma unroll
    for (int r_fma = 0; r_fma < 8; ++r_fma) {     // Output M-dim of C_tile (0..7)
#pragma unroll
        for (int c_fma = 0; c_fma < 8; ++c_fma) { // Output N-dim of C_tile (0..7)
#pragma unroll
            for (int k_fma = 0; k_fma < elements_this_pass; ++k_fma) { // Use actual K elements
                // Direct computation without shared memory for K > 8
                int current_m_logical = ldg_xy_m_base_global + r_fma;
                int current_n_logical = ldg_vw_n_base_global + c_fma;

                if (current_m_logical < M_sub && current_n_logical < N_sub && k_fma < K_sub) {
                    // Load data directly from global memory
                    float x_val = 0.0f;
                    if (X_ptr_is_valid) { x_val = X_gm[current_m_logical * ldx + k_fma]; }

                    float y_val_s = 0.0f;
                    if constexpr (DELTA_VAL != 0) {
                        if (Y_ptr_is_valid) {
                            y_val_s = delta_coeff * Y_gm[current_m_logical * ldy + k_fma];
                        }
                    }
                    float a_val = x_val + y_val_s;

                    float v_val = 0.0f;
                    if (V_ptr_is_valid) { v_val = V_gm[k_fma * ldv + current_n_logical]; }

                    float w_val_s = 0.0f;
                    if constexpr (EPSILON_VAL != 0) {
                        if (W_ptr_is_valid) {
                            w_val_s = epsilon_coeff * W_gm[k_fma * ldw + current_n_logical];
                        }
                    }
                    float b_val = v_val + w_val_s;

                    accumulator[r_fma][c_fma] += a_val * b_val;
                }
            }
        }
    }

    // if (enable_debug_prints && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("[KERNEL DEBUG] accumulator[0][0] after all FMAs = %.3f\n", accumulator[0][0]);
    // }

    // STORE RESULTS TO GLOBAL MEMORY (D_gm, E_gm)
    if (threadIdx.x < 256) {
        const int thread_output_m_start_local = blockIdx.y * STRASSEN_TILE_M
                                                + (threadIdx.x / (STRASSEN_TILE_N / 8)) * 8;
        const int thread_output_n_start_local = blockIdx.x * STRASSEN_TILE_N
                                                + (threadIdx.x % (STRASSEN_TILE_N / 8)) * 8;

        for (int i = 0; i < 8; ++i) {
            int current_global_m = thread_output_m_start_local + i;
            if (current_global_m < M_sub) {
                for (int j = 0; j < 8; ++j) {
                    int current_global_n = thread_output_n_start_local + j;
                    if (current_global_n < N_sub) {
                        float m_val = accumulator[i][j];

                        if (effective_gamma0 != 0.0f) {
                            atomicAdd(&D_gm[current_global_m * ldd + current_global_n],
                                      effective_gamma0 * m_val);
                        }
                        if (effective_gamma1 != 0.0f) {
                            atomicAdd(&E_gm[current_global_m * lde + current_global_n],
                                      effective_gamma1 * m_val);
                        }
                    }
                }
            }
        }
    }
}

#endif
