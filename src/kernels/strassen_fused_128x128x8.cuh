#include "helper_cuda_ptx.h"
#include <cstdint>

template<int DELTA, int EPSILON, int GAMMA0_IS_ONE, int GAMMA1_SIGN>
// DELTA: 0 for no Y, 1 for +Y, -1 for -Y
// EPSILON: 0 for no W, 1 for +W, -1 for -W
// GAMMA0_IS_ONE: 1 if gamma0 is 1.0f. Assume gamma0 is always 1.0f for D updates.
// GAMMA1_SIGN: 0 for no E update, 1 for +E, -1 for -E
__global__ void strassen_fused_kernel(
    int m_sub, int n_sub, int k_sub,
    const float* X_ptr, int ldx,
    const float* Y_ptr, int ldy,
    const float* V_ptr, int ldv,
    const float* W_ptr, int ldw,
    float* D_ptr, int ldd,
    float* E_ptr, int lde) {
    // Operands X, Y, V, W, D, E: row-major format

    // Abbreviations:
    // ldg - load global
    // lds - load shared
    // stg - store global
    // sts - store shared
    // cvta - convert address

    const int smem_a_padding = 256;
    const int smem_a_size = smem_a_padding * 8;
    const int smem_a_ld = 132; // leading dimension
    const int smem_b_padding = 128;
    const int smem_b_size = smem_b_padding * 8;
    const int smem_b_ld = 128; // leading dimension

    __shared__ float __align__(2 * smem_a_size * sizeof(float))
        smem_ptr[2 * (smem_a_size + smem_b_size)];

    // C accumulator
    float accumulator[8][8]{};

    // Registers for (global memory -> shared memory) transfers
    float ldg_X_buffer[4];
    float ldg_Y_buffer[4];
    float ldg_V_buffer[4];
    float ldg_W_buffer[4];

    // Bitmasks to track in-bounds and out-of-bounds global memory reads
    unsigned ldg_X_bitmask = 0x0; // For X and Y reads
    unsigned ldg_V_bitmask = 0x0; // For V and W reads

    float* tile_A_fused = smem_ptr; // For X + delta*Y
    float* tile_B_fused = smem_ptr + 2 * smem_a_size; // For V + epsilon*W

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // --- Setup for X and Y loading (replaces A loading) ---
    int ldg_X_start_k_dim = threadIdx.x % 8; // k-dimension component of X
    int ldg_X_start_m_dim = blockIdx.y * 128 + 4 * (threadIdx.x / 8); // m-dimension component of X
    int ldg_X_start = ldg_X_start_k_dim + ldg_X_start_m_dim * ldx;
    const float* ldg_X_global_ptr = X_ptr + ldg_X_start;
    const float* ldg_Y_global_ptr = Y_ptr + ldg_X_start_k_dim + ldg_X_start_m_dim * ldy; // Y uses same k, m indices but its own ldy

    int ldg_XY_offsets_m[4]; // m-dimension offsets for X and Y
    int ldg_X_offsets_global[4]; // global memory offsets for X
    int ldg_Y_offsets_global[4]; // global memory offsets for Y
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_XY_offsets_m[i] = i;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_X_offsets_global[i] = ldg_XY_offsets_m[i] * ldx;
        if constexpr (DELTA != 0) {
            ldg_Y_offsets_global[i] = ldg_XY_offsets_m[i] * ldy;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int current_m_idx = ldg_X_start_m_dim + ldg_XY_offsets_m[i];
        // if global memory access is in-bounds for m_sub, flip corresponding bit
        if (current_m_idx < m_sub) { ldg_X_bitmask ^= (0x1 << i); }
    }

    // --- Setup for V and W loading (replaces B loading) ---
    int ldg_V_start_n_dim = blockIdx.x * 128 + threadIdx.x % 32; // n-dimension component of V
    int ldg_V_start_k_dim = threadIdx.x / 32; // k-dimension component of V
    int ldg_V_start = ldg_V_start_n_dim + ldg_V_start_k_dim * ldv;
    const float* ldg_V_global_ptr = V_ptr + ldg_V_start;
    const float* ldg_W_global_ptr = W_ptr + ldg_V_start_n_dim + ldg_V_start_k_dim * ldw; // W uses same n, k indices but its own ldw

    int ldg_VW_offsets_n[4]; // n-dimension offsets for V and W
    int ldg_V_offsets_global[4]; // global memory offsets for V
    int ldg_W_offsets_global[4]; // global memory offsets for W
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_VW_offsets_n[i] = 32 * i;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_V_offsets_global[i] = ldg_VW_offsets_n[i];
        if constexpr (EPSILON != 0) {
            ldg_W_offsets_global[i] = ldg_VW_offsets_n[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int current_n_idx = ldg_V_start_n_dim + ldg_VW_offsets_n[i];
        // if global memory access is in-bounds for n_sub, flip corresponding bit
        if (current_n_idx < n_sub) { ldg_V_bitmask ^= (0x1 << i); }
    }

    // Shared memory store pointers for tile_A_fused and tile_B_fused
    int sts_a_start_x = 4 * (threadIdx.x / 8); // m-dimension in shared
    int sts_a_start_y = threadIdx.x % 8;       // k-dimension in shared
    int sts_a_start = sts_a_start_x + sts_a_start_y * smem_a_ld;
    float* sts_tile_A_fused_ptr = tile_A_fused + sts_a_start;

    int sts_b_start_x = threadIdx.x % 32;      // n-dimension in shared
    int sts_b_start_y = threadIdx.x / 32;      // k-dimension in shared
    int sts_b_start = sts_b_start_x + sts_b_start_y * smem_b_ld;
    float* sts_tile_B_fused_ptr = tile_B_fused + sts_b_start;
    int sts_b_offsets[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        sts_b_offsets[i] = 32 * i;
    }

    uint64_t sts_a_addr;
    uint64_t sts_b_addr;

    // Convert from generic to .shared state space
    CVTA_TO_SHARED_PTX(sts_a_addr, sts_tile_A_fused_ptr);
    CVTA_TO_SHARED_PTX(sts_b_addr, sts_tile_B_fused_ptr);

    // if (k_sub % 8 == 0) {n_blocks_k = k_sub/8 - 1} else {n_blocks_k = k_sub/8;}
    int n_blocks_k = (k_sub + 7) / 8 - 1;
    int first_block_k_size = k_sub - 8 * n_blocks_k;

    // Load first blocks from global memory to shared memory (X+delta*Y and V+epsilon*W)
    // {
    float temp_ldg_X_buffer[4]; // Temporary buffer for STS128_PTX which needs 4 distinct float args
    float temp_ldg_V_buffer[4]; // Temporary buffer for STS operations on V+epsilon*W

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        bool guard_k_X = ldg_X_start_k_dim < first_block_k_size;
        bool guard_m_XY = ldg_X_bitmask & (0x1 << i); // Use ldg_X_bitmask for Y as well, as m_sub is the bounding dimension
        bool guard_X = guard_k_X && guard_m_XY;
        LDG32_GUARD_MOV0_PTX(ldg_X_buffer[i], ldg_X_global_ptr + ldg_X_offsets_global[i], (unsigned)guard_X);
        if constexpr (DELTA != 0) {
            bool guard_k_Y = ldg_X_start_k_dim < first_block_k_size; // Y shares k-dim with X
            // guard_m_XY already covers Y's m-dimension check
            bool guard_Y = guard_k_Y && guard_m_XY;
            LDG32_GUARD_MOV0_PTX(ldg_Y_buffer[i], ldg_Y_global_ptr + ldg_Y_offsets_global[i], (unsigned)guard_Y);
            temp_ldg_X_buffer[i] = ldg_X_buffer[i] + (float)DELTA * ldg_Y_buffer[i];
        } else {
            temp_ldg_X_buffer[i] = ldg_X_buffer[i];
        }
    }
    STS128_PTX(temp_ldg_X_buffer[0], temp_ldg_X_buffer[1], temp_ldg_X_buffer[2], temp_ldg_X_buffer[3], sts_a_addr);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        bool guard_k_V = ldg_V_start_k_dim < first_block_k_size;
        bool guard_n_VW = ldg_V_bitmask & (0x1 << i); // Use ldg_V_bitmask for W as well
        bool guard_V = guard_k_V && guard_n_VW;
        LDG32_GUARD_MOV0_PTX(ldg_V_buffer[i], ldg_V_global_ptr + ldg_V_offsets_global[i], (unsigned)guard_V);
        if constexpr (EPSILON != 0) {
            bool guard_k_W = ldg_V_start_k_dim < first_block_k_size; // W shares k-dim with V
            // guard_n_VW already covers W's n-dimension check
            bool guard_W = guard_k_W && guard_n_VW;
            LDG32_GUARD_MOV0_PTX(ldg_W_buffer[i], ldg_W_global_ptr + ldg_W_offsets_global[i], (unsigned)guard_W);
            temp_ldg_V_buffer[i] = ldg_V_buffer[i] + (float)EPSILON * ldg_W_buffer[i];
        } else {
            temp_ldg_V_buffer[i] = ldg_V_buffer[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i += 1) { // Store V+epsilon*W to tile_B_fused
        STS32_PTX(temp_ldg_V_buffer[i], sts_b_addr + sts_b_offsets[i] * sizeof(float));
    }
    __syncthreads();
    // }

    // FRAGMENT LOADING AND COMPUTATION WILL BE MODIFIED IN LATER SUBTASKS
    // For now, adapt pointers to use tile_A_fused and tile_B_fused
    float frag_tile_A_fused[2][8];
    float frag_tile_B_fused[2][8];

    uint64_t lds_a_addr; // For tile_A_fused
    uint64_t lds_b_addr; // For tile_B_fused

    int lane_id_mapped_x = 2 * (lane_id / 8) + (lane_id % 2);
    int lane_id_mapped_y = (lane_id / 2) % 4;
    int warp_id_mapped_x = 64 * (warp_id % 2);
    int warp_id_mapped_y = 32 * (warp_id / 2);

    int lds_a_start = 4 * lane_id_mapped_y + warp_id_mapped_y;
    int lds_b_start = 4 * lane_id_mapped_x + warp_id_mapped_x;
    float* lds_tile_A_fused_ptr = tile_A_fused + lds_a_start;
    float* lds_tile_B_fused_ptr = tile_B_fused + lds_b_start;

    // Convert from generic to .shared state space
    CVTA_TO_SHARED_PTX(lds_a_addr, lds_tile_A_fused_ptr);
    CVTA_TO_SHARED_PTX(lds_b_addr, lds_tile_B_fused_ptr);

    // Load first fragments from shared memory
    // {
    LDS128_PTX(frag_tile_A_fused[0][0], frag_tile_A_fused[0][1], frag_tile_A_fused[0][2], frag_tile_A_fused[0][3], lds_a_addr);
    LDS128_PTX(frag_tile_A_fused[0][4],
               frag_tile_A_fused[0][5],
               frag_tile_A_fused[0][6],
               frag_tile_A_fused[0][7],
               lds_a_addr + 16 * sizeof(float));
    LDS128_PTX(frag_tile_B_fused[0][0], frag_tile_B_fused[0][1], frag_tile_B_fused[0][2], frag_tile_B_fused[0][3], lds_b_addr);
    LDS128_PTX(frag_tile_B_fused[0][4],
               frag_tile_B_fused[0][5],
               frag_tile_B_fused[0][6],
               frag_tile_B_fused[0][7],
               lds_b_addr + 32 * sizeof(float));
    // }

    // Move global pointers to next blocks
    ldg_X_global_ptr += first_block_k_size;
    if constexpr (DELTA != 0) {
        ldg_Y_global_ptr += first_block_k_size;
    }
    ldg_V_global_ptr += first_block_k_size * ldv;
    if constexpr (EPSILON != 0) {
        ldg_W_global_ptr += first_block_k_size * ldw;
    }

    // Switch shared memory buffers
    sts_a_addr ^= 8192;
    sts_b_addr ^= 4096;

    // Iterate over k_sub and divide into k_sub_blocks
    for (int block_k = 0; block_k < n_blocks_k; block_k++) {

        // Prefetch next blocks from global memory (X, Y, V, W)
        // {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            bool guard_m_XY = (ldg_X_bitmask & (0x1 << i)); // Guard for m_sub dimension
            LDG32_GUARD_PTX(ldg_X_buffer[i], ldg_X_global_ptr + ldg_X_offsets_global[i], (unsigned)guard_m_XY);
            if constexpr (DELTA != 0) {
                LDG32_GUARD_PTX(ldg_Y_buffer[i], ldg_Y_global_ptr + ldg_Y_offsets_global[i], (unsigned)guard_m_XY);
            }

            bool guard_n_VW = (ldg_V_bitmask & (0x1 << i)); // Guard for n_sub dimension
            LDG32_GUARD_PTX(ldg_V_buffer[i], ldg_V_global_ptr + ldg_V_offsets_global[i], (unsigned)guard_n_VW);
            if constexpr (EPSILON != 0) {
                LDG32_GUARD_PTX(ldg_W_buffer[i], ldg_W_global_ptr + ldg_W_offsets_global[i], (unsigned)guard_n_VW);
            }
        }
        // }

        // --- CORE COMPUTATION (using frag_a, frag_b from current tile_A_fused, tile_B_fused) ---
        // --- This part will be significantly modified in a later subtask. ---
        // --- For now, it's mostly placeholder, using existing loop structure. ---
#pragma unroll
        for (int warp_k = 0; warp_k < 8; warp_k += 1) {
            int prefetch = (warp_k + 1) % 8;
            int frag_idx = warp_k & 1;
            int frag_next_idx = (warp_k + 1) & 1;

            // Prefetch next fragments from shared memory
            // {
            LDS128_PTX(frag_tile_A_fused[frag_next_idx][0],
                       frag_tile_A_fused[frag_next_idx][1],
                       frag_tile_A_fused[frag_next_idx][2],
                       frag_tile_A_fused[frag_next_idx][3],
                       lds_a_addr + prefetch * smem_a_ld * sizeof(float)); // lds_a_addr points to tile_A_fused
            LDS128_PTX(frag_tile_A_fused[frag_next_idx][4],
                       frag_tile_A_fused[frag_next_idx][5],
                       frag_tile_A_fused[frag_next_idx][6],
                       frag_tile_A_fused[frag_next_idx][7],
                       lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
            LDS128_PTX(frag_tile_B_fused[frag_next_idx][0],
                       frag_tile_B_fused[frag_next_idx][1],
                       frag_tile_B_fused[frag_next_idx][2],
                       frag_tile_B_fused[frag_next_idx][3],
                       lds_b_addr + prefetch * smem_b_ld * sizeof(float)); // lds_b_addr points to tile_B_fused
            LDS128_PTX(frag_tile_B_fused[frag_next_idx][4],
                       frag_tile_B_fused[frag_next_idx][5],
                       frag_tile_B_fused[frag_next_idx][6],
                       frag_tile_B_fused[frag_next_idx][7],
                       lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));
            // }

            // Update the accumulator (placeholder)
            // {
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    accumulator[i][j] += frag_tile_A_fused[frag_idx][i] * frag_tile_B_fused[frag_idx][j];
                }
            }
            // }
        }
        // --- END OF CORE COMPUTATION PLACEHOLDER ---

        // Store prefetched and summed blocks to shared memory (for *next* iteration's use)
        // This happens *before* __syncthreads() and buffer switch for sts_a_addr/sts_b_addr
        // {
#pragma unroll
        for (int i = 0; i < 4; ++i) { // Prepare data for tile_A_fused (X + delta*Y)
            if constexpr (DELTA != 0) {
                temp_ldg_X_buffer[i] = ldg_X_buffer[i] + (float)DELTA * ldg_Y_buffer[i];
            } else {
                temp_ldg_X_buffer[i] = ldg_X_buffer[i];
            }
        }
        STS128_PTX(temp_ldg_X_buffer[0], temp_ldg_X_buffer[1], temp_ldg_X_buffer[2], temp_ldg_X_buffer[3], sts_a_addr);

#pragma unroll
        for (int i = 0; i < 4; ++i) { // Prepare and store data for tile_B_fused (V + epsilon*W)
            if constexpr (EPSILON != 0) {
                temp_ldg_V_buffer[i] = ldg_V_buffer[i] + (float)EPSILON * ldg_W_buffer[i];
            } else {
                temp_ldg_V_buffer[i] = ldg_V_buffer[i];
            }
            STS32_PTX(temp_ldg_V_buffer[i], sts_b_addr + sts_b_offsets[i] * sizeof(float));
        }
        __syncthreads(); // Synchronize before switching buffers and loading from new shared mem
        // }

        // Switch shared memory buffers (for storing next prefetched global data)
        sts_a_addr ^= 8192;
        sts_b_addr ^= 4096;
        // Switch shared memory buffers (for loading fragments for computation)
        lds_a_addr ^= 8192;
        lds_b_addr ^= 4096;

        // Move global pointers to next blocks
        ldg_X_global_ptr += 8; // Each block has a k-depth of 8
        if constexpr (DELTA != 0) {
            ldg_Y_global_ptr += 8;
        }
        ldg_V_global_ptr += 8 * ldv; // V is indexed k,n so pointer moves by 8 * ldv
        if constexpr (EPSILON != 0) {
            ldg_W_global_ptr += 8 * ldw; // W is indexed k,n so pointer moves by 8 * ldw
        }

        // Load first fragments from shared memory
        // {
        LDS128_PTX(frag_tile_A_fused[0][0], frag_tile_A_fused[0][1], frag_tile_A_fused[0][2], frag_tile_A_fused[0][3], lds_a_addr);
        LDS128_PTX(frag_tile_A_fused[0][4],
                   frag_tile_A_fused[0][5],
                   frag_tile_A_fused[0][6],
                   frag_tile_A_fused[0][7],
                   lds_a_addr + 16 * sizeof(float));
        LDS128_PTX(frag_tile_B_fused[0][0], frag_tile_B_fused[0][1], frag_tile_B_fused[0][2], frag_tile_B_fused[0][3], lds_b_addr);
        LDS128_PTX(frag_tile_B_fused[0][4],
                   frag_tile_B_fused[0][5],
                   frag_tile_B_fused[0][6],
                   frag_tile_B_fused[0][7],
                   lds_b_addr + 32 * sizeof(float));
        // }
    }

    // Compute last block
    // {
#pragma unroll
    for (int warp_k = 0; warp_k < 8; warp_k += 1) {
        int prefetch = (warp_k + 1) % 8;
        int frag_idx = warp_k & 1;
        int frag_next_idx = (warp_k + 1) & 1;

        LDS128_PTX(frag_tile_A_fused[frag_next_idx][0],
                   frag_tile_A_fused[frag_next_idx][1],
                   frag_tile_A_fused[frag_next_idx][2],
                   frag_tile_A_fused[frag_next_idx][3],
                   lds_a_addr + prefetch * smem_a_ld * sizeof(float));
        LDS128_PTX(frag_tile_A_fused[frag_next_idx][4],
                   frag_tile_A_fused[frag_next_idx][5],
                   frag_tile_A_fused[frag_next_idx][6],
                   frag_tile_A_fused[frag_next_idx][7],
                   lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
        LDS128_PTX(frag_tile_B_fused[frag_next_idx][0],
                   frag_tile_B_fused[frag_next_idx][1],
                   frag_tile_B_fused[frag_next_idx][2],
                   frag_tile_B_fused[frag_next_idx][3],
                   lds_b_addr + prefetch * smem_b_ld * sizeof(float));
        LDS128_PTX(frag_tile_B_fused[frag_next_idx][4],
                   frag_tile_B_fused[frag_next_idx][5],
                   frag_tile_B_fused[frag_next_idx][6],
                   frag_tile_B_fused[frag_next_idx][7],
                   lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));

#pragma unroll
        for (int i = 0; i < 8; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                accumulator[i][j] += frag_tile_A_fused[frag_idx][i] * frag_tile_B_fused[frag_idx][j];
            }
        }
    }
    // }

    // Phase 3: Fused Output Accumulation (Registers to Global Memory)
    // The accumulator[8][8] holds M_tile.
    // D += gamma0 * M_tile  (where gamma0 is treated as 1.0 via GAMMA0_IS_ONE)
    // E += gamma1 * M_tile  (where gamma1 is GAMMA1_SIGN)

    // Shared memory region for staging output. Each warp handles 32x32 output region.
    // Each thread in a warp handles 4x4 from the accumulator.
    // Staging area: warp_id maps to a 512 float region (smem_ptr + 512 * warp_id).
    // Within this, lane_id maps to its specific part.
    // sts_c_addr in original kernel: smem_ptr + 512 * warp_id + 4 * 32 * lane_id_mapped_y + 4 * lane_id_mapped_x
    // lds_c_ptr in original kernel: (float*)((smem_ptr + 512 * warp_id + lane_id))
    // The original kernel wrote 16 floats per thread to shared mem (4x STS128 for the warp), then read them back for STG.

    uint64_t output_staging_sts_addr_base;
    // Each thread is responsible for a 4x4 region of the 8x8 accumulator.
    // lane_id_mapped_y maps to rows (0..3), lane_id_mapped_x maps to cols (0..7 for 32 threads, then further sub-divided)
    // For output, each thread handles 4 rows and 1 col from its 4x4 accumulator part per STG iteration.
    // Let's use a simpler shared memory layout for staging: each thread writes its 16 accumulator values.
    // Total shared memory for output staging per warp: 32 threads * 16 floats = 512 floats.
    // This matches the original C output staging area size per warp.
    float* output_staging_smem_ptr = smem_ptr + 2 * (smem_a_size + smem_b_size); // Use dedicated region after A and B tiles
                                                                                // Ensure this doesn't exceed total shared mem.
                                                                                // Original smem_ptr is 2*(smem_a_size+smem_b_size) = 2*(2048+1024) = 6144 floats.
                                                                                // Max shared mem is typically 48KB (12288 floats) or more.
                                                                                // Output staging size: 8 warps * 512 floats/warp = 4096 floats.
                                                                                // Total: tile_A(2*2048) + tile_B(2*1024) + output(4096) = 4096+2048+4096 = 10240 floats. OK.

    float* thread_output_staging_ptr = output_staging_smem_ptr + (threadIdx.x * 16);
    uint64_t thread_output_staging_addr_64;
    CVTA_TO_SHARED_PTX(thread_output_staging_addr_64, thread_output_staging_ptr);

    // Stage accumulator values to shared memory
    // Each thread has 8x8 accumulator, but is responsible for writing a 4x4 portion of the output block per iteration.
    // The mapping from accumulator[8][8] per thread to the 128x128 output block is complex.
    // The original mapping:
    // m_out_idx = blockIdx.y * 128 + warp_id_mapped_y; (base row for the 32x32 warp output)
    // n_out_idx = blockIdx.x * 128 + warp_id_mapped_x + lane_id; (base col for a single thread's output strip)
    // This implies each thread writes a strip of 16 elements in m-dim and 1 element in n-dim per p-loop.

    // Staging accumulator[8][8] to shared memory:
    // Each thread handles 16 elements from its accumulator that map to a 16xm_inputs x 1xn_inputs region.
    // The accumulator is 8x8. This maps to a 32x32 tile in C per warp.
    // lane_id_mapped_y = (lane_id/2)%4 -> maps to one of 4 sets of 4 rows in accumulator for this thread
    // lane_id_mapped_x = 2*(lane_id/8) + (lane_id%2) -> maps to one of 8 sets of 4 cols in accumulator
    // This means accumulator[i][j] is specific to a thread.

    // Store D_ptr += GAMMA0_IS_ONE * M_tile
    if constexpr (GAMMA0_IS_ONE != 0) { // GAMMA0_IS_ONE is 1 if gamma0 is 1.0f
        // Stage accumulator to shared memory (scaled by GAMMA0_IS_ONE, which is 1.0f)
        // Each thread has its own 8x8 accumulator.
        // The original code iterates i=0..1, j=0..1, p_sts=0..3, p_stg=0..15
        // i,j -> select one of four 4x4 sub-blocks of the accumulator.
        // p_sts -> selects which row within that 4x4 sub-block.
        // This means sts_c_addr needs to be based on warp_id, lane_id_mapped_x/y to correctly map accumulator.
        // The original sts_c_addr was: smem_ptr + 512 * warp_id + 4*32*lane_id_mapped_y + 4*lane_id_mapped_x
        // This is the base for a 4x4 tile in shared mem for this thread's portion of current C sub-block.
        
        uint64_t sts_target_smem_addr_base;
        int sts_offset_comp = 512 * warp_id + 4 * 32 * lane_id_mapped_y + 4 * lane_id_mapped_x;
        CVTA_TO_SHARED_PTX(sts_target_smem_addr_base, smem_ptr + sts_offset_comp); // Using original C staging area

        float* lds_src_smem_ptr = (float*)(smem_ptr + 512 * warp_id + lane_id); // Aligned with original lds_c_ptr

        int base_m_idx = blockIdx.y * 128 + warp_id_mapped_y; // Base m-index for this warp's 32-row high strip
        int base_n_idx = blockIdx.x * 128 + warp_id_mapped_x; // Base n-index for this thread group's 32-col wide strip

        if (base_m_idx < m_sub) { // Warp-level check for m-bound
            #pragma unroll 1
            for (int i_outer = 0; i_outer < 2; ++i_outer) { // iterates over 2x2 sub-blocks of 64x64 within 128x128 tile
                #pragma unroll 1
                for (int j_outer = 0; j_outer < 2; ++j_outer) {
                    __syncthreads(); // Sync before STS to shared mem
                    #pragma unroll 2 // Process 4 rows from accumulator for current sub-block
                    for (int p_sts = 0; p_sts < 4; ++p_sts) {
                        // accumulator[row_in_tile_A][col_in_tile_B]
                        // row_in_tile_A from warp_id_mapped_y and lane_id_mapped_y
                        // col_in_tile_B from warp_id_mapped_x and lane_id_mapped_x
                        // This is already handled by accumulator[acc_r][acc_c] being thread-local.
                        // We need to map i_outer, j_outer, p_sts to acc_r, acc_c
                        // acc_r = i_outer * 4 + p_sts
                        // acc_c = j_outer * 4 + {0,1,2,3} for STS128
                        STS128_PTX(accumulator[i_outer * 4 + p_sts][j_outer * 4 + 0],
                                   accumulator[i_outer * 4 + p_sts][j_outer * 4 + 1],
                                   accumulator[i_outer * 4 + p_sts][j_outer * 4 + 2],
                                   accumulator[i_outer * 4 + p_sts][j_outer * 4 + 3],
                                   sts_target_smem_addr_base + p_sts * 8 * sizeof(float4)); // Store 4x4 block part
                    }
                    __syncthreads(); // Sync after STS, before STG from shared mem

                    #pragma unroll 4 // Each thread handles 16 rows, 1 column from its sub-block for global store
                    for (int p_stg = 0; p_stg < 16; ++p_stg) { // p_stg is row within the 16x32 C tile this thread works on
                        int current_m_offset = base_m_idx + i_outer * 16 + p_stg;
                        int current_n_offset = base_n_idx + j_outer * 32 + lane_id; // lane_id directly maps to column within 32-wide strip
                        
                        bool guard = (current_m_offset < m_sub) && (current_n_offset < n_sub);
                        
                        if (guard) {
                            float val_m = lds_src_smem_ptr[p_stg * 32]; // Read value staged by this thread
                                                                       // (or corresponding thread if STS was different)
                                                                       // Original lds_c_ptr[p*32] suggests each thread reads its own column data
                                                                       // from shared memory that was written by potentially different threads.
                                                                       // The STS128 writes 4 floats for a row.
                                                                       // The LDS by lds_src_smem_ptr[p_stg*32] means thread `lane_id` reads
                                                                       // the `lane_id`-th float from every 32-float line `p_stg`.
                                                                       // This matches the original C write pattern.

                            float* d_glob_ptr = D_ptr + current_m_offset * ldd + current_n_offset;
                            float d_current_val;
                            LDG32_GUARD_MOV0_PTX(d_current_val, d_glob_ptr, guard);
                            d_current_val += val_m; // GAMMA0_IS_ONE is 1.0f
                            STG32_GUARD_PTX(d_current_val, d_glob_ptr, guard);
                        }
                    }
                }
            }
        }
    }

    // Store E_ptr += GAMMA1_SIGN * M_tile
    if constexpr (GAMMA1_SIGN != 0) {
        uint64_t sts_target_smem_addr_base;
        int sts_offset_comp = 512 * warp_id + 4 * 32 * lane_id_mapped_y + 4 * lane_id_mapped_x;
        CVTA_TO_SHARED_PTX(sts_target_smem_addr_base, smem_ptr + sts_offset_comp); // Reuse same C staging area

        float* lds_src_smem_ptr = (float*)(smem_ptr + 512 * warp_id + lane_id);

        int base_m_idx = blockIdx.y * 128 + warp_id_mapped_y;
        int base_n_idx = blockIdx.x * 128 + warp_id_mapped_x;

        if (base_m_idx < m_sub) { // Warp-level check for m-bound
            #pragma unroll 1
            for (int i_outer = 0; i_outer < 2; ++i_outer) {
                #pragma unroll 1
                for (int j_outer = 0; j_outer < 2; ++j_outer) {
                    __syncthreads(); 
                    #pragma unroll 2
                    for (int p_sts = 0; p_sts < 4; ++p_sts) {
                        // Scale accumulator values by GAMMA1_SIGN before staging
                        float acc_val0 = (float)GAMMA1_SIGN * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 0];
                        float acc_val1 = (float)GAMMA1_SIGN * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 1];
                        float acc_val2 = (float)GAMMA1_SIGN * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 2];
                        float acc_val3 = (float)GAMMA1_SIGN * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 3];
                        STS128_PTX(acc_val0, acc_val1, acc_val2, acc_val3,
                                   sts_target_smem_addr_base + p_sts * 8 * sizeof(float4));
                    }
                    __syncthreads(); 

                    #pragma unroll 4
                    for (int p_stg = 0; p_stg < 16; ++p_stg) {
                        int current_m_offset = base_m_idx + i_outer * 16 + p_stg;
                        int current_n_offset = base_n_idx + j_outer * 32 + lane_id;
                        
                        bool guard = (current_m_offset < m_sub) && (current_n_offset < n_sub);

                        if (guard) {
                            float val_m_scaled = lds_src_smem_ptr[p_stg * 32]; // Value is already scaled by GAMMA1_SIGN

                            float* e_glob_ptr = E_ptr + current_m_offset * lde + current_n_offset;
                            float e_current_val;
                            LDG32_GUARD_MOV0_PTX(e_current_val, e_glob_ptr, guard);
                            e_current_val += val_m_scaled;
                            STG32_GUARD_PTX(e_current_val, e_glob_ptr, guard);
                        }
                    }
                }
            }
        }
    }
}
