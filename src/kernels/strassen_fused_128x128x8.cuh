#ifndef STRASSEN_FUSED_128X128X8_KERNEL_CUH_
#define STRASSEN_FUSED_128X128X8_KERNEL_CUH_

#include <cstdio> // For printf in kernel if needed for debugging
#include <cstdint>
#include "common/helper_cuda ptx.h" // Corrected path

// Tile dimensions (fixed for this kernel)
constexpr int M_TILE_SIZE = 128;
constexpr int N_TILE_SIZE = 128;
constexpr int K_TILE_SIZE = 8; // K-depth processed by one iteration of the main loop using one set of shared mem buffers

// Shared memory layout parameters for A and B tiles
constexpr int SMEM_A_ROWS = M_TILE_SIZE; // e.g., 128 rows for A tile
constexpr int SMEM_A_COLS = K_TILE_SIZE; // e.g., 8 columns for A tile
constexpr int SMEM_A_LD_PADDING = 4; // Padding for leading dimension of A in shared mem (e.g., 8+4=12)
constexpr int SMEM_A_ELEMENTS_PER_TILE = SMEM_A_ROWS * (SMEM_A_COLS + SMEM_A_LD_PADDING); // Total elements for one A tile buffer

constexpr int SMEM_B_ROWS = K_TILE_SIZE; // e.g., 8 rows for B tile
constexpr int SMEM_B_COLS = N_TILE_SIZE; // e.g., 128 columns for B tile
// No padding usually needed for B if it's read column-wise for MMA, or if STS is simple.
// The original sgemm_128x128x8.cuh uses: smem_b_ld = 128 (no padding)
constexpr int SMEM_B_LD_PADDING = 0;
constexpr int SMEM_B_ELEMENTS_PER_TILE = SMEM_B_ROWS * (SMEM_B_COLS + SMEM_B_LD_PADDING); // Total elements for one B tile buffer

// Double buffered shared memory: 2 tiles for A, 2 for B
// Kernel will use dynamically allocated shared memory. The host launcher will calculate this size.
extern __shared__ float smem_storage[];


// Helper function to get coefficient value based on template parameter
// (0 -> 0.0f, 1 -> 1.0f, -1 -> -1.0f)
__device__ inline float get_strassen_coeff_val(int coeff_template_param) {
    if (coeff_template_param == 0) return 0.0f;
    if (coeff_template_param == 1) return 1.0f;
    // if (coeff_template_param == -1) return -1.0f;
    return -1.0f; // Defaulting for -1 or any other non-zero/one value
}


template<
    int DELTA_XY_COEFF, // 0: X, 1: X+Y, -1: X-Y
    int EPSILON_VW_COEFF, // 0: V, 1: V+W, -1: V-W
    int GAMMA0_DEST_COEFF, // Coeff for D_ptr output (0, 1, -1)
    int GAMMA1_DEST_COEFF  // Coeff for E_ptr output (0, 1, -1)
>
__global__
__launch_bounds__(256, 2) // Standard for this tile size
void strassen_fused_kernel_128x128x8_vP(
    int m_sub, int n_sub, int k_sub,      // Sub-problem dimensions (M, N, K for this call)
    const float* X_ptr, int ldx,         // Input X matrix
    const float* Y_ptr, int ldy,         // Input Y matrix
    const float* V_ptr, int ldv,         // Input V matrix
    const float* W_ptr, int ldw,         // Input W matrix
    float* D_ptr, int ldd,               // Output D matrix
    float* E_ptr, int lde                // Output E matrix
) {
    // Shared memory pointers
    float* tile_A_fused_smem[2];
    tile_A_fused_smem[0] = smem_storage;
    tile_A_fused_smem[1] = smem_storage + SMEM_A_ELEMENTS_PER_TILE;

    float* tile_B_fused_smem[2];
    tile_B_fused_smem[0] = smem_storage + 2 * SMEM_A_ELEMENTS_PER_TILE;
    tile_B_fused_smem[1] = smem_storage + 2 * SMEM_A_ELEMENTS_PER_TILE + SMEM_B_ELEMENTS_PER_TILE;
    
    // --- Start of code from original sgemm_128x128x8.cuh, adapted ---
    // C accumulator
    float accumulator[8][8]{}; // Each thread computes an 8x8 tile of the output C (D or E)

    // Registers for (global memory -> shared memory) transfers
    float ldg_X_buffer[4];
    float ldg_Y_buffer[4]; // Only used if DELTA_XY_COEFF != 0
    float ldg_V_buffer[4];
    float ldg_W_buffer[4]; // Only used if EPSILON_VW_COEFF != 0

    // Bitmasks to track in-bounds and out-of-bounds global memory reads
    unsigned ldg_X_Y_m_guard_bitmask = 0x0; // For X and Y reads, m-dimension guard
    unsigned ldg_V_W_n_guard_bitmask = 0x0; // For V and W reads, n-dimension guard

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // --- Setup for X and Y loading (Tile A loading: X + delta*Y) ---
    // Each thread loads 4 elements for X and potentially 4 for Y along K-dimension over M-dimension
    int ldg_XY_k_offset_in_tile = threadIdx.x % K_TILE_SIZE; // 0..7, which k-element in the 8-element deep tile
    int ldg_XY_m_start_block = blockIdx.y * M_TILE_SIZE; // Starting M-dim row for this CUDA block
    // Each group of K_TILE_SIZE threads (e.g., 8 threads) loads one "row" of the A tile (128 elements high in M, 1 element deep in K)
    int ldg_XY_m_base_thread_group = (threadIdx.x / K_TILE_SIZE) * 4; // Base M-offset for this thread group (0, 4, 8 .. up to 124 for 256 threads / 8 k_threads_per_group)
                                                                 // Max threadIdx.x = 255. Max (255/8)*4 = 31*4 = 124. OK.
    
    const float* ldg_X_global_base_ptr = X_ptr + ldg_XY_m_start_block * ldx + ldg_XY_k_offset_in_tile;
    const float* ldg_Y_global_base_ptr = Y_ptr + ldg_XY_m_start_block * ldy + ldg_XY_k_offset_in_tile;

    // Offsets for the 4 M-elements each thread handles
    int ldg_XY_m_offsets_vals[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_XY_m_offsets_vals[i] = ldg_XY_m_base_thread_group + i;
        if ((ldg_XY_m_start_block + ldg_XY_m_offsets_vals[i]) < m_sub) {
            ldg_X_Y_m_guard_bitmask |= (1 << i);
        }
    }
    
    // --- Setup for V and W loading (Tile B loading: V + epsilon*W) ---
    // Each thread loads 4 elements for V (and pot. W) along N-dim over K-dim
    int ldg_VW_k_base_thread_group = (threadIdx.x / 32); // K-dim row for this warp (0..7)
    int ldg_VW_n_start_block = blockIdx.x * N_TILE_SIZE; // Starting N-dim col for this CUDA block
    int ldg_VW_n_base_thread_in_warp = threadIdx.x % 32; // N-dim base offset within the warp (0..31)

    const float* ldg_V_global_base_ptr = V_ptr + ldg_VW_k_base_thread_group * ldv + ldg_VW_n_start_block;
    const float* ldg_W_global_base_ptr = W_ptr + ldg_VW_k_base_thread_group * ldw + ldg_VW_n_start_block;

    int ldg_VW_n_offsets_vals[4];
    #pragma unroll
    for (int i=0; i<4; ++i) {
        ldg_VW_n_offsets_vals[i] = ldg_VW_n_base_thread_in_warp + i * 32; // Strides of 32 to cover N_TILE_SIZE=128 cols
        if ((ldg_VW_n_start_block + ldg_VW_n_offsets_vals[i]) < n_sub) {
            ldg_V_W_n_guard_bitmask |= (1 << i);
        }
    }

    // Shared memory store pointers
    // For tile A (X+dY): threads store elements column-wise into shared memory (matching K-dim)
    // SMEM_A_COLS = K_TILE_SIZE (e.g. 8), SMEM_A_LD_PADDING (e.g. 4), effective_smem_a_ld = 12
    int sts_XY_m_offset = ldg_XY_m_base_thread_group; // Base M-dim row in shared memory tile for this thread's writes
    int sts_XY_k_offset = ldg_XY_k_offset_in_tile;    // K-dim col in shared memory tile
    float* sts_tile_A_fused_smem_ptr_base = tile_A_fused_smem[0] + sts_XY_m_offset * (SMEM_A_COLS + SMEM_A_LD_PADDING) + sts_XY_k_offset;

    // For tile B (V+eW): threads store elements row-wise into shared memory
    // SMEM_B_ROWS = K_TILE_SIZE (e.g. 8), SMEM_B_COLS = N_TILE_SIZE (e.g. 128)
    int sts_VW_k_offset = ldg_VW_k_base_thread_group; // K-dim row in shared memory tile
    int sts_VW_n_offset = ldg_VW_n_base_thread_in_warp; // N-dim col in shared memory tile (base for 4 elements)
    float* sts_tile_B_fused_smem_ptr_base = tile_B_fused_smem[0] + sts_VW_k_offset * (SMEM_B_COLS + SMEM_B_LD_PADDING) + sts_VW_n_offset;
    
    float delta_coeff = get_strassen_coeff_val(DELTA_XY_COEFF);
    float epsilon_coeff = get_strassen_coeff_val(EPSILON_VW_COEFF);

    int num_k_tiles = (k_sub + K_TILE_SIZE - 1) / K_TILE_SIZE;
    int smem_buffer_idx = 0; // Start with buffer 0 for STS, compute on buffer 1 (pre-filled if k_tile_iter > 0)

    // --- Main K-loop ---
    for (int k_tile_iter = 0; k_tile_iter < num_k_tiles; ++k_tile_iter) {
        int current_k_block_start_offset_in_K = k_tile_iter * K_TILE_SIZE; // k offset for global pointers
        
        // Determine actual K elements for this tile (handling last potentially partial tile)
        int k_elements_in_this_physical_tile = K_TILE_SIZE;
        if (k_tile_iter == num_k_tiles - 1) { // If it's the last tile
            k_elements_in_this_physical_tile = k_sub - current_k_block_start_offset_in_K;
        }

        // --- Load data from Global to Shared Memory (current STS buffer) ---
        uint64_t sts_smem_A_addr_u64, sts_smem_B_addr_u64;
        CVTA_TO_SHARED_PTX(sts_smem_A_addr_u64, sts_tile_A_fused_smem_ptr_base + smem_buffer_idx * SMEM_A_ELEMENTS_PER_TILE);
        CVTA_TO_SHARED_PTX(sts_smem_B_addr_u64, sts_tile_B_fused_smem_ptr_base + smem_buffer_idx * SMEM_B_ELEMENTS_PER_TILE);

        // Guard for k-dimension on global loads
        bool k_guard_XY = (ldg_XY_k_offset_in_tile < k_elements_in_this_physical_tile);

        #pragma unroll
        for (int i=0; i<4; ++i) { 
            bool m_guard_XY = (ldg_X_Y_m_guard_bitmask >> i) & 0x1;
            bool final_guard_XY = k_guard_XY && m_guard_XY;

            LDG32_GUARD_MOV0_PTX(ldg_X_buffer[i], ldg_X_global_base_ptr + ldg_XY_m_offsets_vals[i] * ldx + (ptrdiff_t)current_k_block_start_offset_in_K, final_guard_XY);
            if (DELTA_XY_COEFF != 0) {
                LDG32_GUARD_MOV0_PTX(ldg_Y_buffer[i], ldg_Y_global_base_ptr + ldg_XY_m_offsets_vals[i] * ldy + (ptrdiff_t)current_k_block_start_offset_in_K, final_guard_XY);
                ldg_X_buffer[i] = ldg_X_buffer[i] + delta_coeff * ldg_Y_buffer[i];
            } else if (!final_guard_XY) { // Ensure zero if out of bounds and no Y to potentially zero it out
                 ldg_X_buffer[i] = 0.0f;
            }
            STS32_PTX(ldg_X_buffer[i], sts_smem_A_addr_u64 + i * (SMEM_A_COLS + SMEM_A_LD_PADDING) * sizeof(float));
        }

        bool k_guard_VW = (ldg_VW_k_base_thread_group < k_elements_in_this_physical_tile);
        #pragma unroll
        for (int i=0; i<4; ++i) { 
            bool n_guard_VW = (ldg_V_W_n_guard_bitmask >> i) & 0x1;
            bool final_guard_VW = k_guard_VW && n_guard_VW;

            LDG32_GUARD_MOV0_PTX(ldg_V_buffer[i], ldg_V_global_base_ptr + (ptrdiff_t)current_k_block_start_offset_in_K * ldv + ldg_VW_n_offsets_vals[i], final_guard_VW);
            if (EPSILON_VW_COEFF != 0) {
                LDG32_GUARD_MOV0_PTX(ldg_W_buffer[i], ldg_W_global_base_ptr + (ptrdiff_t)current_k_block_start_offset_in_K * ldw + ldg_VW_n_offsets_vals[i], final_guard_VW);
                ldg_V_buffer[i] = ldg_V_buffer[i] + epsilon_coeff * ldg_W_buffer[i];
            } else if (!final_guard_VW) {
                ldg_V_buffer[i] = 0.0f;
            }
            STS32_PTX(ldg_V_buffer[i], sts_smem_B_addr_u64 + i * 32 * sizeof(float) ); 
        }
        __syncthreads();

        // --- Compute MMA operations using data from shared memory (previous LDS buffer) ---
        // LDS buffer is the one NOT currently being written to by STS
        float* current_lds_tile_A_smem_base = tile_A_fused_smem[(smem_buffer_idx + 1) % 2]; 
        float* current_lds_tile_B_smem_base = tile_B_fused_smem[(smem_buffer_idx + 1) % 2];
        
        uint64_t lds_smem_A_addr_u64, lds_smem_B_addr_u64;
        
        int lane_id_mapped_x_lds = 2 * (lane_id / 8) + (lane_id % 2); 
        int lane_id_mapped_y_lds = (lane_id / 2) % 4; 
        int warp_id_mapped_x_lds = (N_TILE_SIZE/2) * (warp_id % 2); 
        int warp_id_mapped_y_lds = (M_TILE_SIZE/4) * (warp_id / 2); 

        // Effective pointers for fragment loading
        float* lds_A_ptr_for_frag = current_lds_tile_A_smem_base + (warp_id_mapped_y_lds + 4 * lane_id_mapped_y_lds) * (SMEM_A_COLS + SMEM_A_LD_PADDING);
        float* lds_B_ptr_for_frag = current_lds_tile_B_smem_base + (warp_id_mapped_x_lds + 4 * lane_id_mapped_x_lds); // B is KxN, each thread loads 4 elements of a row of B for its 8x8 computation

        CVTA_TO_SHARED_PTX(lds_smem_A_addr_u64, lds_A_ptr_for_frag);
        CVTA_TO_SHARED_PTX(lds_smem_B_addr_u64, lds_B_ptr_for_frag); // This points to start of a 4-element group

        float frag_A[2][8], frag_B[2][8]; 

        LDS128_PTX(frag_A[0][0], frag_A[0][1], frag_A[0][2], frag_A[0][3], lds_smem_A_addr_u64);
        LDS128_PTX(frag_A[0][4], frag_A[0][5], frag_A[0][6], frag_A[0][7], lds_smem_A_addr_u64 + 4 * sizeof(float)); // Corrected offset (16 bytes)
        
        LDS128_PTX(frag_B[0][0], frag_B[0][1], frag_B[0][2], frag_B[0][3], lds_smem_B_addr_u64); // Loads B[k][0,1,2,3] for this thread's 8x8 block
        LDS128_PTX(frag_B[0][4], frag_B[0][5], frag_B[0][6], frag_B[0][7], lds_smem_B_addr_u64 + 4*sizeof(float)); // Loads B[k][4,5,6,7]

        #pragma unroll
        for (int k_frag_iter = 0; k_frag_iter < K_TILE_SIZE; ++k_frag_iter) {
            int current_frag_buf_idx = k_frag_iter % 2;
            int next_frag_buf_idx = (k_frag_iter + 1) % 2;

            if (k_frag_iter < K_TILE_SIZE -1) { 
                 uint64_t next_lds_A_addr_base_for_frag = lds_smem_A_addr_u64 + (k_frag_iter + 1) * sizeof(float); // Base for the next K-slice for this M-row group
                 LDS128_PTX(frag_A[next_frag_buf_idx][0], frag_A[next_frag_buf_idx][1], frag_A[next_frag_buf_idx][2], frag_A[next_frag_buf_idx][3], next_lds_A_addr_base_for_frag);
                 LDS128_PTX(frag_A[next_frag_buf_idx][4], frag_A[next_frag_buf_idx][5], frag_A[next_frag_buf_idx][6], frag_A[next_frag_buf_idx][7], next_lds_A_addr_base_for_frag + 4*sizeof(float)); // Corrected offset

                 uint64_t next_lds_B_addr_base_for_frag = lds_smem_B_addr_u64 + (k_frag_iter + 1) * (SMEM_B_COLS + SMEM_B_LD_PADDING) * sizeof(float); // Base for the next K-row for this N-col group
                 LDS128_PTX(frag_B[next_frag_buf_idx][0], frag_B[next_frag_buf_idx][1], frag_B[next_frag_buf_idx][2], frag_B[next_frag_buf_idx][3], next_lds_B_addr_base_for_frag);
                 LDS128_PTX(frag_B[next_frag_buf_idx][4], frag_B[next_frag_buf_idx][5], frag_B[next_frag_buf_idx][6], frag_B[next_frag_buf_idx][7], next_lds_B_addr + 4*sizeof(float));
            }

            #pragma unroll
            for (int i = 0; i < 8; ++i) { 
                #pragma unroll
                for (int j = 0; j < 8; ++j) { 
                    accumulator[i][j] += frag_A[current_frag_buf_idx][i] * frag_B[current_frag_buf_idx][j];
                }
            }
        }
        smem_buffer_idx = (smem_buffer_idx + 1) % 2; 
    } // End K-loop

    // --- Write results from accumulator to Global Memory (D and E matrices) ---
    float gamma0_val = get_strassen_coeff_val(GAMMA0_DEST_COEFF);
    float gamma1_val = get_strassen_coeff_val(GAMMA1_DEST_COEFF);

    // Output mapping from original sgemm_128x128x8.cuh
    uint64_t sts_output_smem_addr_base;
    int sts_output_offset_comp = 512 * warp_id + 4 * 32 * lane_id_mapped_y_lds + 4 * lane_id_mapped_x_lds;
    CVTA_TO_SHARED_PTX(sts_output_smem_addr_base, smem_buffer + sts_output_offset_comp); // Use start of smem_buffer for output staging

    float* lds_output_smem_ptr = (float*)(smem_buffer + 512 * warp_id + lane_id);

    int out_base_m = blockIdx.y * M_TILE_SIZE + warp_id_mapped_y_lds;
    int out_base_n = blockIdx.x * N_TILE_SIZE + warp_id_mapped_x_lds;

    if (GAMMA0_DEST_COEFF != 0) {
        if (out_base_m < m_sub) { 
            #pragma unroll 1
            for (int i_outer = 0; i_outer < 2; ++i_outer) { 
                #pragma unroll 1
                for (int j_outer = 0; j_outer < 2; ++j_outer) {
                    __syncthreads(); 
                    #pragma unroll 2
                    for (int p_sts = 0; p_sts < 4; ++p_sts) { 
                        STS128_PTX(gamma0_val * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 0],
                                   gamma0_val * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 1],
                                   gamma0_val * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 2],
                                   gamma0_val * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 3],
                                   sts_output_smem_addr_base + p_sts * 8 * sizeof(float4));
                    }
                    __syncthreads(); 
                    #pragma unroll 4
                    for (int p_stg = 0; p_stg < 16; ++p_stg) { 
                        int current_m = out_base_m + i_outer * 16 + p_stg;
                        int current_n = out_base_n + j_outer * 32 + lane_id;
                        bool guard = (current_m < m_sub) && (current_n < n_sub);
                        if (guard) {
                            float val_to_add = lds_output_smem_ptr[p_stg * 32];
                            float* d_glob_ptr = D_ptr + (ptrdiff_t)current_m * ldd + current_n;
                            float d_current_val = 0.0f; // Initialize to 0 before LDG if D_ptr is not pre-initialized with beta*C
                            LDG32_GUARD_MOV0_PTX(d_current_val, d_glob_ptr, guard);
                            d_current_val += val_to_add;
                            STG32_GUARD_PTX(d_current_val, d_glob_ptr, guard);
                        }
                    }
                }
            }
        }
    }

    if (GAMMA1_DEST_COEFF != 0) {
         if (out_base_m < m_sub) { 
            #pragma unroll 1
            for (int i_outer = 0; i_outer < 2; ++i_outer) { 
                #pragma unroll 1
                for (int j_outer = 0; j_outer < 2; ++j_outer) {
                    __syncthreads(); 
                    #pragma unroll 2
                    for (int p_sts = 0; p_sts < 4; ++p_sts) { 
                        STS128_PTX(gamma1_val * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 0],
                                   gamma1_val * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 1],
                                   gamma1_val * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 2],
                                   gamma1_val * accumulator[i_outer * 4 + p_sts][j_outer * 4 + 3],
                                   sts_output_smem_addr_base + p_sts * 8 * sizeof(float4));
                    }
                    __syncthreads(); 
                    #pragma unroll 4
                    for (int p_stg = 0; p_stg < 16; ++p_stg) { 
                        int current_m = out_base_m + i_outer * 16 + p_stg;
                        int current_n = out_base_n + j_outer * 32 + lane_id;
                        bool guard = (current_m < m_sub) && (current_n < n_sub);
                        if (guard) {
                            float val_to_add = lds_output_smem_ptr[p_stg * 32];
                            float* e_glob_ptr = E_ptr + (ptrdiff_t)current_m * lde + current_n;
                            float e_current_val = 0.0f; // Initialize to 0 if E_ptr is not pre-initialized
                            LDG32_GUARD_MOV0_PTX(e_current_val, e_glob_ptr, guard);
                            e_current_val += val_to_add;
                            STG32_GUARD_PTX(e_current_val, e_glob_ptr, guard);
                        }
                    }
                }
            }
        }
    }
}

#endif // STRASSEN_FUSED_128X128X8_KERNEL_CUH_
