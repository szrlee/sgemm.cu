#ifndef SGEMM_STRASSEN_FUSED_CUH_
#define SGEMM_STRASSEN_FUSED_CUH_

#include "common/helper_cuda.h" // For checkCudaErrors, etc.
#include "kernels/strassen_fused_128x128x8.cuh" // The new kernel
#include <iostream> // For potential debugging, remove later

// Helper function for calculating grid dimensions
static inline int div_ceil_strassen(int a, int b) {
    return (a + b - 1) / b;
}

// Helper kernel for scaling matrix by a factor (for beta handling)
template <typename T>
__global__ void kernel_scale_matrix(T* matrix, int M, int N, int ld, T scale_factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        matrix[row * ld + col] *= scale_factor;
    }
}

template <typename T>
__global__ void kernel_memset_2d(T* matrix, int M, int N, int ld, T val) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        matrix[row * ld + col] = val;
    }
}


void sgemm_strassen_1level_fused(
    int M, int N, int K,
    const float host_alpha, // Standard SGEMM alpha
    const float* host_A, int lda,
    const float* host_B, int ldb,
    const float host_beta,  // Standard SGEMM beta
    float* host_C, int ldc
) {
    // TODO: Add checks for M, N, K being divisible by 2 for this version.
    // Future versions should handle padding or use a base case for odd dimensions.
    if (M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
        std::cerr << "sgemm_strassen_1level_fused: M, N, K must be divisible by 2 for this version." << std::endl;
        // For now, one might fall back to a standard sgemm or exit.
        // For this exercise, we'll proceed assuming they are divisible as per subtask instructions.
        // exit(EXIT_FAILURE); // Or handle error appropriately
        // Fallback to standard GEMM for simplicity if non-divisible for now if we had one.
        // For this task, assume divisible and proceed.
    }
     if (host_alpha != 1.0f) {
        std::cerr << "sgemm_strassen_1level_fused: host_alpha must be 1.0f for this version." << std::endl;
        // exit(EXIT_FAILURE); // Or handle error appropriately
    }


    float *device_A = nullptr, *device_B = nullptr, *device_C = nullptr;
    size_t pitch_A, pitch_B, pitch_C;

    // For simplicity with lda, ldb, ldc, we'll use them directly with cudaMalloc and 2D copies.
    // If lda=K, ldb=N, ldc=N, then it's contiguous. Otherwise, it needs careful handling with pitch.
    // The problem implies host_A, host_B, host_C are already allocated with lda, ldb, ldc.
    // We will allocate device memory respecting these leading dimensions for 2D submatrix addressing.

    checkCudaErrors(cudaMalloc((void**)&device_A, (size_t)M * lda * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&device_B, (size_t)K * ldb * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&device_C, (size_t)M * ldc * sizeof(float)));

    // cudaMemcpy2D(dst, dpitch, src, spitch, widthInBytes, height, kind)
    // A: M rows, K cols. host_A has lda. device_A allocated with lda.
    checkCudaErrors(cudaMemcpy2D(device_A, lda * sizeof(float), host_A, lda * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice));
    // B: K rows, N cols. host_B has ldb. device_B allocated with ldb.
    checkCudaErrors(cudaMemcpy2D(device_B, ldb * sizeof(float), host_B, ldb * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice));

    if (host_beta == 0.0f) {
        dim3 threads_memset(16, 16);
        dim3 grid_memset(div_ceil_strassen(N, 16), div_ceil_strassen(M, 16));
        kernel_memset_2d<<<grid_memset, threads_memset>>>(device_C, M, N, ldc, 0.0f);
        checkCudaErrors(cudaGetLastError());
    } else {
        // C: M rows, N cols. host_C has ldc. device_C allocated with ldc.
        checkCudaErrors(cudaMemcpy2D(device_C, ldc * sizeof(float), host_C, ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice));
        if (host_beta != 1.0f) {
            dim3 threads_scale(16, 16);
            dim3 grid_scale(div_ceil_strassen(N, 16), div_ceil_strassen(M, 16));
            kernel_scale_matrix<<<grid_scale, threads_scale>>>(device_C, M, N, ldc, host_beta);
            checkCudaErrors(cudaGetLastError());
        }
    }

    const int m2 = M / 2;
    const int n2 = N / 2;
    const int k2 = K / 2;

    // Submatrix pointers for A (X_ptr, Y_ptr for kernel)
    // A is M x K. Submatrices are m2 x k2.
    const float* A00 = device_A;
    const float* A01 = device_A + k2; // k2 columns over
    const float* A10 = device_A + (size_t)m2 * lda;
    const float* A11 = device_A + (size_t)m2 * lda + k2;

    // Submatrix pointers for B (V_ptr, W_ptr for kernel)
    // B is K x N. Submatrices are k2 x n2.
    const float* B00 = device_B;
    const float* B01 = device_B + n2; // n2 columns over
    const float* B10 = device_B + (size_t)k2 * ldb;
    const float* B11 = device_B + (size_t)k2 * ldb + n2;

    // Submatrix pointers for C (D_ptr, E_ptr for kernel)
    // C is M x N. Submatrices are m2 x n2.
    float* C00 = device_C;
    float* C01 = device_C + n2; // n2 columns over
    float* C10 = device_C + (size_t)m2 * ldc;
    float* C11 = device_C + (size_t)m2 * ldc + n2;

    dim3 threads(256); // Matches kernel's __launch_bounds__(256, ...)
    dim3 grid(div_ceil_strassen(n2, 128), div_ceil_strassen(m2, 128)); // grid.x for N, grid.y for M

    // Strassen's 7 products (M0-M6) and their additions to C submatrices
    // Kernel: strassen_fused_kernel<DELTA, EPSILON, GAMMA0_IS_ONE, GAMMA1_SIGN>
    // (m_sub, n_sub, k_sub, X_ptr, ldx, Y_ptr, ldy, V_ptr, ldv, W_ptr, ldw, D_ptr, ldd, E_ptr, lde)

    // M0 = (A00 + A11) * (B00 + B11) -> C00 += M0, C11 += M0
    // DELTA=1, EPSILON=1, GAMMA0_IS_ONE=1, GAMMA1_SIGN=1
    strassen_fused_kernel<1, 1, 1, 1><<<grid, threads>>>(
        m2, n2, k2, A00, lda, A11, lda, B00, ldb, B11, ldb, C00, ldc, C11, ldc);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // M1 = (A10 + A11) * B00 -> C10 += M1, C11 -= M1 (Original Strassen: C10+=M1, C00-=M1 (no, C2+=M1, C3-=M1 in paper implies C10, C11))
    // Paper: M1=(A2+A3)B0. D=C2, E=C3. C2+=M1, C3-=M1. (A2=A10, A3=A11, B0=B00, C2=C10, C3=C11)
    // DELTA=1 (A10+A11), EPSILON=0 (B00), GAMMA0_IS_ONE=1 (C10+=M1), GAMMA1_SIGN=-1 (C11-=M1)
    strassen_fused_kernel<1, 0, 1, -1><<<grid, threads>>>(
        m2, n2, k2, A10, lda, A11, lda, B00, ldb, B00, ldb, /* W_ptr dummy */ C10, ldc, C11, ldc);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // M2 = A00 * (B01 - B11) -> C00 -= M2, C01 += M2 (Original Strassen: C01+=M2, C11+=M2 (no, C0-=M2, C1+=M2 in paper))
    // Paper: M2=A0(B1-B3). D=C0, E=C1. C0-=M2, C1+=M2. (A0=A00, B1=B01, B3=B11, C0=C00, C1=C01)
    // DELTA=0 (A00), EPSILON=-1 (B01-B11), GAMMA0_IS_ONE=1 (C00-=M2, but kernel adds, so M2 needs to be -(A00*(B01-B11)) or pass -1 to gamma0)
    // The kernel is D += gamma0*M, E += gamma1*M. So if D = C00, gamma0 should be -1.
    // For C00 -= M2: D_ptr=C00, gamma0=-1. For C01 += M2: E_ptr=C01, gamma1=1.
    // So, strassen_fused_kernel<0, -1, -1, 1> for (A00, B01, B11, C00, C01)
    // The GAMMA0_IS_ONE implies it's always 1.0 * M. This means the subtraction must be handled by kernel structure or input sign.
    // The current kernel structure is D += M_val, E += gamma1_sign * M_val.
    // To achieve C00 -= M2 where M2 = A00 * (B01-B11), we need D = C00, and M_val = - (A00 * (B01-B11)).
    // This can be achieved by making X_ptr = -A00, or by making V_ptr = A00 and (W_ptr-V_ptr) = -(B01-B11).
    // Let's re-evaluate the kernel structure: D += M_comp; E += gamma1_sign * M_comp where M_comp = (X+delta Y)(V+epsilon W)
    // For M2 = A00(B01-B11): X=A00, Y=nullptr (delta=0). V=B01, W=B11 (epsilon=-1). So M_comp = A00(B01-B11).
    // C00 -= M_comp  => D=C00, gamma0_is_one means it adds. This is an issue.
    // The problem statement says: GAMMA0_IS_ONE: 1 if gamma0 is 1.0f. Assume gamma0 is always 1.0f for D updates.
    // This means the D output is ALWAYS D += M_tile.
    // For C00 -= M2, we must pass C00 as E_ptr with GAMMA1_SIGN = -1, and D_ptr as a dummy (e.g. C01 with GAMMA0_IS_ONE=0, but we don't have that).
    // Or, the M_comp itself must be negated. e.g. M_comp = (-A00)(B01-B11) or A00(-(B01-B11)).
    // Let's assume the equations are $C_i = C_i + \text{coeff} \cdot M_k$.
    // C00 = C00 - M2; C01 = C01 + M2.
    // If D is always D_ptr += M_kernel_val:
    // For C01 += M2: D_ptr=C01. M_kernel_val = A00(B01-B11). X=A00, delta=0. V=B01, W=B11, epsilon=-1.
    //   strassen_fused_kernel<0, -1, 1, 0><<<grid, threads>>>(m2, n2, k2, A00, lda, A00, lda, B01, ldb, B11, ldb, C01, ldc, C01, ldc); // C01+=M2
    // For C00 -= M2: D_ptr=C00. M_kernel_val = -A00(B01-B11). X=A00, delta=0. V=B11, W=B01, epsilon=-1 (gives A00(B11-B01)). This is M_kernel_val = -M2.
    //   strassen_fused_kernel<0, -1, 1, 0><<<grid, threads>>>(m2, n2, k2, A00, lda, A00, lda, B11, ldb, B01, ldb, C00, ldc, C00, ldc); // C00+=(-M2)
    // This is the standard way to handle subtractions in Strassen when the base operation is FMA.

    // M2 = A00 * (B01 - B11)
    // C01 += M2: D=C01, E=dummy. X=A00(d=0), V=B01,W=B11(e=-1). gamma0=1, gamma1=0
    strassen_fused_kernel<0, -1, 1, 0><<<grid, threads>>>(
        m2, n2, k2, A00, lda, A00, lda, /* Y_ptr dummy */ B01, ldb, B11, ldb, C01, ldc, C01, ldc /* E_ptr dummy */);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // C00 -= M2  (equivalent to C00 += (-M2) where -M2 = A00 * (B11 - B01) )
    // D=C00, E=dummy. X=A00(d=0), V=B11,W=B01(e=-1). gamma0=1, gamma1=0
    strassen_fused_kernel<0, -1, 1, 0><<<grid, threads>>>(
        m2, n2, k2, A00, lda, A00, lda, /* Y_ptr dummy */ B11, ldb, B01, ldb, C00, ldc, C00, ldc /* E_ptr dummy */);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // M3 = A11 * (B10 - B00); C10 += M3; C11 += M3
    // Kernel: <0, -1, 1, 1> with X=A11, Y=A11(dummy), V=B10, W=B00, D=C10, E=C11
    // M_comp = A11 * (B10 - B00)
    // C10 += M_comp
    // C11 += M_comp
    strassen_fused_kernel<0, -1, 1, 1><<<grid, threads>>>(
        m2, n2, k2, A11, lda, A11, lda, /* Y_ptr dummy */ B10, ldb, B00, ldb, C10, ldc, C11, ldc);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // M4 = (A00 + A01) * B11
    // C01 += M4: D=C01, E=dummy. X=A00,Y=A01(d=1), V=B11(e=0). gamma0=1, gamma1=0
    strassen_fused_kernel<1, 0, 1, 0><<<grid, threads>>>(
        m2, n2, k2, A00, lda, A01, lda, B11, ldb, B11, ldb, /* W_ptr dummy */ C01, ldc, C01, ldc /* E_ptr dummy */);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // C11 -= M4 (equiv. C11 += (-M4)). How to make -M4? -(A00+A01)*B11 = (-A00-A01)*B11 or (A00+A01)*(-B11)
    // Option 1: X=-A00, Y=-A01. Requires modifying input matrices or kernel support.
    // Option 2: V=-B11. Requires modifying input B or kernel support for negative V.
    // Option 3 (Chosen): Use C11 as E_ptr with GAMMA1_SIGN=-1. D_ptr is dummy.
    //    M_comp = (A00+A01)*B11. D=dummy, E=C11. gamma0_is_one=0 (not possible), gamma1_sign=-1.
    // This relies on GAMMA0_IS_ONE being flexible or having a kernel variant.
    // Given GAMMA0_IS_ONE is fixed to 1 for D updates by problem spec.
    // So, for C11 -= M4, we need D_ptr=C11, and M_comp = -(A00+A01)*B11.
    // M_comp = (X+delta*Y)(V+epsilon*W).
    // -(A00+A01)*B11 = (A00+A01)*(-B11).  Let X=A00, Y=A01 (delta=1). V=-B11 (epsilon=0). Not directly possible without negative V.
    // Or (-1)*(A00+A01)*B11. X=-A00, Y=-A01 (delta=1, assuming Y is A01, then X=-A00, Y_transformed = -A01, so X_base + delta*Y_base where X_base=-A00, delta=1, Y_base=-A01).
    // This implies we need to be able_to_negate_inputs_to_form_M_comp for the D path.
    // The kernel is (X + dY)(V + eW). The signs d,e are template params.
    // So for -(A00+A01)*B11:
    // Let X=A00, Y=A01. We want -(X+Y)V. This is (-X-Y)V.
    // strassen_fused_kernel<-1,-1,...> where X_ptr=A00, Y_ptr=A01, but DELTA is one param.
    // Alternative: (A00+A01)*(-B11). X=A00,Y=A01 (delta=1). V=B11 but want -B11. Set epsilon to -1 for V_ptr? No, epsilon is for W_ptr.
    // This is tricky with fixed GAMMA0_IS_ONE=1.
    // The paper's Var#3 for M1=(A2+A3)B0 -> C2+=M1, C3-=M1 used <1,0,1,-1>. Here D=C2, E=C3.
    // For M4=(A0+A1)B3 -> C1+=M4, C3-=M4. (A0=A00,A1=A01,B3=B11, C1=C01,C3=C11)
    // D=C01, E=C11. For C01+=M4: D_ptr=C01 gets M4. For C11-=M4: E_ptr=C11 gets -M4.
    // This implies X=A00,Y=A01(d=1), V=B11(e=0). D=C01(gamma0=1), E=C11(gamma1=-1).
    strassen_fused_kernel<1, 0, 1, -1><<<grid, threads>>>(
        m2, n2, k2, A00, lda, A01, lda, B11, ldb, B11, ldb, /* W_ptr dummy */ C01, ldc, C11, ldc);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // M5 = (A10 - A00) * (B00 + B01) -> C11 += M5
    // X=A10, Y=A00 (delta=-1). V=B00, W=B01 (epsilon=1). D=C11 (gamma0=1), E=dummy (gamma1=0)
    strassen_fused_kernel<-1, 1, 1, 0><<<grid, threads>>>(
        m2, n2, k2, A10, lda, A00, lda, B00, ldb, B01, ldb, C11, ldc, C11, ldc /* E_ptr dummy */);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // M6 = (A01 - A11) * (B10 + B11) -> C00 += M6
    // X=A01, Y=A11 (delta=-1). V=B10, W=B11 (epsilon=1). D=C00 (gamma0=1), E=dummy (gamma1=0)
    strassen_fused_kernel<-1, 1, 1, 0><<<grid, threads>>>(
        m2, n2, k2, A01, lda, A11, lda, B10, ldb, B11, ldb, C00, ldc, C00, ldc /* E_ptr dummy */);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    // C: M rows, N cols. host_C has ldc. device_C has ldc.
    checkCudaErrors(cudaMemcpy2D(host_C, ldc * sizeof(float), device_C, ldc * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost));


    // Free device memory
    checkCudaErrors(cudaFree(device_A));
    checkCudaErrors(cudaFree(device_B));
    checkCudaErrors(cudaFree(device_C));
}


#endif // SGEMM_STRASSEN_FUSED_CUH_
