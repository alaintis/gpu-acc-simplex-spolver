#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "linalg_gpu.hpp"
#include "logging.hpp"

struct CPUWorkspace {
    // Buffers
    double* B_inv_d; // Basis Inverse Matrix (m * m)
    double* b_d; // RHS Workspace (Solver writes solution x here)
    double* inv_temp_d; // Temp buffer for Identity/Inverse calculation (m * m)

    // LU decomposition Buffers
    int* ipiv_d; // Pivot Array for LU decomposition
    int* info_d; // Error Flag for LU decomposition
    double* work_d; // Scratchpad for LU decomposition
    int work_size; // Scratchpad space for LU decomposition

    // Persistent storage for the full problem (Read-Only)
    double* A_full_d = nullptr; // Size: m * n_total
    double* b_storage_d = nullptr; // Size: m
    double* c_full_d = nullptr; // Size: n_total

    double* sn_d = nullptr; // Size: n_total (Reduced costs on Device)
    double* sn_h = nullptr; // Size: n_total (Reduced costs on Host)
    double* y_temp_d = nullptr; // Size: m (Temp storage for y during pricing)
    int* B_d = nullptr; // Size: n (Buffer for Basis Indices)

    // Handlers
    cusolverDnHandle_t cusolve_handle;
    cublasHandle_t cublas_handle;

    bool initialized = false;
};

static CPUWorkspace ws;

void init_gpu_workspace(int n) {
    if (ws.initialized)
        return;

    cusolverDnCreate(&ws.cusolve_handle);
    cublasCreate(&ws.cublas_handle);

    // Allocate Workspace (Scratchpads)
    cudaMalloc((void**)&ws.B_inv_d, n * n * sizeof(double));
    cudaMalloc((void**)&ws.inv_temp_d, n * n * sizeof(double));
    cudaMalloc((void**)&ws.b_d, n * sizeof(double));
    cudaMalloc((void**)&ws.ipiv_d, n * sizeof(int));
    cudaMalloc((void**)&ws.info_d, sizeof(int));

    // Allocate Helpers
    cudaMalloc((void**)&ws.B_d, n * sizeof(int));
    cudaMalloc((void**)&ws.y_temp_d, n * sizeof(double));

    cusolverDnDgetrf_bufferSize(ws.cusolve_handle, n, n, ws.B_inv_d, n, &ws.work_size);
    cudaMalloc((void**)&ws.work_d, ws.work_size * sizeof(double));

    ws.initialized = true;
}

void destroy_gpu_workspace() {
    if (!ws.initialized)
        return;

    cudaFree(ws.B_inv_d);
    cudaFree(ws.inv_temp_d);
    cudaFree(ws.b_d);
    cudaFree(ws.ipiv_d);
    cudaFree(ws.info_d);
    cudaFree(ws.work_d);
    cusolverDnDestroy(ws.cusolve_handle);

    if (ws.A_full_d)
        cudaFree(ws.A_full_d);
    if (ws.b_storage_d)
        cudaFree(ws.b_storage_d);
    if (ws.c_full_d)
        cudaFree(ws.c_full_d);
    if (ws.sn_d)
        cudaFree(ws.sn_d);
    if (ws.y_temp_d)
        cudaFree(ws.y_temp_d);
    if (ws.sn_h)
        free(ws.sn_h);
    if (ws.B_d)
        cudaFree(ws.B_d);

    cublasDestroy(ws.cublas_handle);
    ws.initialized = false;
}

void gpu_load_problem(int m, int n_total, const double* A_flat, const double* b, const double* c) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // Allocate persistent memory
    cudaMalloc((void**)&ws.A_full_d, n_total * m * sizeof(double));
    cudaMalloc((void**)&ws.c_full_d, n_total * sizeof(double));
    cudaMalloc((void**)&ws.b_storage_d, m * sizeof(double));

    // Allocate Pricing buffers
    cudaMalloc((void**)&ws.sn_d, n_total * sizeof(double));
    ws.sn_h = (double*)malloc(n_total * sizeof(double));

    // Copy data
    cudaMemcpy(ws.A_full_d, A_flat, n_total * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.c_full_d, c, n_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.b_storage_d, b, m * sizeof(double), cudaMemcpyHostToDevice);
}

// ============================================================
// KERNELS
// ============================================================

// Assembles Basis Matrix, spawns tons of threads (everything on gpu, all at once)
__global__ void gather_kernel(int m, const double* A_full, double* A_basis, const int* B_indices) {
    // Calculate unique Thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total elements in basis matrix = m * m (Guard)
    if (idx >= m * m)
        return;

    // Map linear index to (row, col) in the Basis Matrix (Target)
    // Since it's Column Major:
    // idx = col * m + row  =>  col = idx / m, row = idx % m
    int row = idx % m;
    int col_basis = idx / m;

    // Find which original column corresponds to this basis column
    int col_original = B_indices[col_basis];

    // Copy the value
    // Target: A_basis[col_basis][row] (which is just A_basis[idx])
    // Source: A_full[col_original][row]
    A_basis[idx] = A_full[col_original * m + row];
}

// Initialize matrix to Identity (for inversion)
__global__ void set_identity_kernel(int m, double* Matrix) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)m * m)
        return;

    int row = idx % m;
    int col = idx / m;

    Matrix[idx] = (row == col) ? 1.0 : 0.0;
}

// Sherman-Morrison Update Kernel
// Inputs:
//   B_inv: The current Inverse Matrix
//   d:     The direction vector (B^-1 * a_entering)
//   pivot_row: The index of the row leaving the basis (j)
__global__ void update_inverse_kernel(int m,
                                      double* __restrict__ B_inv,
                                      const double* __restrict__ d,
                                      int pivot_row) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)m * m)
        return;

    int row = idx % m;
    int col = idx / m;

    // Get the pivot element from the direction vector
    // All threads read the same d[pivot_row], GPU cache handles this broadcast well
    double pivot_val = d[pivot_row];

    // Read the current value at this thread's position
    double b_val = B_inv[idx];

    // Read the element of the Pivot Row corresponding to this column
    double b_pivot_row_val = B_inv[(size_t)col * m + pivot_row];

    // Calculate the new value for the Pivot Row
    double new_pivot_row_val = b_pivot_row_val / pivot_val;

    if (row == pivot_row) {
        // If this thread is in the pivot row, update it directly
        B_inv[idx] = new_pivot_row_val;
    } else {
        // If this thread is in any other row, apply the elimination step
        double d_val = d[row];
        B_inv[idx] = b_val - (d_val * new_pivot_row_val);
    }
}

// ============================================================
// LOGIC
// ============================================================

// Function: Full Refactorization (slow and stable path)
// 1. Rebuilds the basis matrix B from scratch.
// 2. Factorizes it (LU).
// 3. Computes the explicit inverse B^-1.
void gpu_build_basis_and_invert(int m, int n_total, const int* B_indices) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // 1. Copy indices to GPU (tiny copy: m integers)
    cudaMemcpy(ws.B_d, B_indices, m * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Launch Gather Kernel: Gather Basis Columns into B_inv_d
    size_t total_elements = (size_t)m * m;
    int threadsPerBlock = 256; // 8 Warps (I think it's the sweet spot)
    // We round up, we want more threads than total elements (that's why we have guard in Kernel)
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    gather_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, ws.A_full_d, ws.B_inv_d, ws.B_d);

    // 3. LU Factorization (in-place on B_inv_d)
    cusolverDnDgetrf(ws.cusolve_handle, m, m, ws.B_inv_d, m, ws.work_d, ws.ipiv_d, ws.info_d);

    // 4. Inversion via Solve: B_inv_d * X = I  => X = B_inv_d^-1
    // 4a. Initialize temporary buffer to Identity
    set_identity_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, ws.inv_temp_d);

    // 4b. Solve for Identity. The result X is stored in inv_temp_d.
    cusolverDnDgetrs(ws.cusolve_handle, CUBLAS_OP_N, m, m, ws.B_inv_d, m, ws.ipiv_d, ws.inv_temp_d,
                     m, ws.info_d);

    // 4c. Copy result (Inverse) back to B_inv_d to be used as the Basis Inverse
    cudaMemcpy(ws.B_inv_d, ws.inv_temp_d, m * m * sizeof(double), cudaMemcpyDeviceToDevice);
}

// Uses the Sherman-Morrison kernel to update AB_d in-place.
void gpu_update_basis_fast(int m, int pivot_row) {
    size_t total_elements = (size_t)m * m;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    // This relies on 'ws.b_d' containing the direction vector 'd'!
    // This 'd' must have been computed in the previous step (gpu_calc_direction).
    update_inverse_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, ws.B_inv_d, ws.b_d, pivot_row);
}

// Computes d = B^-1 * A_j using Matrix-Vector Multiplication.
void gpu_calc_direction(int m, int col_idx, double* d_out) {
    // d = B^-1 * A_col
    double* A_col_ptr = ws.A_full_d + (static_cast<size_t>(col_idx) * m);
    double alpha = 1.0;
    double beta = 0.0;

    // cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    // cublasDgemv: y = alpha * A * x + beta * y
    // We compute: ws.b_d = 1.0 * AB_d * A_col_ptr
    // B_inv_d is B^-1, so this computes d = B^-1 * A_j
    cublasDgemv(ws.cublas_handle, CUBLAS_OP_N, m, m, &alpha, ws.B_inv_d, m, A_col_ptr, 1, &beta,
                ws.b_d, 1);

    // Copy result back to CPU for the Ratio Test for now...
    cudaMemcpy(d_out, ws.b_d, m * sizeof(double), cudaMemcpyDeviceToHost);
}

void gpu_mult_inverse(int m, const double* b, double* x, bool transpose) {
    cudaMemcpy(ws.b_d, b, m * sizeof(double), cudaMemcpyHostToDevice);

    double alpha = 1.0;
    double beta = 0.0;
    // Select Transpose (for Duals y = B^-T * c) or Normal (for x = B^-1 * b)
    cublasOperation_t op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasDgemv(ws.cublas_handle, op, m, m, &alpha, ws.B_inv_d, m, ws.b_d, 1, &beta, ws.y_temp_d,
                1);

    cudaMemcpy(x, ws.y_temp_d, m * sizeof(double), cudaMemcpyDeviceToHost);
}

void gpu_recalc_x_from_persistent_b(int m, double* x_out) {
    double alpha = 1.0;
    double beta = 0.0;
    // Use the persistent b vector stored in VRAM
    cublasDgemv(ws.cublas_handle, CUBLAS_OP_N, m, m, &alpha, ws.B_inv_d, m, ws.b_storage_d, 1,
                &beta, ws.b_d, 1);

    cudaMemcpy(x_out, ws.b_d, m * sizeof(double), cudaMemcpyDeviceToHost);
}

// This computes sn = c - A^T * y
// (we do this for the whole A, I think it's faster than building the non-basic Matrix)
const double* gpu_compute_reduced_costs(int m, int n_total, const double* y_host) {
    // 1. Copy y to device (small copy, size m)
    cudaMemcpy(ws.y_temp_d, y_host, m * sizeof(double), cudaMemcpyHostToDevice);

    // 2. Calculate A^T * y
    // Initialize sn with c
    cudaMemcpy(ws.sn_d, ws.c_full_d, n_total * sizeof(double), cudaMemcpyDeviceToDevice);

    double alpha = -1.0;
    double beta = 1.0;

    // cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    // We compute: sn = -1.0 * (A^T * y) + 1.0 * sn
    cublasDgemv(ws.cublas_handle,
                CUBLAS_OP_T, // Transpose A because we want dot product of cols with y
                m, n_total, &alpha, ws.A_full_d, m, // LDA is m
                ws.y_temp_d, 1, &beta, ws.sn_d, 1); // incx & incy = 1 are strides

    // 3. Copy result back to host
    cudaMemcpy(ws.sn_h, ws.sn_d, n_total * sizeof(double), cudaMemcpyDeviceToHost);

    return ws.sn_h;
}

// Helper: Update the stored RHS b (used when applying perturbation)
void gpu_update_rhs_storage(int m, const double* b_new) {
    cudaMemcpy(ws.b_storage_d, b_new, m * sizeof(double), cudaMemcpyHostToDevice);
}