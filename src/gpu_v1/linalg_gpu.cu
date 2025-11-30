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
    double* AB_d; // Basis Matrix
    double* b_d; // RHS Workspace (Solver writes solution x here)

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
    cudaMalloc((void**)&ws.AB_d, n * n * sizeof(double));
    cudaMalloc((void**)&ws.b_d, n * sizeof(double));
    cudaMalloc((void**)&ws.ipiv_d, n * sizeof(int));
    cudaMalloc((void**)&ws.info_d, sizeof(int));

    // Allocate Helpers
    cudaMalloc((void**)&ws.B_d, n * sizeof(int));
    cudaMalloc((void**)&ws.y_temp_d, n * sizeof(double));

    cusolverDnDgetrf_bufferSize(ws.cusolve_handle, n, n, ws.AB_d, n, &ws.work_size);
    cudaMalloc((void**)&ws.work_d, ws.work_size * sizeof(double));

    ws.initialized = true;
}

void destroy_gpu_workspace() {
    if (!ws.initialized)
        return;

    cudaFree(ws.AB_d);
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

// Solves A_Basis * d = A_column[col_idx]
void gpu_solve_from_resident_col(int m, int col_idx, double* d_out) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // 1. Calculate the address of the column in GPU memory
    // Address = Start_of_Matrix + (Column_Index * Number_of_Rows)
    double* d_col_ptr = ws.A_full_d + (static_cast<size_t>(col_idx) * m);

    // 2. Copy that column into the solver's RHS buffer (b_d)
    // This happens entirely inside the GPU's memory.
    cudaMemcpy(ws.b_d, d_col_ptr, m * sizeof(double), cudaMemcpyDeviceToDevice);

    // 3. Solve using the existing factorization
    // Solves: A_B * x = b_d
    cusolverDnDgetrs(ws.cusolve_handle, CUBLAS_OP_N, m, 1, ws.AB_d, m, ws.ipiv_d, ws.b_d, m,
                     ws.info_d);

    // 4. Copy the result back to the CPU
    cudaMemcpy(d_out, ws.b_d, m * sizeof(double), cudaMemcpyDeviceToHost);
}

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

void gpu_build_basis_and_factorize(int m, int n_total, const int* B_indices) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // 1. Copy indices to GPU (tiny copy: m integers)
    cudaMemcpy(ws.B_d, B_indices, m * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Launch Gather Kernel
    int total_elements = m * m;
    int threadsPerBlock = 256; // 8 Warps (I think it's the sweet spot)
    // We round up, we want more threads than total elements (that's why we have guard in Kernel)
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    gather_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, ws.A_full_d, ws.AB_d, ws.B_d);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Kernel Error: " << cudaGetErrorString(err) << std::endl;

    // 3. Factorize (Standard LU)
    // AB_d is now filled with the correct columns.
    cusolverDnDgetrf(ws.cusolve_handle, m, m, ws.AB_d, m, ws.work_d, ws.ipiv_d, ws.info_d);
}

// Overwrite the storage with new perturbed values
void gpu_update_rhs_storage(int m, const double* b_new) {
    cudaMemcpy(ws.b_storage_d, b_new, m * sizeof(double), cudaMemcpyHostToDevice);
}

void gpu_solve_from_persistent_b(int m, double* x_out) {
    // 1. Refresh the workspace: Copy Storage -> Workspace
    // We do this because the previous solve destroyed ws.b_d
    cudaMemcpy(ws.b_d, ws.b_storage_d, m * sizeof(double), cudaMemcpyDeviceToDevice);

    // 2. Solve (ws.b_d becomes x)
    cusolverDnDgetrs(ws.cusolve_handle, CUBLAS_OP_N, m, 1, ws.AB_d, m, ws.ipiv_d, ws.b_d, m,
                     ws.info_d);

    // 3. Return result
    cudaMemcpy(x_out, ws.b_d, m * sizeof(double), cudaMemcpyDeviceToHost);
}

void gpu_solve_prefactored(int n, const double* b, double* x, bool transpose) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // Copy b to device
    cudaMemcpy(ws.b_d, b, n * sizeof(double), cudaMemcpyHostToDevice);

    // Select Operation: N (Normal) or T (Transpose)
    cublasOperation_t op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Solve using the existing factorization in ws.A_d and ws.ipiv_d
    cusolverDnDgetrs(ws.cusolve_handle, op, n, 1, ws.AB_d, n, ws.ipiv_d, ws.b_d, n, ws.info_d);

    // Copy result back
    cudaMemcpy(x, ws.b_d, n * sizeof(double), cudaMemcpyDeviceToHost);
}
