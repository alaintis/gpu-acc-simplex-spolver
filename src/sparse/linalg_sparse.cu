#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cudss.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cusparse_v2.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "linalg_sparse.hpp"
#include "logging.hpp"

#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            exit(1);\
        } \
    } while(0);

struct SparseWorkspace {
    // Buffers
    double* AB_d; // Basis Matrix
    double* b_d; // RHS Workspace (Solver writes solution x here)
    double* x_d;

    // LU decomposition Buffers
    int* ipiv_d; // Pivot Array for LU decomposition
    int* info_d; // Error Flag for LU decomposition
    double* work_d; // Scratchpad for LU decomposition
    int work_size; // Scratchpad space for LU decomposition

    // Buffer for sparse matrices.
    int *AT_row_ptr  = nullptr;
    int *AT_col_idx  = nullptr;
    double *AT_value = nullptr;

    // Sparse matrices pointers.
    int *ABT_row_ptr = nullptr;
    int *ABT_col_idx = nullptr;
    double *ABT_value   = nullptr;
    cudssMatrix_t ABT;
    cudssMatrix_t x_cudss;
    cudssMatrix_t b_cudss;

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
    cudssHandle_t  cudss_handle;
    cusparseHandle_t  cusparse_handle;

    cudssConfig_t cudss_config;
    cudssData_t cudss_data;

    bool initialized = false;
};

static SparseWorkspace ws;

void init_sparse_workspace(int m) {
    if (ws.initialized)
        return;

    cusolverDnCreate(&ws.cusolve_handle);
    cublasCreate(&ws.cublas_handle);
    cudssCreate(&ws.cudss_handle);
    cusparseCreate(&ws.cusparse_handle);

    cudssConfigCreate(&ws.cudss_config);
    cudssDataCreate(ws.cudss_handle, &ws.cudss_data);

    int ir_steps = 1;
    cudssConfigSet(ws.cudss_config, CUDSS_CONFIG_IR_N_STEPS, &ir_steps, sizeof(ir_steps));

    // Allocate Workspace (Scratchpads)
    cudaMalloc((void**)&ws.AB_d, m * m * sizeof(double));
    cudaMalloc((void**)&ws.b_d, m * sizeof(double));
    cudaMalloc((void**)&ws.x_d, m * sizeof(double));
    cudaMalloc((void**)&ws.ipiv_d, m * sizeof(int));
    cudaMalloc((void**)&ws.info_d, sizeof(int));
    
    // Sparse Workspace
    cudaMalloc((void**)&ws.ABT_row_ptr, (m+1) * sizeof(int));
    
    // Allocate Helpers
    cudaMalloc((void**)&ws.B_d, m * sizeof(int));
    cudaMalloc((void**)&ws.y_temp_d, m * sizeof(double));

    cusolverDnDgetrf_bufferSize(ws.cusolve_handle, m, m, ws.AB_d, m, &ws.work_size);
    cudaMalloc((void**)&ws.work_d, ws.work_size * sizeof(double));

    ws.initialized = true;
}

void destroy_sparse_workspace() {
    if (!ws.initialized)
        return;

    cudaFree(ws.AB_d);
    cudaFree(ws.b_d);
    cudaFree(ws.x_d);
    cudaFree(ws.ipiv_d);
    cudaFree(ws.info_d);
    cudaFree(ws.work_d);

    cudaFree(ws.ABT_row_ptr);

    cusolverDnDestroy(ws.cusolve_handle);
    cublasDestroy(ws.cublas_handle);
    cudssDataDestroy(ws.cudss_handle, ws.cudss_data);
    cudssConfigDestroy(ws.cudss_config);
    cudssDestroy(ws.cudss_handle);
    cudssMatrixDestroy(ws.ABT);
    cudssMatrixDestroy(ws.x_cudss);
    cudssMatrixDestroy(ws.b_cudss);

    cusparseDestroy(ws.cusparse_handle);

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
    
    if (ws.AT_row_ptr) {
        cudaFree(ws.AT_row_ptr);
        cudaFree(ws.AT_col_idx);
        cudaFree(ws.AT_value);
        cudaFree(ws.ABT_col_idx);
        cudaFree(ws.ABT_value);
    }

    ws.initialized = false;
}

void sparse_load_problem(int m, int n_total, const double* A_flat, const double* b, const double* c) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // Build a sparse CSR matrix
    vector<int> row_ptr;
    vector<int> col_idx;
    vec values;

    row_ptr.push_back(0); // First row always starts at 0
    int nnz = 0;

    // We iterate row-by-row and only store non-zeroes
    for (int i = 0; i < n_total; i++) { // For each column i
        for (int j = 0; j < m; j++) {   // For each row j
            double val = A_flat[i * m + j];  // Get A(j, i) from flat input

            // Only store non-zero values
            if (std::abs(val) > 1e-12) {
                values.push_back(val);
                col_idx.push_back(j);
                nnz++;
            }
        }
        row_ptr.push_back(nnz);
    }

    // Allocate persistent memory for csr AT matrix.
    cudaMalloc((void**)&ws.AT_row_ptr, (n_total+1) * sizeof(int));
    cudaMalloc((void**)&ws.AT_col_idx, nnz * sizeof(int));
    cudaMalloc((void**)&ws.AT_value, nnz * sizeof(double));

    cudaMemcpy(ws.AT_row_ptr, row_ptr.data(), (n_total+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.AT_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.AT_value, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&ws.ABT_col_idx, nnz * sizeof(int));
    cudaMalloc((void**)&ws.ABT_value, nnz * sizeof(double));
    
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
const double* sparse_compute_reduced_costs(int m, int n_total, const double* y_host) {
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
void sparse_solve_from_resident_col(int m, int col_idx, double* d_out) {
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
__global__ void sp_gather_kernel(int m, const double* A_full, double* A_basis, const int* B_indices) {
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

__global__ void set_pointer_kernel(int m, const int *A_row_ptr, const int *B, int *AB_row_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m) {   
        AB_row_ptr[idx] = A_row_ptr[B[idx]+1] - A_row_ptr[B[idx]];
    }
}

/**
 * To fill a matrix AB by selecting a set of rows from A using B, we first calculate the prefix sum over the deltas to get the row pointers.
 * Now the job is just to find the right indices to fill the matrix with.
 */
__global__ void fill_AB_kernel(int m, const int *A_row_ptr, const int *A_col_idx, const double *A_value, const int *B, const int *AB_row_ptr, int *AB_col_idx, double *AB_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int row = idx / 32;
    if(row >= m) return;

    int A_start = A_row_ptr[B[row]];
    int A_end   = A_row_ptr[B[row]+1];
    int cols = A_end-A_start;

    int AB_start = AB_row_ptr[row];
    for(int i = idx % 32; i < cols; i+=32) {
        AB_col_idx[AB_start + i] = A_col_idx[A_start + i];
        AB_value[AB_start + i] = A_value[A_start + i];
    } 
}

void sparse_build_basis_and_factorize(int m, int n_total, const int* B_indices) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // 1. Copy indices to GPU (tiny copy: m integers)
    cudaMemcpy(ws.B_d, B_indices, m * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Launch Gather Kernel
    int total_elements = m * m;
    int threadsPerBlock = 256; // 8 Warps (I think it's the sweet spot)
    // We round up, we want more threads than total elements (that's why we have guard in Kernel)
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    sp_gather_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, ws.A_full_d, ws.AB_d, ws.B_d);
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Kernel Error: " << cudaGetErrorString(err) << std::endl;

    // 3. Factorize (Standard LU)
    // AB_d is now filled with the correct columns.
    cusolverDnDgetrf(ws.cusolve_handle, m, m, ws.AB_d, m, ws.work_d, ws.ipiv_d, ws.info_d);

    // 4. Redefine ABT
    int tpb = 512;
    int blocks = (m + tpb - 1) / tpb;
    set_pointer_kernel<<<blocks, tpb>>>(m, ws.AT_row_ptr, ws.B_d, ws.ABT_row_ptr);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, ws.ABT_row_ptr, &ws.ABT_row_ptr[m+1], ws.ABT_row_ptr);

    int nnz = 0;
    cudaMemcpy(&nnz, &ws.ABT_row_ptr[m], sizeof(int), cudaMemcpyDeviceToHost);

    // Fill the kernel.
    int fill_tpb = 512;
    int fill_blocks = (32 * m + tpb - 1) / tpb;
    fill_AB_kernel<<<fill_blocks, fill_tpb>>>(m, ws.AT_row_ptr, ws.AT_col_idx, ws.AT_value, ws.B_d, ws.ABT_row_ptr, ws.ABT_col_idx, ws.ABT_value);
    cudaDeviceSynchronize();

    // std::cout << "NNZ | " << nnz << std::endl;

    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&ws.ABT, m, m, nnz, ws.ABT_row_ptr, NULL,
                         ws.ABT_col_idx, ws.ABT_value, CUDA_R_32I, CUDA_R_64F, mtype, mview,
                         base), status, "CSR Matrix");
    cudssMatrixCreateDn(&ws.x_cudss, m, 1, m, ws.x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&ws.b_cudss, m, 1, m, ws.b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);


    // cusparseSpMatDescr_t ABT;
    // cusparseCreateCsr(&ABT, m, m, nnz, ws.ABT_row_ptr, ws.ABT_col_idx, ws.ABT_value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    csrilu02Info_t info;
    cusparseCreateCsrilu02Info(&info);

    cusparseMatDescr_t ABT_descr;
    cusparseCreateMatDescr(&ABT_descr);
    cusparseSetMatType(ABT_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(ABT_descr, CUSPARSE_INDEX_BASE_ZERO);

    int buffer_size;
    void* buffer;
    
    // set buffer size
    cusparseDcsrilu02_bufferSize(ws.cusparse_handle, m, nnz, ABT_descr, ws.ABT_value, ws.ABT_row_ptr, ws.ABT_col_idx, info, &buffer_size);
    cudaMalloc(&buffer, buffer_size);

    cusparseDcsrilu02_analysis(ws.cusparse_handle, m, nnz, ABT_descr, ws.ABT_value, ws.ABT_row_ptr, ws.ABT_col_idx, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer);
    int structural_zero;
    cusparseStatus_t sparse_status = cusparseXcsrilu02_zeroPivot(ws.cusparse_handle, info, &structural_zero);
    if (sparse_status == CUSPARSE_STATUS_ZERO_PIVOT) {
        // printf("Warning: structural zero at (%d,%d)\n", structural_zero, structural_zero);
    }

    cusparseDcsrilu02(
        ws.cusparse_handle, m, nnz,
        ABT_descr, ws.ABT_value, ws.ABT_row_ptr, ws.ABT_col_idx,
        info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer
    );
    
    int numerical_zero;
    // Check for numerical zeros
    sparse_status = cusparseXcsrilu02_zeroPivot(ws.cusparse_handle, info, &numerical_zero);
    if (sparse_status == CUSPARSE_STATUS_ZERO_PIVOT) {
        // printf("Warning: numerical zero at (%d,%d)\n", numerical_zero, numerical_zero);
    }
    
    cudaFree(buffer);
    cusparseDestroyCsrilu02Info(info);
    cusparseDestroyMatDescr(ABT_descr);


    // CUDSS_CALL_AND_CHECK(cudssExecute(ws.cudss_handle, CUDSS_PHASE_ANALYSIS, ws.cudss_config, ws.cudss_data, ws.ABT, ws.x_cudss, ws.b_cudss), status, "ANAL");
    // CUDSS_CALL_AND_CHECK(cudssExecute(ws.cudss_handle, CUDSS_PHASE_FACTORIZATION, ws.cudss_config, ws.cudss_data, ws.ABT, ws.x_cudss, ws.b_cudss), status, "FACT");
}

// Overwrite the storage with new perturbed values
void sparse_update_rhs_storage(int m, const double* b_new) {
    cudaMemcpy(ws.b_storage_d, b_new, m * sizeof(double), cudaMemcpyHostToDevice);
}

void sparse_solve_from_persistent_b(int m, double* x_out) {
    // 1. Refresh the workspace: Copy Storage -> Workspace
    // We do this because the previous solve destroyed ws.b_d
    cudaMemcpy(ws.b_d, ws.b_storage_d, m * sizeof(double), cudaMemcpyDeviceToDevice);

    // 2. Solve (ws.b_d becomes x)
    cusolverDnDgetrs(ws.cusolve_handle, CUBLAS_OP_N, m, 1, ws.AB_d, m, ws.ipiv_d, ws.b_d, m,
                     ws.info_d);

    // 3. Return result
    cudaMemcpy(x_out, ws.b_d, m * sizeof(double), cudaMemcpyDeviceToHost);
}

void sparse_solve_prefactored(int n, const double* b, double* x, bool transpose) {
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

void sparse_solve_ABT(int n, const double* b, double* x) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");
    
    // Copy b to device
    cudaMemcpy(ws.b_d, b, n * sizeof(double), cudaMemcpyHostToDevice);

    // Solve using the existing factorization.
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    // CUDSS_CALL_AND_CHECK(cudssExecute(ws.cudss_handle, CUDSS_PHASE_SOLVE, ws.cudss_config, ws.cudss_data, ws.ABT, ws.x_cudss, ws.b_cudss), status, "SOLVE");
    // cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(x, ws.x_d, n * sizeof(double), cudaMemcpyDeviceToHost);
}
