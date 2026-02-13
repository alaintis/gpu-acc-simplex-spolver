#include <assert.h>
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/permutation_iterator.h>
#include <vector>

#include "linalg_gpu.hpp"
#include "logging.hpp"

struct CPUWorkspace {
    // Buffers
    double* B_inv_d; // Basis Inverse Matrix (m * m)
    double* b_d; // Temp Workspace: Holds direction 'd' or intermediate vectors
    double* inv_temp_d; // Temp buffer for Identity/Inverse calculation (m * m)
    double* x_B_d; // Solution Vector for Basis Variables (m)
    double* c_B_d; // Dedicated Persistent buffer for Basis Costs (m)
    double* current_col_d; // Temp buffer to hold the extracted column A_j (m)

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
    double* y_d = nullptr; // Size: m (Storage for y)
    int* B_d = nullptr; // Size: m (Buffer for Basis Indices)
    int* N_d = nullptr; // Non-Basic Indices on GPU (n)

    // GPU State Variables
    int* entering_idx_d = nullptr; // j_i (Index in N)
    int* entering_col_d = nullptr; // jj (Actual Column Index)
    int* leaving_idx_d = nullptr; // r (Index in B)

    // Mapped Optimality Flag
    int* optimality_flag_h = nullptr; // Host pointer (mapped)
    int* optimality_flag_host_shadow_d = nullptr; // Host shadow pointer (for syncing)
    int* optimality_flag_d = nullptr; // VRAM pointer (fast)

    // CUB Reduction Buffers
    void* cub_temp_d = nullptr;
    size_t cub_temp_size_d = 0;
    cub::KeyValuePair<int, double>* pricing_out_d = nullptr;

    // Harris Pivoting Result Buffers
    cub::KeyValuePair<int, double>* h_harris_result = nullptr;
    cub::KeyValuePair<int, double>* d_harris_result_mapped = nullptr;

    // Handlers
    cusolverDnHandle_t cusolve_handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;

    bool initialized = false;
};

static CPUWorkspace ws;

void init_gpu_workspace(int n) {
    if (ws.initialized)
        return;

    cudaSetDeviceFlags(cudaDeviceMapHost);

    cusolverDnCreate(&ws.cusolve_handle);
    cublasCreate(&ws.cublas_handle);

    // Allocate Workspace (Scratchpads)
    cudaMalloc((void**)&ws.B_inv_d, n * n * sizeof(double));
    cudaMalloc((void**)&ws.inv_temp_d, n * n * sizeof(double));
    cudaMalloc((void**)&ws.b_d, n * sizeof(double));
    cudaMalloc((void**)&ws.x_B_d, n * sizeof(double));
    cudaMalloc((void**)&ws.c_B_d, n * sizeof(double));
    cudaMalloc((void**)&ws.current_col_d, n * sizeof(double));
    cudaMalloc((void**)&ws.ipiv_d, n * sizeof(int));
    cudaMalloc((void**)&ws.info_d, sizeof(int));

    // Allocate Helpers
    cudaMalloc((void**)&ws.B_d, n * sizeof(int));
    cudaMalloc((void**)&ws.y_d, n * sizeof(double));

    // State Variables
    cudaMalloc((void**)&ws.entering_idx_d, sizeof(int));
    cudaMalloc((void**)&ws.entering_col_d, sizeof(int));
    cudaMalloc((void**)&ws.leaving_idx_d, sizeof(int));

    // Mapped Optimality Flag
    // Allocate VRAM flag (Fast GPU access)
    cudaMalloc((void**)&ws.optimality_flag_d, sizeof(int));
    cudaMemset(ws.optimality_flag_d, 0, sizeof(int));
    // Allocate Mapped Host flag (CPU access)
    cudaHostAlloc((void**)&ws.optimality_flag_h, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&ws.optimality_flag_host_shadow_d, ws.optimality_flag_h, 0);
    *ws.optimality_flag_h = 0;

    cusolverDnDgetrf_bufferSize(ws.cusolve_handle, n, n, ws.B_inv_d, n, &ws.work_size);
    cudaMalloc((void**)&ws.work_d, ws.work_size * sizeof(double));

    // CUB
    cudaMalloc((void**)&ws.pricing_out_d, sizeof(cub::KeyValuePair<int, double>));
    ws.cub_temp_size_d = 2 * 1024 * 1024; // Increase scratchpad just in case
    cudaMalloc((void**)&ws.cub_temp_d, ws.cub_temp_size_d);

    // Harris Test Buffers
    cudaHostAlloc((void**)&ws.h_harris_result, sizeof(cub::KeyValuePair<int, double>),
                  cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&ws.d_harris_result_mapped, ws.h_harris_result, 0);

    ws.initialized = true;
}

void destroy_gpu_workspace() {
    if (!ws.initialized)
        return;

// Helper macro to free CUDA memory
#define FREE_GPU(ptr)  \
    if (ptr) {         \
        cudaFree(ptr); \
        ptr = nullptr; \
    }
#define FREE_HOST(ptr)     \
    if (ptr) {             \
        cudaFreeHost(ptr); \
        ptr = nullptr;     \
    }

    FREE_GPU(ws.B_inv_d);
    FREE_GPU(ws.inv_temp_d);
    FREE_GPU(ws.b_d);
    FREE_GPU(ws.x_B_d);
    FREE_GPU(ws.c_B_d);
    FREE_GPU(ws.current_col_d);
    FREE_GPU(ws.ipiv_d);
    FREE_GPU(ws.info_d);
    FREE_GPU(ws.work_d);

    FREE_GPU(ws.A_full_d);
    FREE_GPU(ws.b_storage_d);
    FREE_GPU(ws.c_full_d);

    FREE_GPU(ws.sn_d);
    FREE_GPU(ws.y_d);
    FREE_GPU(ws.B_d);
    FREE_GPU(ws.N_d);

    FREE_GPU(ws.entering_idx_d);
    FREE_GPU(ws.entering_col_d);
    FREE_GPU(ws.leaving_idx_d);

    FREE_GPU(ws.optimality_flag_d);
    FREE_HOST(ws.optimality_flag_h);

    FREE_GPU(ws.cub_temp_d);
    FREE_GPU(ws.pricing_out_d);
    FREE_HOST(ws.h_harris_result);

    if (ws.cusolve_handle) {
        cusolverDnDestroy(ws.cusolve_handle);
        ws.cusolve_handle = nullptr;
    }

    if (ws.cublas_handle) {
        cublasDestroy(ws.cublas_handle);
        ws.cublas_handle = nullptr;
    }

    ws.initialized = false;
}

void gpu_load_problem(int m, int n_total, const double* A_flat, const double* b, const double* c) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    cudaMalloc((void**)&ws.A_full_d, n_total * m * sizeof(double));
    cudaMalloc((void**)&ws.c_full_d, n_total * sizeof(double));
    cudaMalloc((void**)&ws.b_storage_d, m * sizeof(double));
    cudaMalloc((void**)&ws.sn_d, n_total * sizeof(double));

    cudaMemcpy(ws.A_full_d, A_flat, n_total * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.c_full_d, c, n_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.b_storage_d, b, m * sizeof(double), cudaMemcpyHostToDevice);
}

void gpu_init_partition(int m, int n_count, const int* B_indices, const int* N_indices) {
    if (!ws.N_d)
        cudaMalloc((void**)&ws.N_d, n_count * sizeof(int));
    cudaMemcpy(ws.B_d, B_indices, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.N_d, N_indices, n_count * sizeof(int), cudaMemcpyHostToDevice);
}

void gpu_download_basis(int m, int* B_out) {
    cudaMemcpy(B_out, ws.B_d, m * sizeof(int), cudaMemcpyDeviceToHost);
}

// ============================================================
// KERNELS
// ============================================================

// Assembles Basis Matrix, spawns tons of threads (everything on gpu, all at once)
__global__ void gather_kernel(int m, const double* A_full, double* A_basis, const int* B_indices) {
    // Calculate unique Thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
                                      const int* __restrict__ pivot_row_ptr,
                                      const int* __restrict__ optimality_flag) {
    // Guard: if already optimal, skip all logic
    if (*optimality_flag == 1)
        return;

    int pivot_row = *pivot_row_ptr;

    // Safety check: If ratio test failed, r will be -1. Return immediately.
    if (pivot_row < 0)
        return;

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

__global__ void harris_fused_kernel(int m,
                                    const double* __restrict__ x,
                                    const double* __restrict__ d,
                                    int* result_idx,
                                    const int* __restrict__ optimality_flag) {
    // Guard: if already optimal, skip all logic
    if (*optimality_flag == 1)
        return;

    struct MinOp {
        __device__ __forceinline__ double operator()(const double& a, const double& b) const {
            return (a < b) ? a : b;
        }
    };

    struct MaxPivotOp {
        __device__ __forceinline__ cub::KeyValuePair<int, double> operator()(
            const cub::KeyValuePair<int, double>& a,
            const cub::KeyValuePair<int, double>& b) const {
            if (a.value > b.value)
                return a;
            if (b.value > a.value)
                return b;
            return (a.key < b.key) ? a : b;
        }
    };

    __shared__ double s_min_theta;

    // 1. Find Min Theta
    typedef cub::BlockReduce<double, 512> BlockReduceDouble;
    __shared__ typename BlockReduceDouble::TempStorage temp_storage_theta;

    double tol = 1e-7;
    double epsilon = 1e-9;
    double min_theta = DBL_MAX;

    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        double d_val = d[i];
        if (d_val > epsilon) {
            double theta = (x[i] + tol) / d_val;
            if (theta < min_theta) {
                min_theta = theta;
            }
        }
    }

    double block_min = BlockReduceDouble(temp_storage_theta).Reduce(min_theta, MinOp());

    if (threadIdx.x == 0)
        s_min_theta = block_min;

    // Barrier to ensure s_min_theta is visible to all threads
    __syncthreads();

    // 2. Find Best Pivot among those with theta <= s_min_theta
    typedef cub::BlockReduce<cub::KeyValuePair<int, double>, 512> BlockReducePivot;
    __shared__ typename BlockReducePivot::TempStorage temp_storage_pivot;

    double best_theta = s_min_theta;

    // Initialize: Key = Index (-1), Value = Magnitude (-1.0)
    cub::KeyValuePair<int, double> thread_best = {-1, -1.0};

    // If best_theta is still DBL_MAX, problem is unbounded, skip logic
    if (best_theta < DBL_MAX) {
        for (int i = threadIdx.x; i < m; i += blockDim.x) {
            double d_val = d[i];
            if (d_val > epsilon) {
                double theta = x[i] / d_val;
                if (theta <= best_theta) {
                    if (d_val > thread_best.value) {
                        thread_best.value = d_val;
                        thread_best.key = i;
                    }
                }
            }
        }
    }

    cub::KeyValuePair<int, double> block_best =
        BlockReducePivot(temp_storage_pivot).Reduce(thread_best, MaxPivotOp());

    // Thread 0 writes result directly to Mapped Host Memory
    if (threadIdx.x == 0) {
        *result_idx = block_best.key;
    }
}

// Checks optimality from the CUB reduction result and unpacks indices if not optimal
__global__ void unpack_pricing_kernel(const cub::KeyValuePair<int, double>* reduction_result,
                                      const int* N,
                                      int* entering_idx_out,
                                      int* entering_col_out,
                                      int* optimality_flag_vram,
                                      int* optimality_flag_host_shadow) {
    if (threadIdx.x == 0) {
        int idx_in_N = reduction_result->key;
        double val = reduction_result->value;

        if (val >= -1e-7 || idx_in_N == -1) {
            *optimality_flag_vram = 1; // Stop GPU loop
            *optimality_flag_host_shadow = 1; // Signal CPU
        } else {
            *optimality_flag_vram = 0;
            *entering_idx_out = idx_in_N;
            *entering_col_out = N[idx_in_N];
        }
    }
}

// Extracts a single column from A_full using the index stored in GPU memory
__global__ void extract_column_kernel(int m,
                                      const double* A_full,
                                      const int* col_idx_ptr,
                                      double* col_out) {
    int col = *col_idx_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        col_out[idx] = A_full[col * m + idx];
    }
}

// Reads everything from GPU pointers
__global__ void update_state_kernel(const int* __restrict__ r_ptr,
                                    const int* __restrict__ j_i_ptr,
                                    const int* __restrict__ entering_col_ptr,
                                    int* __restrict__ B,
                                    int* __restrict__ N,
                                    double* __restrict__ c_B,
                                    const double* __restrict__ c_full,
                                    const int* __restrict__ optimality_flag) {
    // Guard: if already optimal, skip all logic
    if (*optimality_flag == 1)
        return;

    if (threadIdx.x == 0) {
        // Read the winner of the Ratio Test directly from GPU memory
        int r = *r_ptr;
        if (r < 0)
            return; // Unbounded or error

        // Lookup the leaving column index ourselves
        int j_i = *j_i_ptr;
        int entering_col = *entering_col_ptr;
        int leaving_col = B[r];

        // Update Basis Indexs
        B[r] = entering_col;
        // Update Non-Basic Index
        N[j_i] = leaving_col;
        // Update Basis Costs
        c_B[r] = c_full[entering_col];
    }
}

__global__ void gather_costs_kernel(int m, const double* c_full, double* c_B, const int* B_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        c_B[idx] = c_full[B_idx[idx]];
    }
}

// ============================================================
// LOGIC
// ============================================================

// Function: Full Refactorization (slow and stable path)
// 1. Rebuilds the basis matrix B from scratch.
// 2. Factorizes it (LU).
// 3. Computes the explicit inverse B^-1.
void gpu_build_basis_and_invert(int m) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // 1. Gather Basis Columns into B_inv_d using GPU indices
    size_t total_elements = (size_t)m * m;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    gather_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, ws.A_full_d, ws.B_inv_d, ws.B_d);

    // 2. Gather Costs using GPU indices
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    gather_costs_kernel<<<blocks, threads>>>(m, ws.c_full_d, ws.c_B_d, ws.B_d);

    // 3. LU Factorization
    cusolverDnDgetrf(ws.cusolve_handle, m, m, ws.B_inv_d, m, ws.work_d, ws.ipiv_d, ws.info_d);

    // 4. Inversion via Solve (B_inv * X = I)
    set_identity_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, ws.inv_temp_d);
    cusolverDnDgetrs(ws.cusolve_handle, CUBLAS_OP_N, m, m, ws.B_inv_d, m, ws.ipiv_d, ws.inv_temp_d,
                     m, ws.info_d);

    // 5. Copy result back
    cudaMemcpy(ws.B_inv_d, ws.inv_temp_d, m * m * sizeof(double), cudaMemcpyDeviceToDevice);
}

// Uses the Sherman-Morrison kernel to update AB_d in-place.
void gpu_update_basis_fast(int m) {
    size_t total_elements = (size_t)m * m;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    // This relies on 'ws.b_d' containing the direction vector 'd'!
    // This 'd' must have been computed in the previous step (gpu_calc_direction).
    update_inverse_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        m, ws.B_inv_d, ws.b_d, ws.leaving_idx_d, ws.optimality_flag_d);
}

// Extracts column internally and runs GEMV
void gpu_calc_direction(int m) {
    // 1. Extract the entering column (A_entering) into ws.current_col_d
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    extract_column_kernel<<<blocks, threads>>>(m, ws.A_full_d, ws.entering_col_d, ws.current_col_d);

    // 2. Compute d = B^-1 * A_entering
    double alpha = 1.0;
    double beta = 0.0;
    // We use ws.current_col_d
    cublasDgemv(ws.cublas_handle, CUBLAS_OP_N, m, m, &alpha, ws.B_inv_d, m, ws.current_col_d, 1,
                &beta, ws.b_d, 1);
}

void gpu_solve_duals(int m) {
    double alpha = 1.0;
    double beta = 0.0;

    // y = B^-T * c_B
    // Result stored in ws.y_d
    cublasDgemv(ws.cublas_handle, CUBLAS_OP_T, m, m, &alpha, ws.B_inv_d, m, ws.c_B_d, 1, &beta,
                ws.y_d, 1);
}

void gpu_recalc_x_from_persistent_b(int m, double* x_out) {
    double alpha = 1.0;
    double beta = 0.0;
    // Use the persistent b vector stored in VRAM
    cublasDgemv(ws.cublas_handle, CUBLAS_OP_N, m, m, &alpha, ws.B_inv_d, m, ws.b_storage_d, 1,
                &beta, ws.x_B_d, 1);

    cudaMemcpy(x_out, ws.x_B_d, m * sizeof(double), cudaMemcpyDeviceToHost);
}

// This computes sn = c - A^T * y
// (we do this for the whole A, I think it's faster than building the non-basic Matrix)
void gpu_compute_reduced_costs(int m, int n_total) {
    // 1. Calculate A^T * y
    // Initialize sn with c
    cudaMemcpy(ws.sn_d, ws.c_full_d, n_total * sizeof(double), cudaMemcpyDeviceToDevice);

    double alpha = -1.0;
    double beta = 1.0;

    // cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    // We compute: sn = -1.0 * (A^T * y) + 1.0 * sn
    cublasDgemv(ws.cublas_handle,
                CUBLAS_OP_T, // Transpose A because we want dot product of cols with y
                m, n_total, &alpha, ws.A_full_d, m, // LDA is m
                ws.y_d, 1, &beta, ws.sn_d, 1); // incx & incy = 1 are strides
}

// Helper: Update the stored RHS b
void gpu_update_rhs_storage(int m, const double* b_new) {
    cudaMemcpy(ws.b_storage_d, b_new, m * sizeof(double), cudaMemcpyHostToDevice);
}

void gpu_update_state_fused(int m) {
    update_state_kernel<<<1, 1>>>(ws.leaving_idx_d, ws.entering_idx_d, ws.entering_col_d, ws.B_d,
                                  ws.N_d, ws.c_B_d, ws.c_full_d, ws.optimality_flag_d);
}

// Returns true if optimal, false otherwise
void gpu_pricing_dantzig(int n_count) {
    if (n_count <= 0)
        return;

    // 1. Run CUB Reduction
    thrust::device_ptr<int> dev_N(ws.N_d);
    thrust::device_ptr<double> dev_rc(ws.sn_d);
    auto iterator = thrust::make_permutation_iterator(dev_rc, dev_N);

    // CUB ArgMin is guaranteed to overwrite ws.pricing_out_d since n_count > 0.

    cub::DeviceReduce::ArgMin(ws.cub_temp_d, ws.cub_temp_size_d, iterator, ws.pricing_out_d,
                              n_count);

    // 2. Run Unpack Kernel (Checks optimality and sets up state on GPU)
    unpack_pricing_kernel<<<1, 1>>>(ws.pricing_out_d, ws.N_d, ws.entering_idx_d, ws.entering_col_d,
                                    ws.optimality_flag_d, ws.optimality_flag_host_shadow_d);
}

bool gpu_peek_optimality() {
    // PURE MEMORY READ - NO SYNC
    // This reads the Mapped Pin Memory (Zero-Copy)
    // It might be stale by a few microseconds, but that is safe for our logic.
    return (*ws.optimality_flag_h == 1);
}

bool gpu_check_optimality() {
    // 1. Sync the stream (ensure all previous kernels finished writing)
    cudaStreamSynchronize(0);

    // 2. Direct read from Mapped Host Memory (Zero overhead)
    return (*ws.optimality_flag_h == 1);
}

void gpu_run_ratio_test(int m) {
    // 1. Recalculate x_B = B^-1 * b
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemv(ws.cublas_handle, CUBLAS_OP_N, m, m, &alpha, ws.B_inv_d, m, ws.b_storage_d, 1,
                &beta, ws.x_B_d, 1);

    // 2. Launch Fused Kernel
    // Writes to GPU memory ws.leaving_idx_d
    harris_fused_kernel<<<1, 512>>>(m, ws.x_B_d, ws.b_d, ws.leaving_idx_d, ws.optimality_flag_d);
}