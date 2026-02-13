#include <assert.h>
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <stdexcept>
#include <thrust/device_ptr.h>
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
    double* A_tr_d = nullptr; // Size: n_total * m
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
    double* theta_d = nullptr;

    // Mapped Optimality Flag
    int* optimality_flag_h = nullptr; // Host pointer (mapped)
    int* optimality_flag_d = nullptr; // VRAM pointer (fast)

    // CUB Reduction Buffers
    void* cub_temp_d = nullptr;
    size_t cub_temp_size_d = 0;
    cub::KeyValuePair<int, double>* pricing_out_d = nullptr;

    // Handlers
    cusolverDnHandle_t cusolve_handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;

    bool initialized = false;
};

static CPUWorkspace ws;

void init_gpu_workspace(int n) {
    if (ws.initialized)
        return;

    // Create a specific stream for the Graph
    cudaStreamCreate(&ws.stream);

    cusolverDnCreate(&ws.cusolve_handle);
    cusolverDnSetStream(ws.cusolve_handle, ws.stream);
    cublasCreate(&ws.cublas_handle);
    cublasSetStream(ws.cublas_handle, ws.stream);

    // Standard Allocations
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
    cudaMalloc((void**)&ws.theta_d, sizeof(double));

    // Optimality Flag
    cudaMalloc((void**)&ws.optimality_flag_d, sizeof(int));
    cudaHostAlloc((void**)&ws.optimality_flag_h, sizeof(int), cudaHostAllocDefault);

    cusolverDnDgetrf_bufferSize(ws.cusolve_handle, n, n, ws.B_inv_d, n, &ws.work_size);
    cudaMalloc((void**)&ws.work_d, ws.work_size * sizeof(double));

    // CUB
    cudaMalloc((void**)&ws.pricing_out_d, sizeof(cub::KeyValuePair<int, double>));
    ws.cub_temp_size_d = 4 * 1024 * 1024;
    cudaMalloc((void**)&ws.cub_temp_d, ws.cub_temp_size_d);

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
    FREE_GPU(ws.A_tr_d);
    FREE_GPU(ws.b_storage_d);
    FREE_GPU(ws.c_full_d);

    FREE_GPU(ws.sn_d);
    FREE_GPU(ws.y_d);
    FREE_GPU(ws.B_d);
    FREE_GPU(ws.N_d);

    FREE_GPU(ws.entering_idx_d);
    FREE_GPU(ws.entering_col_d);
    FREE_GPU(ws.leaving_idx_d);
    FREE_GPU(ws.theta_d);

    FREE_GPU(ws.optimality_flag_d);
    FREE_HOST(ws.optimality_flag_h);

    FREE_GPU(ws.cub_temp_d);
    FREE_GPU(ws.pricing_out_d);

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
    cudaMalloc((void**)&ws.A_tr_d, n_total * m * sizeof(double));
    cudaMalloc((void**)&ws.c_full_d, n_total * sizeof(double));
    cudaMalloc((void**)&ws.b_storage_d, m * sizeof(double));
    cudaMalloc((void**)&ws.sn_d, n_total * sizeof(double));

    cudaMemcpy(ws.A_full_d, A_flat, n_total * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.c_full_d, c, n_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.b_storage_d, b, m * sizeof(double), cudaMemcpyHostToDevice);

    // Compute Transpose A once for fast pricing
    double alpha = 1.0, beta = 0.0;
    cublasDgeam(ws.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n_total, m, &alpha, ws.A_full_d, m,
                &beta, ws.A_tr_d, n_total, ws.A_tr_d, n_total);

    cudaStreamSynchronize(ws.stream);
}

void gpu_init_partition(int m, int n_count, const int* B_indices, const int* N_indices) {
    if (!ws.N_d)
        cudaMalloc((void**)&ws.N_d, n_count * sizeof(int));
    cudaMemcpy(ws.B_d, B_indices, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.N_d, N_indices, n_count * sizeof(int), cudaMemcpyHostToDevice);
}

void gpu_download_basis(int m, int* B_out) {
    cudaMemcpyAsync(B_out, ws.B_d, m * sizeof(int), cudaMemcpyDeviceToHost, ws.stream);
    cudaStreamSynchronize(ws.stream);
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

// Harris Ratio Test
struct HarrisCandidate {
    double theta;
    double pivot_val;
    int index;
};

__global__ void harris_stage1_kernel(int m,
                                     const double* __restrict__ x,
                                     const double* __restrict__ d,
                                     HarrisCandidate* block_results,
                                     const int* __restrict__ optimality_flag) {
    // Check Sticky Flag
    if (*optimality_flag != 0)
        return;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double min_theta = DBL_MAX;
    double best_pivot = -1.0;
    int best_idx = -1;
    double tol = 1e-7;
    double epsilon = 1e-9;

    for (int i = gid; i < m; i += gridDim.x * blockDim.x) {
        double d_val = d[i];
        if (d_val > epsilon) {
            double theta = (x[i] + tol) / d_val;

            if (theta < min_theta - 1e-12) {
                min_theta = theta;
                best_pivot = d_val;
                best_idx = i;
            } else if (abs(theta - min_theta) <= 1e-12) {
                if (d_val > best_pivot) {
                    best_pivot = d_val;
                    best_idx = i;
                }
            }
        }
    }

    typedef cub::BlockReduce<HarrisCandidate, 256> BlockReduceHarris;
    __shared__ typename BlockReduceHarris::TempStorage temp_storage;

    HarrisCandidate local = {min_theta, best_pivot, best_idx};

    HarrisCandidate block_best =
        BlockReduceHarris(temp_storage)
            .Reduce(local, [](const HarrisCandidate& a, const HarrisCandidate& b) {
                if (a.theta < b.theta - 1e-12)
                    return a;
                if (b.theta < a.theta - 1e-12)
                    return b;
                return (a.pivot_val > b.pivot_val) ? a : b;
            });

    if (tid == 0) {
        block_results[blockIdx.x] = block_best;
    }
}

__global__ void harris_stage2_kernel(int num_blocks,
                                     const HarrisCandidate* __restrict__ block_results,
                                     int* result_idx,
                                     double* theta_out,
                                     int* optimality_flag) {
    // Check Sticky Flag
    if (*optimality_flag != 0)
        return;

    if (threadIdx.x == 0) {
        double min_theta = DBL_MAX;
        double best_pivot = -1.0;
        int best_idx = -1;

        for (int i = 0; i < num_blocks; ++i) {
            HarrisCandidate c = block_results[i];
            if (c.index == -1)
                continue;

            if (c.theta < min_theta - 1e-12) {
                min_theta = c.theta;
                best_pivot = c.pivot_val;
                best_idx = c.index;
            } else if (abs(c.theta - min_theta) <= 1e-12 && c.pivot_val > best_pivot) {
                best_pivot = c.pivot_val;
                best_idx = c.index;
            }
        }
        *result_idx = best_idx;
        *theta_out = min_theta;

        if (best_idx == -1) {
            *optimality_flag = 2; // Unbounded
        }
    }
}

// Checks optimality from the CUB reduction result and unpacks indices if not optimal
__global__ void unpack_pricing_kernel(const cub::KeyValuePair<int, double>* reduction_result,
                                      const int* N,
                                      int* entering_idx_out,
                                      int* entering_col_out,
                                      int* optimality_flag) {
    if (threadIdx.x == 0) {
        if (*optimality_flag != 0) {
            return;
        }
        int idx_in_N = reduction_result->key;
        double val = reduction_result->value;

        if (val >= -1e-7 || idx_in_N == -1) {
            *optimality_flag = 1; // Stop GPU loop
            *entering_col_out = 0;
        } else {
            *entering_idx_out = idx_in_N;
            *entering_col_out = N[idx_in_N];
        }
    }
}

// Extracts a single column from A_full using the index stored in GPU memory
__global__ void extract_column_kernel(int m,
                                      const double* A_full,
                                      const int* col_idx_ptr,
                                      double* col_out,
                                      const int* optimality_flag) {
    if (*optimality_flag != 0) {
        return;
    }

    int col = *col_idx_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        col_out[idx] = A_full[col * m + idx];
    }
}

__global__ void state_update_kernel(int m,
                                    double* __restrict__ x,
                                    const double* __restrict__ d,
                                    const double* __restrict__ theta_ptr,
                                    const int* __restrict__ r_ptr,
                                    const int* __restrict__ entering_idx_ptr,
                                    const int* __restrict__ entering_col_ptr,
                                    int* __restrict__ B,
                                    int* __restrict__ N,
                                    double* __restrict__ c_B,
                                    const double* __restrict__ c_full,
                                    const int* __restrict__ optimality_flag) {
    if (*optimality_flag != 0)
        return;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double theta = *theta_ptr;
    int r = *r_ptr;

    // 1. Vector Update (Parallel across threads)
    for (int i = idx; i < m; i += stride) {
        if (i == r) {
            x[i] = theta; // Stability fix
        } else {
            x[i] -= theta * d[i];
        }
    }

    // 2. Scalar Update (Thread 0 only)
    if (idx == 0) {
        int j_i = *entering_idx_ptr;
        int entering_col = *entering_col_ptr;

        // Swap indices
        int leaving_col = B[r];
        B[r] = entering_col;
        N[j_i] = leaving_col;

        // Update costs
        c_B[r] = c_full[entering_col];
    }
}

__global__ void prepare_pivot_row_kernel(int m,
                                         const double* __restrict__ B_inv,
                                         const double* __restrict__ d,
                                         double* __restrict__ pivot_row_storage,
                                         const int* __restrict__ r_ptr,
                                         const int* __restrict__ optimality_flag) {
    if (*optimality_flag != 0)
        return;

    int r = *r_ptr;
    if (r < 0)
        return;

    // Grid Stride Loop over columns j
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double pivot_val = d[r];
    double inv_pivot = 1.0 / pivot_val;

    for (int j = idx; j < m; j += stride) {
        // Read B_inv[r, j]. Matrix is Col-Major, so index is j*m + r
        double val = B_inv[j * m + r];

        // Normalize and store in temp buffer
        pivot_row_storage[j] = val * inv_pivot;
    }
}

__global__ void matrix_update_kernel(int m,
                                     double* __restrict__ B_inv,
                                     const double* __restrict__ d,
                                     const double* __restrict__ pivot_row_normalized, // The buffer
                                     const int* __restrict__ r_ptr,
                                     const int* __restrict__ optimality_flag) {
    if (*optimality_flag != 0)
        return;
    int r = *r_ptr;
    if (r < 0)
        return;

    // Grid Stride Loop for Matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = m * m;

    for (int k = idx; k < size; k += stride) {
        int row = k % m;
        int col = k / m;

        // Don't read B_inv if we are overwriting it (row == r)
        if (row == r) {
            // For the pivot row, the new value is just the normalized value
            B_inv[k] = pivot_row_normalized[col];
        } else {
            // Standard update: B_ij -= d_i * NormalizedPivotRow_j
            double d_val = d[row];
            double norm_pivot_val = pivot_row_normalized[col];
            B_inv[k] -= d_val * norm_pivot_val;
        }
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

    gather_kernel<<<blocksPerGrid, threadsPerBlock, 0, ws.stream>>>(m, ws.A_full_d, ws.B_inv_d,
                                                                    ws.B_d);

    // 2. Gather Costs using GPU indices
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    gather_costs_kernel<<<blocks, threads, 0, ws.stream>>>(m, ws.c_full_d, ws.c_B_d, ws.B_d);

    // 3. LU Factorization
    cusolverDnDgetrf(ws.cusolve_handle, m, m, ws.B_inv_d, m, ws.work_d, ws.ipiv_d, ws.info_d);

    // 4. Inversion via Solve (B_inv * X = I)
    set_identity_kernel<<<blocksPerGrid, threadsPerBlock, 0, ws.stream>>>(m, ws.inv_temp_d);
    cusolverDnDgetrs(ws.cusolve_handle, CUBLAS_OP_N, m, m, ws.B_inv_d, m, ws.ipiv_d, ws.inv_temp_d,
                     m, ws.info_d);

    // 5. Copy result back
    cudaMemcpyAsync(ws.B_inv_d, ws.inv_temp_d, m * m * sizeof(double), cudaMemcpyDeviceToDevice,
                    ws.stream);

    // 6. Compute x_B = B^-1 * b
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemv(ws.cublas_handle, CUBLAS_OP_N, m, m, &alpha, ws.B_inv_d, m, ws.b_storage_d, 1,
                &beta, ws.x_B_d, 1);
}

// Extracts column internally and runs GEMV
void gpu_calc_direction(int m) {
    // 1. Extract the entering column (A_entering) into ws.current_col_d
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    extract_column_kernel<<<blocks, threads, 0, ws.stream>>>(
        m, ws.A_full_d, ws.entering_col_d, ws.current_col_d, ws.optimality_flag_d);

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
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(ws.cublas_handle, CUBLAS_OP_N, m, m, &alpha, ws.B_inv_d, m, ws.b_storage_d, 1,
                &beta, ws.x_B_d, 1);
    if (x_out) {
        cudaMemcpyAsync(x_out, ws.x_B_d, m * sizeof(double), cudaMemcpyDeviceToHost, ws.stream);
        cudaStreamSynchronize(ws.stream);
    }
}

// This computes sn = c - A^T * y
// (we do this for the whole A, I think it's faster than building the non-basic Matrix)
void gpu_compute_reduced_costs(int m, int n_total) {
    // 1. Calculate A^T * y
    // Initialize sn with c
    cudaMemcpyAsync(ws.sn_d, ws.c_full_d, n_total * sizeof(double), cudaMemcpyDeviceToDevice,
                    ws.stream);

    double alpha = -1.0;
    double beta = 1.0;

    // cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    // We compute: sn = -1.0 * (A^T * y) + 1.0 * sn
    cublasDgemv(ws.cublas_handle,
                CUBLAS_OP_N, // Transpose A because we want dot product of cols with y
                n_total, m, &alpha, ws.A_tr_d, n_total, // LDA is n_total
                ws.y_d, 1, &beta, ws.sn_d, 1); // incx & incy = 1 are strides
}

// Helper: Update the stored RHS b
void gpu_update_rhs_storage(int m, const double* b_new) {
    cudaMemcpyAsync(ws.b_storage_d, b_new, m * sizeof(double), cudaMemcpyHostToDevice, ws.stream);
}

void gpu_update_all_state(int m) {
    // 1. Snapshot Pivot Row (Safety Step)
    // We reuse ws.current_col_d as a temporary buffer for the pivot row.
    // We fix Cache Thrashing by saving in seperate buffer before matrix update.
    int threads = 256;
    int blocks_snap = (m + threads - 1) / threads;
    if (blocks_snap > 64)
        blocks_snap = 64;

    prepare_pivot_row_kernel<<<blocks_snap, threads, 0, ws.stream>>>(
        m, ws.B_inv_d, ws.b_d,
        ws.current_col_d, // temp buffer
        ws.leaving_idx_d, ws.optimality_flag_d);

    // 2. Matrix Update
    // Update B_inv using the buffer.
    size_t total_elements = (size_t)m * m;
    int blocks_mat = (total_elements + threads - 1) / threads;

    // Safety cap for huge matrices to avoid launch failures
    if (blocks_mat > 65535)
        blocks_mat = 65535;

    matrix_update_kernel<<<blocks_mat, threads, 0, ws.stream>>>(m, ws.B_inv_d, ws.b_d,
                                                                ws.current_col_d, // temp buffer
                                                                ws.leaving_idx_d,
                                                                ws.optimality_flag_d);

    // 3. Vector & State Update
    int blocks_vec = (m + threads - 1) / threads;
    if (blocks_vec > 64)
        blocks_vec = 64;

    state_update_kernel<<<blocks_vec, threads, 0, ws.stream>>>(
        m, ws.x_B_d, ws.b_d, ws.theta_d, ws.leaving_idx_d, ws.entering_idx_d, ws.entering_col_d,
        ws.B_d, ws.N_d, ws.c_B_d, ws.c_full_d, ws.optimality_flag_d);
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
                              n_count, ws.stream);

    unpack_pricing_kernel<<<1, 1, 0, ws.stream>>>(ws.pricing_out_d, ws.N_d, ws.entering_idx_d,
                                                  ws.entering_col_d, ws.optimality_flag_d);
}

void gpu_run_ratio_test(int m) {
    int threads = 256;
    int num_blocks = (m + threads - 1) / threads;
    if (num_blocks > 64)
        num_blocks = 64;
    if (num_blocks < 1)
        num_blocks = 1;

    HarrisCandidate* block_results_d = (HarrisCandidate*)ws.cub_temp_d;

    harris_stage1_kernel<<<num_blocks, threads, 0, ws.stream>>>(
        m, ws.x_B_d, ws.b_d, block_results_d, ws.optimality_flag_d);
    harris_stage2_kernel<<<1, 1, 0, ws.stream>>>(num_blocks, block_results_d, ws.leaving_idx_d,
                                                 ws.theta_d, ws.optimality_flag_d);
}

int gpu_check_status() {
    cudaMemcpyAsync(ws.optimality_flag_h, ws.optimality_flag_d, sizeof(int), cudaMemcpyDeviceToHost,
                    ws.stream);
    cudaStreamSynchronize(ws.stream);
    return *ws.optimality_flag_h;
}

cudaStream_t gpu_get_stream() { return ws.stream; }