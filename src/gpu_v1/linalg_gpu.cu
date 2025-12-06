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
    double* B_inv2_d; // Basis Inverse Matrix (m * m) write 2 buffer
    double* b_d; // Temp Workspace: Holds direction 'd' or intermediate vectors
    double* inv_temp_d; // Temp buffer for Identity/Inverse calculation (m * m)
    double* x_B_d; // Solution Vector for Basis Variables (m)

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
    double* y_d = nullptr; // Size: m (Storage for y)
    int* B_d = nullptr; // Size: n (Buffer for Basis Indices)
    int* N_d = nullptr; // Non-Basic Indices on GPU (n)

    // CUB Reduction Buffers (Persistent to avoid malloc in loop!)
    void* cub_temp_d = nullptr;
    size_t cub_temp_size_d = 0;
    cub::KeyValuePair<int, double>* pricing_out_d = nullptr;
    cub::KeyValuePair<int, double>* pricing_out_h = nullptr;

    // Harris Pivoting Result Buffers
    cub::KeyValuePair<int, double>* h_harris_result = nullptr;
    cub::KeyValuePair<int, double>* d_harris_result_mapped = nullptr;

    // Handlers
    cusolverDnHandle_t cusolve_handle;
    cublasHandle_t cublas_handle;

    bool initialized = false;
};

static CPUWorkspace ws;

struct PrivateWorkspace {
    double *u_d; // Matrix of update vectors, to enable sparse update of (A_)B_inv.
    int    *p_d; // Matrix of pivots of the different updates.
    int capacity;
    int count;
};

static PrivateWorkspace pws;

void init_gpu_workspace(int n) {
    if (ws.initialized)
        return;

    cudaSetDeviceFlags(cudaDeviceMapHost);

    cusolverDnCreate(&ws.cusolve_handle);
    cublasCreate(&ws.cublas_handle);

    // Allocate Workspace (Scratchpads)
    cudaMalloc((void**)&ws.B_inv_d, n * n * sizeof(double));
    cudaMalloc((void**)&ws.B_inv2_d, n * n * sizeof(double));
    cudaMalloc((void**)&ws.inv_temp_d, n * n * sizeof(double));
    cudaMalloc((void**)&ws.b_d, n * sizeof(double));
    cudaMalloc((void**)&ws.x_B_d, n * sizeof(double));
    cudaMalloc((void**)&ws.ipiv_d, n * sizeof(int));
    cudaMalloc((void**)&ws.info_d, sizeof(int));

    // Allocate Helpers
    cudaMalloc((void**)&ws.B_d, n * sizeof(int));
    cudaMalloc((void**)&ws.y_d, n * sizeof(double));

    cusolverDnDgetrf_bufferSize(ws.cusolve_handle, n, n, ws.B_inv_d, n, &ws.work_size);
    cudaMalloc((void**)&ws.work_d, ws.work_size * sizeof(double));

    // CUB
    cudaMalloc((void**)&ws.pricing_out_d, sizeof(cub::KeyValuePair<int, double>));
    cudaMallocHost((void**)&ws.pricing_out_h,
                   sizeof(cub::KeyValuePair<int, double>)); // Mapped memory
    ws.cub_temp_size_d = 1024 * 1024; // 1MB scratchpad is plenty for reductions
    cudaMalloc((void**)&ws.cub_temp_d, ws.cub_temp_size_d);

    pws.capacity = 20;
    pws.count = 0;
    cudaMalloc((void **) &pws.u_d, n * pws.capacity * sizeof(double));
    cudaMalloc((void **) &pws.p_d, pws.capacity * sizeof(double));

    // Harris Test Buffers
    cudaHostAlloc((void**)&ws.h_harris_result, sizeof(cub::KeyValuePair<int, double>),
                  cudaHostAllocMapped); // Mapped memory
    // Get the device pointer for this host memory
    cudaHostGetDevicePointer((void**)&ws.d_harris_result_mapped, ws.h_harris_result, 0);

    ws.initialized = true;
}

void destroy_gpu_workspace() {
    if (!ws.initialized)
        return;

    // Free and NULL workspace buffers
    if (ws.B_inv_d) {
        cudaFree(ws.B_inv_d);
        ws.B_inv_d = nullptr;
    }
    if (ws.inv_temp_d) {
        cudaFree(ws.B_inv2_d);
    cudaFree(ws.inv_temp_d);
        ws.inv_temp_d = nullptr;
    }
    if (ws.b_d) {
        cudaFree(ws.b_d);
        ws.b_d = nullptr;
    }
    if (ws.ipiv_d) {
        cudaFree(ws.ipiv_d);
        ws.ipiv_d = nullptr;
    }
    if (ws.info_d) {
        cudaFree(ws.info_d);
        ws.info_d = nullptr;
    }
    if (ws.work_d) {
        cudaFree(ws.work_d);
        ws.work_d = nullptr;
    }

    if (ws.cusolve_handle) {
    
    pws.capacity = 0;
    pws.count = 0;
    cudaFree(pws.u_d);
    cudaFree(pws.p_d);
    cusolverDnDestroy(ws.cusolve_handle);
        ws.cusolve_handle = nullptr;
    }

    // Free and NULL persistent storage
    if (ws.A_full_d) {
        cudaFree(ws.A_full_d);
        ws.A_full_d = nullptr;
    }
    if (ws.b_storage_d) {
        cudaFree(ws.b_storage_d);
        ws.b_storage_d = nullptr;
    }
    if (ws.c_full_d) {
        cudaFree(ws.c_full_d);
        ws.c_full_d = nullptr;
    }

    // Free and NULL helpers
    if (ws.sn_d) {
        cudaFree(ws.sn_d);
        ws.sn_d = nullptr;
    }
    if (ws.y_d) {
        cudaFree(ws.y_d);
        ws.y_d = nullptr;
    }
    if (ws.B_d) {
        cudaFree(ws.B_d);
        ws.B_d = nullptr;
    }

    if (ws.N_d) {
        cudaFree(ws.N_d);
        ws.N_d = nullptr;
    }

    if (ws.sn_h) {
        free(ws.sn_h);
        ws.sn_h = nullptr;
    }

    if (ws.cublas_handle) {
        cublasDestroy(ws.cublas_handle);
        ws.cublas_handle = nullptr;
    }

    if (ws.cub_temp_d)
        cudaFree(ws.cub_temp_d);
    if (ws.pricing_out_d)
        cudaFree(ws.pricing_out_d);

    if (ws.x_B_d)
        cudaFree(ws.x_B_d);
    ws.x_B_d = nullptr;

    if (ws.h_harris_result) {
        cudaFreeHost(ws.h_harris_result);
        ws.h_harris_result = nullptr;
    }
    ws.d_harris_result_mapped = nullptr;

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

// Sherman-Morrison Update Kernel with sparse update.
// Inputs:
//   B_inv:  The current Inverse Matrix
//   B_inv2: The next Inverse Matrix
//   p_d: The pivot columns for each update
//   u_d: The actual update vectors
//   count: The number of sparse updates.
__global__ void update_inverse_kernel(int m,
                                      const double* __restrict__ B_inv,
                                      double* __restrict__ B_inv2,
                                      const int* __restrict__ p_d,
                                      const double* __restrict__ u_d,
                                      int count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)m * m)
        return;

    int row = idx % m;
    int col = idx / m;

    // For each of the updates, we find the pivot row value and multiply it with the
    // column of the u vector. E^-1 = (I - u e_p^T - ...). Applying this yields:
    // B_now^-1 = B^-1 - u * B^-1_p - ....
    // This is implemented here.
    double b_val = B_inv[idx];
    for(int i = 0; i < count; i++) {
        int pivot_row = p_d[i];
        double b_pivot_row_val = B_inv[(size_t)col * m + pivot_row];
        double update_val = u_d[i * m + row];
        b_val -= update_val * b_pivot_row_val;
    }
    B_inv2[idx] = b_val;
}

// Add sparse updates with the new pivot and d.
// We assume that all threads that work for a given index are in the same block. There is a block per update.
// p_d the pivot vector.
// u_d the update vectors.
// count the number of updates before this is applied.
// b_d the delta vector.
// pivot_idx the pivot index.
__global__ void add_sparse_update_kernel(
    int m,
    int* __restrict__ p_d,
    double* __restrict__ u_d,
    int count,
    const double* __restrict__ b_d,
    int pivot_idx
) {
    int idx = blockIdx.x;
    if(idx > count) return;

    double dp_n = b_d[pivot_idx];

    // The new update is on the pivot_idx and is 1/d_p * (d - e_p).
    if(idx == count) {
        p_d[idx] = pivot_idx;
        for(int i = threadIdx.x; i < m; i+=blockDim.x) {
            u_d[m * idx + i] = (b_d[i] - ((i == pivot_idx) ? 1 : 0)) / dp_n;
        }

        return;
    }

    double up = u_d[m * idx + pivot_idx];
    __syncthreads();

    for(int i = threadIdx.x; i < m; i+=blockDim.x) {
        double ui = (b_d[i] - ((i == pivot_idx) ? 1 : 0)) / dp_n;
        u_d[m * idx + i] -= ui * up;
    }
}

// Apply sparse updates in place within y.
// p_d the pivot vector.
// u_d the update vectors.
// count the number of update vectors.
// y_d the y vector we are working with.
// y_out the y vector we write to in the end. MUST be different from y_d.
__global__ void apply_sparse_update_N_kernel(
    int m,
    const int* __restrict__ p_d,
    const double* __restrict__ u_d,
    int count,
    const double* __restrict__ y_d,
    double* __restrict__ y_out
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= m) return;

    double out = y_d[idx];
    for(int i = 0; i < count; i++) {
        double y_p = y_d[p_d[i]];
        out -= y_p * u_d[m * i + idx];
    }
    y_out[idx] = out;
}

// Apply sparse update calculation on y. This takes the product vector we get by U^T y.
// p_d the pivot vector.
// prod_d the dot products.
// count the number of update vectors.
// y_d the y vector we are working with.
__global__ void apply_sparse_selection_T_kernel(
    int m,
    const int* __restrict__ p_d,
    const double* __restrict__ prod_d,
    int count,
    double * __restrict__ y_d
) {
    // only executed by a single thread. Very small kernel, I wish it were smarter.
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > 0) return;

    for(int i = 0; i < count; i++) {
        y_d[p_d[i]] -= prod_d[i];
    }
}

__global__ void harris_fused_kernel(int m,
                                    const double* __restrict__ x,
                                    const double* __restrict__ d,
                                    cub::KeyValuePair<int, double>* d_result_mapped) {
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
        *d_result_mapped = block_best;
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

    // Remove any remaining sparse updates.
    pws.count = 0;
    
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

// Uses the Sherman-Morrison kernel to update (A)B_inv_d in-place.
// This relies on 'ws.b_d' containing the direction vector 'd'!
void gpu_update_basis_fast(int m, int pivot_row) {
    add_sparse_update_kernel<<<pws.count + 1, 512>>>(m, pws.p_d, pws.u_d, pws.count, ws.b_d, pivot_row);
    cudaDeviceSynchronize();
    pws.count += 1;

    if (pws.count == pws.capacity) {
        size_t total_elements = (size_t)m * m;
        int threadsPerBlock = 256;
        int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
        // This 'd' must have been computed in the previous step (gpu_calc_direction).
        update_inverse_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, ws.B_inv_d, ws.B_inv2_d, pws.p_d, pws.u_d, pws.count);
        cudaDeviceSynchronize();
        pws.count = 0;

        double *tmp_d = ws.B_inv_d;
        ws.B_inv_d = ws.B_inv2_d;
        ws.B_inv2_d = tmp_d;
    }
}

// Apply the sparse updated B_inv matrix.
static void gpu_apply_B_inv(int m, cublasOperation_t op, double *y_d, double *x_d) {
    double alpha = 1.0;
    double beta = 0.0;

    if(op == CUBLAS_OP_N) {
        int threadsPerBlock = 512;
        int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

        // y' = B^-1 * y
        cublasDgemv(ws.cublas_handle, CUBLAS_OP_N, m, m, &alpha, ws.B_inv_d, m, y_d, 1, &beta, ws.inv_temp_d, 1);
        
        // x = y' - sum(u_i * y'_p_i)
        apply_sparse_update_N_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, pws.p_d, pws.u_d, pws.count, ws.inv_temp_d, x_d);
        cudaDeviceSynchronize();
    } else if(op == CUBLAS_OP_T) {
        // prod = U^T * y
        cublasDgemv(ws.cublas_handle, CUBLAS_OP_T, m, pws.count, &alpha, pws.u_d, m, y_d, 1, &beta, ws.inv_temp_d, 1);
        
        // y' = y - sum(i: prod_i * e_p_i)
        apply_sparse_selection_T_kernel<<<1,1>>>(m, pws.p_d, ws.inv_temp_d, pws.count, y_d);
        cudaDeviceSynchronize();

        // x = B^-T * y'
        cublasDgemv(ws.cublas_handle, CUBLAS_OP_T, m, m, &alpha, ws.B_inv_d, m, y_d, 1, &beta, x_d, 1);
    } else {
        fprintf(stderr, "Invalid CUBLAS_OP: %d\n", op);
    }
}

// Computes d = B^-1 * A_j using Matrix-Vector Multiplication.
void gpu_calc_direction(int m, int col_idx) {
    // d = B^-1 * A_col
    double* A_col_ptr = ws.A_full_d + (static_cast<size_t>(col_idx) * m);

    // We compute: ws.b_d = 1.0 * AB_d * A_col_ptr
    // This computes d = B^-1 * A_j
    gpu_apply_B_inv(m, CUBLAS_OP_N, A_col_ptr, ws.b_d);
}

void gpu_solve_duals(int m, const double* c_B_host) {
    // Copy c_B to Device (ws.b_d is safe to reuse here)
    cudaMemcpy(ws.b_d, c_B_host, m * sizeof(double), cudaMemcpyHostToDevice);

    // y = B^-T * c_B
    // Result stored in ws.y_d
    gpu_apply_B_inv(m, CUBLAS_OP_T, ws.b_d, ws.y_d);
}

void gpu_recalc_x_from_persistent_b(int m, double* x_out) {
    // Use the persistent b vector stored in VRAM
    // x_out = B^-1 * b
    gpu_apply_B_inv(m, CUBLAS_OP_N, ws.b_storage_d, ws.x_B_d);
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

// Helper: Update the stored RHS b (used when applying perturbation)
void gpu_update_rhs_storage(int m, const double* b_new) {
    cudaMemcpy(ws.b_storage_d, b_new, m * sizeof(double), cudaMemcpyHostToDevice);
}

void gpu_init_non_basic(int n_count, const int* N_indices) {
    if (!ws.N_d)
        cudaMalloc((void**)&ws.N_d, n_count * sizeof(int));
    cudaMemcpy(ws.N_d, N_indices, n_count * sizeof(int), cudaMemcpyHostToDevice);
}

void gpu_update_non_basic_index(int offset, int new_val) {
    // Offset is j_i, new_val is ii
    cudaMemcpy(ws.N_d + offset, &new_val, sizeof(int), cudaMemcpyHostToDevice);
}

// Optimized CUB Implementation
PricingResult gpu_pricing_dantzig(int n_count) {
    if (n_count <= 0)
        return {-1, 0.0};

    ws.pricing_out_h->key = -1;
    ws.pricing_out_h->value = 1.0e30;

    thrust::device_ptr<int> dev_N(ws.N_d);
    thrust::device_ptr<double> dev_rc(ws.sn_d);
    auto iterator = thrust::make_permutation_iterator(dev_rc, dev_N);

    cub::DeviceReduce::ArgMin(ws.cub_temp_d, ws.cub_temp_size_d, iterator, ws.pricing_out_d,
                              n_count);

    cudaMemcpyAsync(ws.pricing_out_h, ws.pricing_out_d, sizeof(cub::KeyValuePair<int, double>),
                    cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);

    int best_idx = ws.pricing_out_h->key;
    double best_val = ws.pricing_out_h->value;

    if (best_val >= -1e-7)
        return {-1, best_val};
    return {best_idx, best_val};
}

int gpu_run_ratio_test(int m) {
    // 1. Launch Fused Kernel
    // Writes directly to ws.d_harris_result_mapped (which is an alias for ws.h_harris_result)
    harris_fused_kernel<<<1, 512>>>(m, ws.x_B_d, ws.b_d, ws.d_harris_result_mapped);

    // 2. Synchronize Stream
    // This is required to ensure the GPU has finished writing to the mapped memory.
    // It replaces the explicit cudaMemcpy.
    cudaStreamSynchronize(0);

    // 3. Read directly from host pointer
    return ws.h_harris_result->key;
}
