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
#include <curand_kernel.h>

#include "workspace.hpp"
#include "linalg_gpu.hpp"
#include "logging.hpp"

__global__ void print_kernel(int m, int n, double *M);

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
    const double * __restrict__ y_d,
    double * __restrict__ y_tmp
) {
    // only executed by a single thread. Very small kernel, I wish it were smarter.
    if(blockIdx.x > 0) return;
    int idx = threadIdx.x;

    for(int i = idx; i < m; i += blockDim.x) {
        y_tmp[i] = y_d[i];
    }

    __syncthreads();

    if(idx > 0) return;
    for(int i = 0; i < count; i++) {
        y_tmp[p_d[i]] -= prod_d[i];
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

// Kernel 
__global__ void obj_kernel(
    int m,
    const double* x_B, const int* B,
    const double* c_full, double* obj
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;

    double local_sum = 0.0;
    for (int k = tid; k < m; k += blockDim.x) {
        int j = B[k];
        double xj = x_B[k];
        local_sum += xj * c_full[j];
    }

    // Store in shared memory for reduction
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce (standard parallel reduction)
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (tid == 0) {
        *obj = sdata[0];
    }
}

__global__ void select_kernel(
    int m, const double* x, const int* B, double *x_B
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = blockDim.x * gridDim.x;
    for(int i = idx; i < m; i += total) {
        x_B[i] = x[B[i]];
    }
}

__global__ void print_kernel(int m, int n, double *M) {
    printf("%p:\n", M);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%lf ", M[j * m + i]);
        }
        printf("\n");
    }
}


__global__ void print_int_kernel(int m, int n, int *M) {
    printf("%p:\n", M);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%d ", M[j * m + i]);
        }
        printf("\n");
    }
}

// ============================================================
// LOGIC
// ============================================================

// Function: Full Refactorization (slow and stable path)
// 1. Rebuilds the basis matrix B from scratch.
// 2. Factorizes it (LU).
// 3. Computes the explicit inverse B^-1.
void gpu_build_basis_and_invert(SharedWorkspace &ws, const PrivateWorkspace &pws) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");
    
    // 0. Attach to the stream of the current workspace.
    const InverseWs &inv = ws.inv;
    int m = ws.m;
    cusolverDnSetStream(inv.cusolve_handle, pws.stream);

    // 1. Launch Gather Kernel: Gather Basis Columns into B_inv_d
    size_t total_elements = (size_t)m * m;
    int threadsPerBlock = 256; // 8 Warps (I think it's the sweet spot)
    // We round up, we want more threads than total elements (that's why we have guard in Kernel)
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    gather_kernel<<<blocksPerGrid, threadsPerBlock, 0, pws.stream>>>(m, ws.scale.A_d, inv.A_B_inv_d, pws.B_d);

    // 2. LU Factorization (in-place on B_inv_d)
    cusolverDnDgetrf(inv.cusolve_handle, m, m, inv.A_B_inv_d, m, inv.work_d, inv.ipiv_d, inv.info_d);

    // 3. Inversion via Solve: B_inv_d * X = I  => X = B_inv_d^-1
    // 3a. Initialize temporary buffer to Identity
    set_identity_kernel<<<blocksPerGrid, threadsPerBlock, 0, pws.stream>>>(m, inv.inv_temp_d);

    // 3b. Solve for Identity. The result X is stored in inv_temp_d.
    cusolverDnDgetrs(inv.cusolve_handle, CUBLAS_OP_N, m, m, inv.A_B_inv_d, m, inv.ipiv_d, inv.inv_temp_d,
                     m, inv.info_d);

    // 3c. Copy result (Inverse) back to B_inv_d to be used as the Basis Inverse
    cudaMemcpyAsync(inv.A_B_inv_d, inv.inv_temp_d, m * m * sizeof(double), cudaMemcpyDeviceToDevice, pws.stream);
}

// Uses the Sherman-Morrison kernel to update (A)B_inv_d in-place.
// This relies on 'ws.b_d' containing the direction vector 'd'!
void gpu_update_basis_fast(PrivateWorkspace &pws, int pivot_row, int new_row) {
    int m = pws.ws->m;

    // Keep B_d in sync.
    cudaMemcpyAsync(&pws.B_d[pivot_row], &new_row, sizeof(int), cudaMemcpyHostToDevice, pws.stream);

    SparseUpdateWs &sp_u = pws.sp_u;

    // Perform sparse update
    add_sparse_update_kernel<<<sp_u.count + 1, 512, 0, pws.stream>>>(m, sp_u.p_d, sp_u.u_d, sp_u.count, pws.d_d, pivot_row);
    sp_u.count += 1;


    // Wait before we overwrite new_row.
    cudaStreamSynchronize(pws.stream);
}

void gpu_apply_sparse_basis(SharedWorkspace &ws, PrivateWorkspace &pws) {
    SparseUpdateWs &sp_u = pws.sp_u;
    int m = ws.m;

    size_t total_elements = (size_t)m * m;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    // This 'd' must have been computed in the previous step (gpu_calc_direction).
    update_inverse_kernel<<<blocksPerGrid, threadsPerBlock, 0, pws.stream>>>(m, ws.inv.A_B_inv_d, ws.inv.inv_temp_d, sp_u.p_d, sp_u.u_d, sp_u.count);
    sp_u.count = 0;
    
    double *tmp_d = ws.inv.inv_temp_d;
    ws.inv.inv_temp_d = ws.inv.A_B_inv_d;
    ws.inv.A_B_inv_d = tmp_d;
}

// Apply the sparse updated B_inv matrix.
void gpu_apply_B_inv(const PrivateWorkspace &pws, cublasOperation_t op, const double *y_d, double *x_d) {
    int m = pws.ws->m;
    double alpha = 1.0;
    double beta = 0.0;

    const InverseWs &inv = pws.ws->inv;
    const SparseUpdateWs &sp_u = pws.sp_u;

    // Fast Path (there are no sparse updates to apply.)
    if(sp_u.count == 0) {
        cublasDgemv_v2(pws.lib.cublas_handle, op, m, m, &alpha, inv.A_B_inv_d, m, y_d, 1, &beta, x_d, 1);
        return;
    }

    if(op == CUBLAS_OP_N) {
        int threadsPerBlock = 512;
        int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

        // y' = B^-1 * y
        cublasDgemv(pws.lib.cublas_handle, CUBLAS_OP_N, m, m, &alpha, inv.A_B_inv_d, m, y_d, 1, &beta, pws.inv_tmp_d, 1);
        
        // x = y' - sum(u_i * y'_p_i)
        apply_sparse_update_N_kernel<<<blocksPerGrid, threadsPerBlock, 0, pws.stream>>>(m, sp_u.p_d, sp_u.u_d, sp_u.count, pws.inv_tmp_d, x_d);
    } else if(op == CUBLAS_OP_T) {
        // prod = U^T * y
        cublasDgemv(pws.lib.cublas_handle, CUBLAS_OP_T, m, sp_u.count, &alpha, sp_u.u_d, m, y_d, 1, &beta, pws.inv_tmp_d, 1);
        
        // y' = y - sum(i: prod_i * e_p_i)
        apply_sparse_selection_T_kernel<<<1,256,0,pws.stream>>>(m, sp_u.p_d, pws.inv_tmp_d, sp_u.count, y_d, sp_u.y_tmp);

        // x = B^-T * y'
        cublasDgemv(pws.lib.cublas_handle, CUBLAS_OP_T, m, m, &alpha, inv.A_B_inv_d, m, sp_u.y_tmp, 1, &beta, x_d, 1);
    } else {
        fprintf(stderr, "Invalid CUBLAS_OP: %d\n", op);
    }
}

// This computes sn = c - A^T * y
// (we do this for the whole A, I think it's faster than building the non-basic Matrix)
void gpu_compute_reduced_costs(const PrivateWorkspace &pws) {
    int m = pws.ws->m, n = pws.ws->n;

    // 1. Calculate A^T * y
    // Initialize sn with c
    cudaMemcpyAsync(pws.sn_d, pws.ws->scale.c_d, n * sizeof(double), cudaMemcpyDeviceToDevice, pws.stream);

    double alpha = -1.0;
    double beta = 1.0;

    // cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    // We compute: sn = -1.0 * (A^T * y) + 1.0 * sn
    cublasDgemv(pws.lib.cublas_handle,
                CUBLAS_OP_T, // Transpose A because we want dot product of cols with y
                m, n, &alpha, pws.ws->scale.A_d, m, // LDA is m
                pws.y_d, 1, &beta, pws.sn_d, 1); // incx & incy = 1 are strides
}

void gpu_init_non_basic(const PrivateWorkspace &pws, const int* N_indices) {
    cudaMemcpyAsync(pws.N_d, N_indices, pws.ws->n * sizeof(int), cudaMemcpyHostToDevice, pws.stream);
}

void gpu_update_non_basic_index(PrivateWorkspace &pws, int offset, int new_val) {
    // Offset is j_i, new_val is ii
    cudaMemcpyAsync(pws.N_d + offset, &new_val, sizeof(int), cudaMemcpyHostToDevice, pws.stream);
}


// Optimized CUB Implementation
PricingResult gpu_pricing_dantzig(PrivateWorkspace &pws) {
    if (pws.ws->n <= 0)
        return {-1, 0.0};
    
    select_kernel<<<16, 256, 0, pws.stream>>>(pws.ws->n - pws.ws->m, pws.sn_d, pws.N_d, pws.sn_N_d);
    cub::DeviceReduce::ArgMin(pws.lib.cub_temp_d, pws.lib.cub_temp_size_d, pws.sn_N_d, pws.lib.pricing_out_d, pws.ws->n - pws.ws->m, pws.stream);


    cub::KeyValuePair<int, double> pricing_out;
    cudaMemcpyAsync(&pricing_out, pws.lib.pricing_out_d, sizeof(cub::KeyValuePair<int, double>),
                    cudaMemcpyDeviceToHost, pws.stream);

    cudaStreamSynchronize(pws.stream);
    int best_idx = pricing_out.key;
    double best_val = pricing_out.value;

    if (best_val >= -1e-7)
        return {-1, best_val};
    return {best_idx, best_val};
}

__global__ void random_pricing_kernel(
    int n,
    const double *sn,
    const int *N,
    double *out_min,
    int *out_min_arg,
    int seed
) {
    extern __shared__ unsigned char smem[];
    double *s_key = (double*)smem;
    int *s_idx = (int*)(s_key + blockDim.x);

    int tid = threadIdx.x;

    // Initialize local best
    double best_key = -1e300;
    int best_idx = -1;

    // Initialize RNG
    curandStatePhilox4_32_10_t rng;
    curand_init( seed, tid, 0, &rng );

    for (int i = tid; i < n; i += blockDim.x) {
        int idx = N[i];
        double val = sn[idx];

        if (val < -1e-7) {
            double w = -val;
            double u = curand_uniform_double(&rng);
            double key = log(u) / w;

            if (key > best_key) {
                best_key = key;
                best_idx = idx;
            }
        }
    }

    s_key[tid] = best_key;
    s_idx[tid] = best_idx;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_key[tid + offset] > s_key[tid]) {
                s_key[tid] = s_key[tid + offset];
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_min = (s_idx[0] >= 0) ? sn[s_idx[0]] : 0.0;
        *out_min_arg = s_idx[0];
    }
}


PricingResult gpu_pricing_random(PrivateWorkspace &pws, int seed) {
    // pws.sn_N_d abused to carry key value info.
    random_pricing_kernel<<<1, 256, (sizeof(int) + sizeof(double)) * 256, pws.stream>>>(pws.ws->n - pws.ws->m, pws.sn_d, pws.N_d, pws.sn_N_d, (int *) &pws.sn_N_d[1], seed);

    PricingResult res;
    cudaMemcpyAsync(&res.index_in_N, (int *) &pws.sn_N_d[1], sizeof(int), cudaMemcpyDeviceToHost, pws.stream);
    cudaMemcpyAsync(&res.min_val, pws.sn_N_d, sizeof(double), cudaMemcpyDeviceToHost, pws.stream);
    cudaStreamSynchronize(pws.stream);
    return res;
}

int gpu_run_ratio_test(const PrivateWorkspace &pws) {
    // 1. Launch Fused Kernel
    // Writes directly to pws.ws->d_harris_result_mapped (which is an alias for pws.ws->h_harris_result)
    harris_fused_kernel<<<1, 512, 0, pws.stream>>>(pws.ws->m, pws.x_B_d, pws.d_d, pws.lib.d_harris_result_mapped);

    // 2. Synchronize Stream
    // This is required to ensure the GPU has finished writing to the mapped memory.
    // It replaces the explicit cudaMemcpy.

    int key;
    cudaMemcpyAsync(&key, pws.lib.d_harris_result_mapped, sizeof(int), cudaMemcpyDeviceToHost, pws.stream);
    cudaStreamSynchronize(pws.stream);
    return key;
}


double gpu_get_obj(const PrivateWorkspace &pws) {
    obj_kernel<<<1, 256, 256 * sizeof(double), pws.stream>>>(pws.ws->m, pws.x_B_d, pws.B_d, pws.ws->scale.c_d, pws.obj_d);
    double obj = 0;
    cudaMemcpyAsync(&obj, pws.obj_d, sizeof(double), cudaMemcpyDeviceToHost, pws.stream);
    cudaStreamSynchronize(pws.stream);
    return obj;
}

void gpu_set_c_B(const PrivateWorkspace &pws) {
    select_kernel<<<16, 256, 0, pws.stream>>>(pws.ws->m, pws.ws->scale.c_d, pws.B_d, pws.c_B_d);
}

__global__ void scatter_kernel(int n, int m, const int *col_ptr, const int *row_idx, const double *values, double *A) {
    for(int col = blockIdx.x; col < n; col += gridDim.x) {
        for(int row = col_ptr[col] + threadIdx.x; row < col_ptr[col + 1]; row += blockDim.x) {
            A[col * m + row_idx[row]] = values[row];
        }
    }
}

void gpu_scatter_A(SharedWorkspace &ws, cudaStream_t stream) {
    cudaMemsetAsync(ws.origin.A_d, 0, sizeof(double) * ws.n * ws.m, stream);
    scatter_kernel<<<128, 256, 0, stream>>>(ws.n, ws.m, ws.origin.A_col_ptr, ws.origin.A_row_idx, ws.origin.A_values, ws.origin.A_d);
}


void gpu_vec_add(const PrivateWorkspace &pws, int n, const double *a, const double *b_d, double *c_d) {
    double alpha = 1.0;
    cudaMemcpyAsync(c_d, a, n*sizeof(double), cudaMemcpyHostToDevice, pws.stream);
    cublasDaxpy_v2(pws.lib.cublas_handle, n, &alpha, b_d, 1, c_d, 1);
}
