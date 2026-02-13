// Global Scope Workspace Organization
#pragma once
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cub/util_type.cuh>

#include "types.hpp"

#define STR(x) #x
#define CHECK() { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { fprintf(stderr, "Kernel launch failed: %s" STR(__LINE__) "\n", cudaGetErrorString(err)); } }

/**
 * Contains original data by the user.
 */
struct OriginalWs {
    int* A_col_ptr;
    int* A_row_idx;
    double* A_values;

    double* A_d;    // A in column major order
    double* b_d;
    double* c_d;
    double* x_d;
};

/**
 * Contains the scaled versions of all vectors and information about the scaling.
 */
struct ScalingWs {
    double* row_scale_d; // Scaling of every entry in a given row
    double* col_scale_d; // Scaling of every entry in a given column

    // Scaled versions of original.
    double* A_d;
    double* b_d;
    double* c_d;
    double* x_d;
};

/**
 * Container for inverse operation.
 */
struct InverseWs {
    double* A_B_inv_d;  // Column major A_B^{-1}
    double* inv_temp_d; // Buffer for operations.

    int* ipiv_d; // Pivot Array for LU decomposition
    int* info_d; // Error Flag for LU decomposition
    double* work_d; // Scratchpad for LU decomposition
    int work_size; // Scratchpad space for LU decomposition

    cusolverDnHandle_t cusolve_handle;
};

struct SharedWorkspace {
    bool initialized = false;
    int m;          // Number of constraints.
    int n;          // Number of variables.

    // Original problem statement
    OriginalWs origin;
    // Scaling data
    ScalingWs scale;
    // Inverse data
    InverseWs inv;

    int *B_d; // Source B from which all workers continue working.
    double *b_perturbe_d;
};

/**
 * Contains data for sparse updates on A_B_inv.
 */
struct SparseUpdateWs {
    double *u_d; // Matrix of update vectors, to enable sparse update of (A_)B_inv.
    int    *p_d; // Matrix of pivots of the different updates.
    int capacity;
    int count;

    double *y_tmp; // Temporary storage for y calculation.
};

/**
 * Library Handlers.
 */
struct LibraryWs {
    // CUB Reduction Buffers (Persistent to avoid malloc in loop!)
    void* cub_temp_d = nullptr;
    size_t cub_temp_size_d = 0;
    cub::KeyValuePair<int, double>* pricing_out_d = nullptr;
    cub::KeyValuePair<int, double>* pricing_out_h = nullptr;

    // Harris Pivoting Result Buffers
    cub::KeyValuePair<int, double>* h_harris_result = nullptr;
    cub::KeyValuePair<int, double>* d_harris_result_mapped = nullptr;

    // Handlers
    cublasHandle_t cublas_handle;
};

enum ExitState {
    None,
    Optimal,
    Unbounded
};

struct SolverWs {
    int stall_counter;
    bool is_perturbed;
    int iter;

    double prev_obj;

    idx B_h;
    idx N_h;

    ExitState state;
};

struct alignas(64) PrivateWorkspace {
    bool initialized = false;
    cudaStream_t stream = nullptr;

    const SharedWorkspace *ws;

    SparseUpdateWs sp_u;
    LibraryWs lib;
    SolverWs sol;

    // Operators.
    double* x_B_d; // Solution Vector for Basis Variables (m)
    double* c_B_d; // Cost Vector for Basis Variables (m)
    double* obj_d; // Objective value for objective calculation.

    double* sn_d = nullptr; // Size: n_total (Reduced costs on Device)
    double* sn_N_d = nullptr; // Size: n_total - m (Reduced costs on Device)
    double* y_d = nullptr; // Size: m (Storage for y)
    double* d_d = nullptr; // Size: m (Storage for d)
    double* b_active = nullptr; // Size: m (Storage for d)
    int* B_d = nullptr; // Size: n (Buffer for Basis Indices)
    int* N_d = nullptr; // Non-Basic Indices on GPU (n)
    double *inv_tmp_d = nullptr;
};

/**
 * Initialize workspace.
 * m: Number of constraints.
 * n: Number of variables.
 */
void gpu_shared_workspace_init(SharedWorkspace &ws, int m, int n, int nnz);
void gpu_shared_workspace_destroy(SharedWorkspace &ws);

void gpu_private_workspace_init(PrivateWorkspace& pws, const SharedWorkspace& ws, idx& B, idx& N, int capacity);
void gpu_private_workspace_destroy(PrivateWorkspace &ws);
