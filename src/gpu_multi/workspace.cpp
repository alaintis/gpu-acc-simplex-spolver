#include <iostream>
#include "workspace.hpp"

// Init all subcomponents of the Shared Workspace.

static void inv_init(InverseWs &inv, int m) {
    cusolverDnCreate(&inv.cusolve_handle);

    cudaMalloc((void**)&inv.A_B_inv_d, m * m * sizeof(double));
    cudaMalloc((void**)&inv.inv_temp_d, m * m * sizeof(double));
    cudaMalloc((void**)&inv.ipiv_d, m * sizeof(int));
    cudaMalloc((void**)&inv.info_d, sizeof(int));
    
    cusolverDnDgetrf_bufferSize(inv.cusolve_handle, m, m, inv.A_B_inv_d, m, &inv.work_size);
    cudaMalloc((void**)&inv.work_d, inv.work_size * sizeof(double));
}

static void original_init(OriginalWs &origin, int m, int n, int nnz) {
    cudaMalloc((void **)&origin.A_col_ptr, (n + 1) * sizeof(int));
    cudaMalloc((void **)&origin.A_row_idx, nnz * sizeof(int));
    cudaMalloc((void **)&origin.A_values, nnz * sizeof(double));
    cudaMalloc((void **)&origin.A_d, m * n * sizeof(double));
    cudaMalloc((void **)&origin.b_d, m * sizeof(double));
    cudaMalloc((void **)&origin.c_d, n * sizeof(double));
    cudaMalloc((void **)&origin.x_d, n * sizeof(double));
}

static void scale_init(ScalingWs &scale, int m, int n) {
    cudaMalloc((void **)&scale.row_scale_d, m * sizeof(double));
    cudaMalloc((void **)&scale.col_scale_d, n * sizeof(double));

    cudaMalloc((void **)&scale.A_d, m * n * sizeof(double));
    cudaMalloc((void **)&scale.b_d, m * sizeof(double));
    cudaMalloc((void **)&scale.c_d, n * sizeof(double));
    cudaMalloc((void **)&scale.x_d, n * sizeof(double));
}

void gpu_shared_workspace_init(SharedWorkspace &ws, int m, int n, int nnz) {
    if(ws.initialized) std::cerr << "FAILURE! Reinitialize Workspace" << std::endl;
    ws.initialized = true;

    ws.m = m;
    ws.n = n;
    inv_init(ws.inv, m);
    original_init(ws.origin, m, n, nnz);
    scale_init(ws.scale, m, n);
    cudaMalloc((void **)&ws.b_perturbe_d, m * sizeof(double));
    cudaMalloc((void **)&ws.B_d, m * sizeof(int));
}


// Destroy all subcomponents of the private workspace.

static void inv_destroy(InverseWs &inv) {
    if (inv.A_B_inv_d) cudaFree(inv.A_B_inv_d);
    if (inv.inv_temp_d) cudaFree(inv.inv_temp_d);
    if (inv.ipiv_d) cudaFree(inv.ipiv_d);
    if (inv.info_d) cudaFree(inv.info_d);
    if (inv.work_d) cudaFree(inv.work_d);

    if (inv.cusolve_handle)
        cusolverDnDestroy(inv.cusolve_handle);

    inv = {};  // reset struct to zero / null
}

static void original_destroy(OriginalWs &origin) {
    if (origin.A_col_ptr) cudaFree(origin.A_col_ptr);
    if (origin.A_row_idx) cudaFree(origin.A_row_idx);
    if (origin.A_values) cudaFree(origin.A_values);
    if (origin.A_d) cudaFree(origin.A_d);
    if (origin.b_d) cudaFree(origin.b_d);
    if (origin.c_d) cudaFree(origin.c_d);
    if (origin.x_d) cudaFree(origin.x_d);

    origin = {};
}

static void scale_destroy(ScalingWs &scale) {
    if (scale.row_scale_d) cudaFree(scale.row_scale_d);
    if (scale.col_scale_d) cudaFree(scale.col_scale_d);

    if (scale.A_d) cudaFree(scale.A_d);
    if (scale.b_d) cudaFree(scale.b_d);
    if (scale.c_d) cudaFree(scale.c_d);
    if (scale.x_d) cudaFree(scale.x_d);

    scale = {};
}

void gpu_shared_workspace_destroy(SharedWorkspace &ws) {
    if (!ws.initialized) {
        std::cerr << "WARNING: destroying uninitialized workspace" << std::endl;
        return;
    }
    cudaSetDeviceFlags(cudaDeviceMapHost);

    inv_destroy(ws.inv);
    original_destroy(ws.origin);
    scale_destroy(ws.scale);
    cudaFree(ws.b_perturbe_d);
    cudaFree(ws.B_d);

    ws.initialized = false;
    ws = {};
}


void gpu_private_workspace_init(PrivateWorkspace& pws, const SharedWorkspace& ws, idx& B, idx& N, int capacity) {
    if (pws.initialized) {
        std::cerr << "FAILURE! Reinitializing PrivateWorkspace" << std::endl;
        return;
    }
    pws.initialized = true;
    cudaStreamCreate(&pws.stream);
    const SharedWorkspace &ws2 = ws; 
    pws.ws = &ws;

    // Sparse update workspace
    pws.sp_u.capacity = capacity;
    pws.sp_u.count = 0;
    cudaMalloc((void**)&pws.sp_u.u_d, ws.m * (capacity + 1) * sizeof(double)); // m rows, count updates
    cudaMalloc((void**)&pws.sp_u.p_d, (capacity + 1) * sizeof(int));           // pivot indices for updates
    cudaMalloc((void**)&pws.sp_u.y_tmp, ws.m * sizeof(double)); // m rows

    // Operators
    cudaMalloc((void**)&pws.x_B_d, ws.m * sizeof(double));
    cudaMalloc((void**)&pws.c_B_d, ws.m * sizeof(double));
    cudaMalloc((void**)&pws.obj_d, sizeof(double));

    cudaMalloc((void**)&pws.sn_d, ws.n * sizeof(double));
    cudaMalloc((void**)&pws.sn_N_d, ws.n * sizeof(double));
    cudaMalloc((void**)&pws.y_d, ws.m * sizeof(double));
    cudaMalloc((void**)&pws.inv_tmp_d, ws.m * sizeof(double));
    cudaMalloc((void**)&pws.d_d, ws.m * sizeof(double));
    cudaMalloc((void**)&pws.B_d, ws.m * sizeof(int));
    cudaMalloc((void**)&pws.N_d, ws.n * sizeof(int));

    pws.b_active = ws.scale.b_d;

    // Solver state
    pws.sol.is_perturbed = false;
    pws.sol.iter = 0;
    pws.sol.prev_obj = 0;
    pws.sol.state = ExitState::None;
    pws.sol.B_h = B;
    pws.sol.N_h = N;

    // Library handles
    cublasCreate(&pws.lib.cublas_handle);
    cublasSetStream_v2(pws.lib.cublas_handle, pws.stream);
    
    // CUB
    pws.lib.cub_temp_size_d = 1024 * 1024; // 1MB scratchpad is plenty for reductions
    cudaMalloc((void**)&pws.lib.cub_temp_d, pws.lib.cub_temp_size_d);
    cudaMalloc((void**)&pws.lib.pricing_out_d, sizeof(cub::KeyValuePair<int, double>));

    cudaMallocHost((void**)&pws.lib.pricing_out_h, sizeof(cub::KeyValuePair<int, double>)); // Mapped memory
    cudaHostAlloc((void**)&pws.lib.h_harris_result, sizeof(cub::KeyValuePair<int, double>),
                  cudaHostAllocMapped); // Mapped memory
    // Get the device pointer for this host memory
    cudaHostGetDevicePointer((void**)&pws.lib.d_harris_result_mapped, pws.lib.h_harris_result, 0);
}

void gpu_private_workspace_destroy(PrivateWorkspace& pws) {
    if (!pws.initialized) return;
    cudaStreamDestroy(pws.stream);

    // Sparse update
    if (pws.sp_u.u_d) cudaFree(pws.sp_u.u_d);
    if (pws.sp_u.p_d) cudaFree(pws.sp_u.p_d);
    if (pws.sp_u.y_tmp) cudaFree(pws.sp_u.y_tmp);
    pws.sp_u = {};

    // Operators
    if (pws.x_B_d) cudaFree(pws.x_B_d);
    if (pws.c_B_d) cudaFree(pws.c_B_d);
    if (pws.obj_d) cudaFree(pws.obj_d);
    if (pws.sn_d) cudaFree(pws.sn_d);
    if (pws.sn_N_d) cudaFree(pws.sn_N_d);
    if (pws.y_d) cudaFree(pws.y_d);
    if (pws.inv_tmp_d) cudaFree(pws.inv_tmp_d);
    if (pws.d_d) cudaFree(pws.d_d);
    if (pws.B_d) cudaFree(pws.B_d);
    if (pws.N_d) cudaFree(pws.N_d);

    pws.x_B_d = pws.c_B_d = pws.obj_d = nullptr;
    pws.sn_d = pws.y_d = nullptr;
    pws.sn_N_d = pws.d_d = nullptr;
    pws.B_d = pws.N_d = nullptr;
    pws.b_active = nullptr;

    // Solver state
    pws.sol.B_h = idx();
    pws.sol.N_h = idx();

    // Library handles
    if (pws.lib.cublas_handle) cublasDestroy(pws.lib.cublas_handle);
    if (pws.lib.cub_temp_d) cudaFree(pws.lib.cub_temp_d);
    if (pws.lib.pricing_out_d) cudaFree(pws.lib.pricing_out_d);
    if (pws.lib.pricing_out_h) cudaFreeHost(pws.lib.pricing_out_h);
    if (pws.lib.h_harris_result) cudaFreeHost(pws.lib.h_harris_result);
    if (pws.lib.d_harris_result_mapped) cudaFree(pws.lib.d_harris_result_mapped);
    pws.lib = {};

    pws = {};
    pws.initialized = false;
}

