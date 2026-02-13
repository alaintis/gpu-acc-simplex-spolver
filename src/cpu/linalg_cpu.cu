#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "linalg_cpu.hpp"
#include "logging.hpp"

struct CPUWorkspace {
    double* A_d;
    double* b_d;
    int* ipiv_d;
    int* info_d;
    double* work_d;
    int work_size;
    cusolverDnHandle_t handle;
    bool initialized = false;
};

static CPUWorkspace ws;

void init_cpu_workspace(int n) {
    if (ws.initialized)
        return;

    cusolverDnCreate(&ws.handle);

    cudaMalloc((void**)&ws.A_d, n * n * sizeof(double));
    cudaMalloc((void**)&ws.b_d, n * sizeof(double));
    cudaMalloc((void**)&ws.ipiv_d, n * sizeof(int));
    cudaMalloc((void**)&ws.info_d, sizeof(int));

    cusolverDnDgetrf_bufferSize(ws.handle, n, n, ws.A_d, n, &ws.work_size);
    cudaMalloc((void**)&ws.work_d, ws.work_size * sizeof(double));

    ws.initialized = true;
}

void destroy_cpu_workspace() {
    if (!ws.initialized)
        return;

    cudaFree(ws.A_d);
    cudaFree(ws.b_d);
    cudaFree(ws.ipiv_d);
    cudaFree(ws.info_d);
    cudaFree(ws.work_d);
    cusolverDnDestroy(ws.handle);

    ws.initialized = false;
}

void cpu_factorize(int n, const double* A) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // Copy A to device
    cudaMemcpy(ws.A_d, A, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Factorize (A is overwritten with L and U)
    cusolverDnDgetrf(ws.handle, n, n, ws.A_d, n, ws.work_d, ws.ipiv_d, ws.info_d);

    // Check for singularities
    int info_h = 0;
    cudaMemcpy(&info_h, ws.info_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_h != 0 && logging::active) {
        std::cerr << "Warning: Singular matrix in factorization (info=" << info_h << ")"
                  << std::endl;
    }
}

void cpu_solve_prefactored(int n, const double* b, double* x, bool transpose) {
    if (!ws.initialized)
        throw std::runtime_error("Workspace not initialized");

    // Copy b to device
    cudaMemcpy(ws.b_d, b, n * sizeof(double), cudaMemcpyHostToDevice);

    // Select Operation: N (Normal) or T (Transpose)
    cublasOperation_t op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Solve using the existing factorization in ws.A_d and ws.ipiv_d
    cusolverDnDgetrs(ws.handle, op, n, 1, ws.A_d, n, ws.ipiv_d, ws.b_d, n, ws.info_d);

    // Copy result back
    cudaMemcpy(x, ws.b_d, n * sizeof(double), cudaMemcpyDeviceToHost);
}