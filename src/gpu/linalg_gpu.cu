#include <assert.h>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <stdexcept>

#include "linalg_gpu.hpp"

// helpers
inline void cuda_check(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
inline void cublas_check(cublasStatus_t st) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error");
    }
}

// Use the same vec/mat typedefs from types.hpp
// GPU workspace shared by the .cu file
struct GPUWorkspace {
    int *ipiv_d, *info_d;
    int work_size;
    double* work_d;
    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
};

GPUWorkspace ws;

// kernel for vector subtraction
__global__ static void vec_sub_kernel(const double* a, const double* b, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] - b[i];
}

void init_gpu_workspace(int n, int m, int max_cols) { // GPUWorkspace &ws,
    cusolverDnCreate(&ws.cusolverHandle);
    cublasCreate(&ws.cublasHandle);

    cudaMalloc(&ws.ipiv_d, sizeof(int) * m);
    cudaMalloc(&ws.info_d, sizeof(int));

    cusolverDnDgetrf_bufferSize(ws.cusolverHandle, m, m, nullptr, m, &ws.work_size);
    cudaMalloc(&ws.work_d, sizeof(double) * ws.work_size);
}

void destroy_gpu_workspace() {
    cusolverDnDestroy(ws.cusolverHandle);
    cublasDestroy(ws.cublasHandle);

    cudaFree(&ws.ipiv_d);
    cudaFree(&ws.info_d);
    cudaFree(&ws.work_d);
}

void mv_solve_gpu(int n, const mat_cm_gpu A, const vec_gpu b, vec_gpu x) {
    cusolverDnDgetrf(ws.cusolverHandle, n, n, A, n, ws.work_d, ws.ipiv_d, ws.info_d);

    // --- LU info check ---
    int info_h;
    cudaMemcpy(&info_h, ws.info_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_h != 0) {
        printf("LU factorization failed! info = %d\n", info_h);
    }

    cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice);
    cusolverDnDgetrs(ws.cusolverHandle, CUBLAS_OP_N, n, 1, A, n, ws.ipiv_d, x, n, ws.info_d);
    cudaDeviceSynchronize();
}

void mv_mult_gpu(int m, int n, const mat_cm_gpu A, const vec_gpu x, vec_gpu y) {
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(ws.cublasHandle, CUBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, y, 1);
    cudaDeviceSynchronize();
}

void m_transpose_gpu(int m, int n, const mat_cm_gpu A, mat_cm_gpu AT) {
    double alpha = 1.0;
    double beta = 0.0;

    // Use cublasDgeam to compute transpose on GPU
    cublas_check(cublasDgeam(ws.cublasHandle,
                             CUBLAS_OP_T, // op(A) = A^T
                             CUBLAS_OP_T, // op(B) not used
                             n, // rows of C
                             m, // cols of C
                             &alpha, A,
                             m, // lda of original A (rows of A)
                             &beta, A,
                             m, // not used
                             AT,
                             n)); // ldc = rows of C = n
    cudaDeviceSynchronize();
}

void v_minus_gpu(int n, const vec_gpu a, const vec_gpu b, vec_gpu c) {
    int block = 256;
    int grid = (n + block - 1) / block;
    vec_sub_kernel<<<grid, block>>>(a, b, c, n);
    cudaDeviceSynchronize();
}
