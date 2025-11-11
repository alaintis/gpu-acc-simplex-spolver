#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <cublas_v2.h>

#include "linalg_gpu.hpp"
#include <assert.h>
#include <cstring>
#include <iostream>

// helpers
inline void cuda_check(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
inline void cublas_check(cublasStatus_t st)
{
    if (st != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("cuBLAS error");
    }
}

// Use the same vec/mat typedefs from types.hpp
// GPU workspace shared by the .cu file
struct GPUWorkspace
{
    double *A_d, *A_solve, *b_d, *x_d, *y_d, *out_d;
    double *AT_d; // buffer for transpose
    int *ipiv_d, *info_d;
    int work_size;
    double *work_d;
    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
};

GPUWorkspace ws;

// kernel for vector subtraction
__global__ static void vec_sub_kernel(const double *a, const double *b, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] - b[i];
}

void init_gpu_workspace(int n, int m, int max_cols)
{ // GPUWorkspace &ws,
    cusolverDnCreate(&ws.cusolverHandle);
    cublasCreate(&ws.cublasHandle);

    int max = (n > m) ? n : m;

    cudaMalloc(&ws.A_d, sizeof(double) * max * m);    // for A_B or A_N^T blocks
    cudaMalloc(&ws.A_solve, sizeof(double) * m * m);  // for A_B or A_N^T blocks
    cudaMalloc(&ws.b_d, sizeof(double) * max_cols);   // right-hand side for mv_solve
    cudaMalloc(&ws.x_d, sizeof(double) * max_cols);   // for mv_mult input vector
    cudaMalloc(&ws.y_d, sizeof(double) * max_cols);   // for mv_mult output
    cudaMalloc(&ws.out_d, sizeof(double) * max_cols); // for v_minus output
    cudaMalloc(&ws.AT_d, sizeof(double) * max * m);   // enough for transpose
    cudaMalloc(&ws.ipiv_d, sizeof(int) * m);
    cudaMalloc(&ws.info_d, sizeof(int));

    cusolverDnDgetrf_bufferSize(ws.cusolverHandle, m, m, ws.A_solve, m, &ws.work_size);
    cudaMalloc(&ws.work_d, sizeof(double) * ws.work_size);
}

void destroy_gpu_workspace()
{
    cusolverDnDestroy(ws.cusolverHandle);
    cublasDestroy(ws.cublasHandle);

    cudaFree(&ws.A_d);     // for A_B or A_N^T blocks
    cudaFree(&ws.A_solve); // for m * m matrices.
    cudaFree(&ws.b_d);     // right-hand side for mv_solve
    cudaFree(&ws.x_d);     // for mv_mult input vector
    cudaFree(&ws.y_d);     // for mv_mult output
    cudaFree(&ws.out_d);   // for v_minus output
    cudaFree(&ws.AT_d);    // enough for transpose
    cudaFree(&ws.ipiv_d);
    cudaFree(&ws.info_d);
    cudaFree(&ws.work_d);
}

vec_gpu mv_solve_gpu(int n, mat_cm_gpu &A, vec_gpu &b)
{
    cudaMemcpy(ws.A_solve, A.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(ws.b_d, b.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    cusolverDnDgetrf(ws.cusolverHandle, n, n, ws.A_solve, n, ws.work_d, ws.ipiv_d, ws.info_d);

    // --- LU info check ---
    int info_h;
    cudaMemcpy(&info_h, ws.info_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_h != 0)
    {
        printf("LU factorization failed! info = %d\n", info_h);
    }

    cusolverDnDgetrs(ws.cusolverHandle, CUBLAS_OP_N, n, 1, ws.A_solve, n, ws.ipiv_d, ws.b_d, n, ws.info_d);

    vec_gpu x(n);
    cudaMemcpy(x.data(), ws.b_d, sizeof(double) * n, cudaMemcpyDeviceToHost);
    return x;
}

vec_gpu mv_mult_gpu(int m, int n, mat_cm_gpu &A, vec_gpu &x)
{
    cudaMemcpy(ws.A_d, A.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(ws.x_d, x.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    double alpha = 1.0, beta = 0.0;
    cublasDgemv(ws.cublasHandle, CUBLAS_OP_N, m, n, &alpha, ws.A_d, m, ws.x_d, 1, &beta, ws.y_d, 1);

    vec_gpu y(m);
    cudaMemcpy(y.data(), ws.y_d, sizeof(double) * m, cudaMemcpyDeviceToHost);
    return y;
}

mat_cm_gpu m_transpose_gpu(int m, int n, mat_cm_gpu &A)
{
    assert(A.size() == (size_t)m * n);

    mat_cm_gpu AT(n * m);

    // Copy host -> device

    cuda_check(cudaMemcpy(ws.A_d, A.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice));

    double alpha = 1.0;
    double beta = 0.0;

    // Use cublasDgeam to compute transpose on GPU
    cublas_check(cublasDgeam(ws.cublasHandle,
                             CUBLAS_OP_T, // op(A) = A^T
                             CUBLAS_OP_T, // op(B) not used
                             n,           // rows of C
                             m,           // cols of C
                             &alpha,
                             ws.A_d,
                             m, // lda of original A (rows of A)
                             &beta,
                             ws.A_d,
                             m, // not used
                             ws.AT_d,
                             n)); // ldc = rows of C = n

    // Copy back device -> host
    cuda_check(cudaMemcpy(AT.data(), ws.AT_d, sizeof(double) * n * m, cudaMemcpyDeviceToHost));

    return AT;
}

vec_gpu v_minus_gpu(int n, vec_gpu &a, vec_gpu &b)
{
    cudaMemcpy(ws.A_d, a.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(ws.b_d, b.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    vec_sub_kernel<<<grid, block>>>(ws.A_d, ws.b_d, ws.out_d, n);
    cudaDeviceSynchronize();

    vec_gpu diff(n);
    cudaMemcpy(diff.data(), ws.out_d, sizeof(double) * n, cudaMemcpyDeviceToHost);
    return diff;
}
