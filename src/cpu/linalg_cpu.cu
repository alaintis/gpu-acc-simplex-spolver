#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>

#include "linalg_cpu.hpp"
#include <assert.h>

vec mv_solve(int n, mat_cm& A, vec& b) {
    assert(A.size() == n*n);
    assert(b.size() == n);
    
    double *A_d = nullptr;
    double *b_d = nullptr;
    int *ipiv_d = nullptr;
    int *info_d = nullptr;

    int work_size = 0;
    double* work_d = nullptr;
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    cudaMalloc((void**)&A_d, n * n * sizeof(double));
    cudaMalloc((void**)&b_d, n * sizeof(double));
    cudaMalloc((void**)&ipiv_d, n * sizeof(int));
    cudaMalloc((void**)&info_d, sizeof(int));

    cudaMemcpy(A_d, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    cusolverDnDgetrf_bufferSize(handle, n, n, A_d, n, &work_size);
    cudaMalloc((void**)&work_d, work_size * sizeof(double));


    // LU decomposition.
    cusolverDnDgetrf(handle, n, n, A_d, n, work_d, ipiv_d, info_d);
    cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A_d, n, ipiv_d, b_d, n, info_d);

    
    cudaFree(A_d);
    cudaFree(ipiv_d);
    cudaFree(info_d);
    cudaFree(work_d);
    cusolverDnDestroy(handle);
    
    vec x(n);
    cudaMemcpy(x.data(), b_d, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(b_d);
    return x;
}


vec mv_mult(int m, int n, mat_cm& A, vec& x) {
    assert(A.size() == n * m);
    assert(x.size() == n);
    vec y(m);

    for(int i = 0; i < m; i++) {
        double sum = 0;
        for(int j = 0; j < n; j++) {
            sum += A[i + j * m] * x[j];
        }
        y[i] = sum;
    }

    return y;
}

mat_cm m_transpose(int m, int n, mat_cm& A) {
    assert(A.size() == n * m);

    mat_cm AT(n*m);

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            AT[j + i * n] = A[i + j * m];
        }
    }

    return AT;
}

vec v_minus(int n, vec& a, vec& b) {
    vec diff(n);

    for(int i = 0; i < n; i++) {
        diff[i] = a[i] - b[i];
    }

    return diff;
}
