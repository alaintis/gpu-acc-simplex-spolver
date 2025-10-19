#include <cuda_runtime.h>

#include "solver.hpp"

__device__ int val = 0;

__global__ void kernel(int a) {
    atomicExch(&val, a+1);
}


struct result solver(int n, int m, vector<vector<double>> A, vector<double> b, vector<double> c) {
    int zero = 0;
    cudaMemcpyToSymbol(val, &zero, sizeof(int));

    kernel<<<1, 1>>>(n);
    cudaDeviceSynchronize();

    int out = 32;
    cudaMemcpyFromSymbol(&out, val, sizeof(int));
    struct result r;
    return r;
}
