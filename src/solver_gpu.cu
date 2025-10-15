#include <cuda_runtime.h>

#include "solver.hpp"

__device__ int val = 0;

__global__ void kernel(int a) {
    atomicExch(&val, a+1);
}

int solver(int in) {
    int zero = 0;
    cudaMemcpyToSymbol(val, &zero, sizeof(int));

    kernel<<<1, 1>>>(in);
    cudaDeviceSynchronize();

    int out = 32;
    cudaMemcpyFromSymbol(&out, val, sizeof(int));
    return out;
}
