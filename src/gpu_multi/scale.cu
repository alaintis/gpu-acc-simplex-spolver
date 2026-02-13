#include "scale.hpp"

// Scaling Kernels
// ====

// Find row_scale for each row. 1 block per row.
__global__ void row_scale_kernel(
    const double* __restrict__ A,  // column-major, m × n_total
    double* __restrict__ row_scale,
    int m, int n_total
) {
    int row = blockIdx.x;
    if (row >= m) return;

    extern __shared__ double sdata[];
    int tid = threadIdx.x;

    double local_max = 0.0;

    // Each thread scans over its columns
    for (int col = tid; col < n_total; col += blockDim.x) {
        double v = fabs(A[row + col * m]);
        if (v > local_max) local_max = v;
    }

    // Load into shared memory
    sdata[tid] = local_max;
    __syncthreads();

    // Reduction to find max across block
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            double other = sdata[tid + s];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }

    // Thread 0 writes row result (with clamping)
    if (tid == 0) {
        double r = sdata[0];
        if (r < 1e-9) r = 1.0;
        row_scale[row] = r;
    }
}

// Find row_scale for each col. 1 block per col.
__global__ void col_scale_kernel(
    const double* __restrict__ A,  // column-major, m × n_total
    const double* __restrict__ row_scale,
    double* __restrict__ col_scale,
    int m, int n_total
) {
    int col = blockIdx.x;
    if (col >= n_total) return;

    extern __shared__ double sdata[];
    int tid = threadIdx.x;

    double local_max = 0.0;

    // Each thread scans over its columns
    for (int row = tid; row < m; row += blockDim.x) {
        double v = fabs(A[row + col * m] / row_scale[row]);
        if (v > local_max) local_max = v;
    }

    // Load into shared memory
    sdata[tid] = local_max;
    __syncthreads();

    // Reduction to find max across block
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            double other = sdata[tid + s];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }

    // Thread 0 writes row result (with clamping)
    if (tid == 0) {
        double r = sdata[0];
        if (r < 1e-9) r = 1.0;
        col_scale[col] = r;
    }
}


__global__ void apply_scale_kernel(
    int m, int n_total, const double *row_scale, const double *col_scale,
    const double* __restrict__ A, double* __restrict__ A_scaled,
    const double* __restrict__ b, double* __restrict__ b_scaled,
    const double* __restrict__ c, double* __restrict__ c_scaled,
    const double* __restrict__ x, double* __restrict__ x_scaled
) {
    int total = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < m * n_total; i += total) {
        int row = i % m;
        int col = i / m;
        A_scaled[i] = A[i] / (row_scale[row] * col_scale[col]);
    }

    for(int i = idx; i < m; i += total) {
        b_scaled[i] = b[i]/row_scale[i];
    }

    for(int i = idx; i < n_total; i += total) {
        c_scaled[i] = c[i]/col_scale[i];
        x_scaled[i] = x[i]*col_scale[i];
    }
}

// Unscale the solution using a single block.
__global__ void unscale_kernel(
    int m,
    int n_total,
    const int* B,
    const double* x_B,
    const double* col_scale,
    double* x_out
) {
    if(blockIdx.x > 0) return;

    int idx = threadIdx.x;
    for(int i = idx; i < n_total; i+=blockDim.x) {
        x_out[i] = 0.0;
    }

    __syncthreads();

    for (int k = idx; k < m; k+=blockDim.x) {
        // x = x_new / C
        int col = B[k];
        double v = x_B[k] / col_scale[col];
        if (v > -1e-9 && v < 0.0) v = 0.0;
        x_out[col] = v;
    }
}


// Scaling LOGIC
// ====

void gpu_scale_problem(const SharedWorkspace &ws, cudaStream_t stream) {
    int m = ws.m, n = ws.n;

    const OriginalWs &origin = ws.origin;
    const ScalingWs &scale = ws.scale;

    int threads = 256;
    row_scale_kernel<<<m, threads, threads * sizeof(double), stream>>>(
        origin.A_d, scale.row_scale_d, m, n
    );

    col_scale_kernel<<<n, threads, threads * sizeof(double), stream>>>(
        origin.A_d, scale.row_scale_d, scale.col_scale_d, m, n
    );

    int blocks = 1024;
    apply_scale_kernel<<<blocks, threads>>>(
        m, n, scale.row_scale_d, scale.col_scale_d,
        origin.A_d, scale.A_d,
        origin.b_d, scale.b_d,
        origin.c_d, scale.c_d,
        origin.x_d, scale.x_d
    );
}

void gpu_unscale(const SharedWorkspace &ws, const PrivateWorkspace &pws, double *x_out) {
    int m = ws.m, n = ws.n;
    const OriginalWs &origin = ws.origin;
    const ScalingWs &scale = ws.scale;

    unscale_kernel<<<1, 256, 0, pws.stream>>>(m, n, pws.B_d, pws.x_B_d, scale.col_scale_d, origin.x_d);
    cudaMemcpyAsync(x_out, origin.x_d, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(pws.stream);
}
