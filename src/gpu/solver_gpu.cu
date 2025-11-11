#include <assert.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>

#include "linalg_gpu.hpp"
#include "logging.hpp"
#include "solver.hpp"

typedef vector<int> idx;

std::vector<double> pack_A_column_major(int m, int ncols, const std::vector<std::vector<double>>& Acols) {
    std::vector<double> Ahost((size_t)m * ncols);
    for (int col = 0; col < ncols; ++col) {
        for (int row = 0; row < m; ++row) {
            Ahost[(size_t)col * m + row] = Acols[col][row]; // column-major: leading dim = m
        }
    }
    return Ahost;
}

static double eps = 1e-6;

vec Ax_mult(int m, int n, mat& A, vec& x) {
    vec y(m);

    for (int i = 0; i < m; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[j][i] * x[j];
        }
        y[i] = sum;
    }

    return y;
}

// Implementation directly following the Rice Notes:
// https://www.cmor-faculty.rice.edu/~yzhang/caam378/Notes/Save/note2.pdf

// Device helpers: atomicMin/Max for double using atomicCAS on 64-bit
static __inline__ __device__ unsigned long long double_to_ull(double d) {
    unsigned long long ull;
    memcpy(&ull, &d, sizeof(double));
    return ull;
}
static __inline__ __device__ double ull_to_double(unsigned long long ull) {
    double d;
    memcpy(&d, &ull, sizeof(double));
    return d;
}

__device__ double atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    unsigned long long val_ull = double_to_ull(val);
    do {
        assumed = old;
        double assumed_d = ull_to_double(assumed);
        if (assumed_d <= val) break; // already smaller
        old = atomicCAS(address_as_ull, assumed, val_ull);
    } while (assumed != old);
    return ull_to_double(old);
}

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    unsigned long long val_ull = double_to_ull(val);
    do {
        assumed = old;
        double assumed_d = ull_to_double(assumed);
        if (assumed_d >= val) break; // already larger
        old = atomicCAS(address_as_ull, assumed, val_ull);
    } while (assumed != old);
    return ull_to_double(old);
}

// kernel: compute min over arr[0..n-1], store in out_min[0] (initially +inf)
__global__ void kernel_min_val(const double* arr, int n, double* out_min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    double local_min = INFINITY;
    for (int i = idx; i < n; i += stride) {
        double v = arr[i];
        if (v < local_min) local_min = v;
    }
    if (local_min < INFINITY) {
        atomicMinDouble(out_min, local_min);
    }
}

// kernel: find first index i where fabs(arr[i] - val) <= tol, store into out_idx (atomic)
__global__ void kernel_find_index_value(const double* arr, int n, double val, double tol, int* out_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        double v = arr[i];
        if (fabs(v - val) <= tol) {
            // try to set out_idx from -1 to i
            int old = atomicCAS(out_idx, -1, i);
            if (old == -1) {
                // we set it
            }
            return; // if another thread already set, we're done
        }
    }
}

// kernel: compute max over arr[0..n-1], store in out_max[0] (initially -inf)
__global__ void kernel_max_val(const double* arr, int n, double* out_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    double local_max = -INFINITY;
    for (int i = idx; i < n; i += stride) {
        double v = arr[i];
        if (v > local_max) local_max = v;
    }
    if (local_max > -INFINITY) {
        atomicMaxDouble(out_max, local_max);
    }
}

// kernel: compute min ratio xB[i]/d[i] for d[i] > eps, store ratio in out_min_ratio (initially +inf)
__global__ void kernel_min_ratio(const double* xB, const double* d, int m, double eps_local, double* out_min_ratio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    double local_min = INFINITY;
    for (int i = idx; i < m; i += stride) {
        double di = d[i];
        if (di > eps_local) {
            double ratio = xB[i] / di;
            if (ratio < local_min) local_min = ratio;
        }
    }
    if (local_min < INFINITY) {
        atomicMinDouble(out_min_ratio, local_min);
    }
}

// kernel: find first index where fabs(xB[i]/d[i] - minratio) <= tol, and d[i] > eps -> store index in out_idx
__global__ void kernel_find_index_ratio(const double* xB, const double* d, int m, double minratio, double eps_local, double tol, int* out_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < m; i += stride) {
        double di = d[i];
        if (di > eps_local) {
            double ratio = xB[i] / di;
            if (fabs(ratio - minratio) <= tol) {
                int old = atomicCAS(out_idx, -1, i);
                if (old == -1) {
                    return;
                }
            }
        }
    }
}

extern "C" struct result solver(int m, int n, mat A, vec b, vec c, vec x) {
    assert(A.size() == n);
    for (int i = 0; i < n; i++) {
        assert(A[i].size() == m);
    }
    assert(b.size() == m);
    assert(c.size() == n);
    assert(x.size() == n);

    vec y_initial = Ax_mult(m, n, A, x);

    for(int i = 0; i < m; i++){
        vec e_i(m, 0);
        e_i[i] = 1;
        A.push_back(e_i);
        c.push_back(0);
        x.push_back(b[i] - y_initial[i]);
    }

    int zero_count = 0;
    for (int i = 0; i < n + m; i++) {
        if (std::abs(x[i]) < eps) {
            zero_count += 1;
        }
    }

    if(zero_count != n) {
        std::cout << "No Non-Basis / Basis split found." << std::endl;
        result res;
        res.success = false;
        return res;
    }

    idx B(m);
    idx N(n);
    int n_count = 0;
    int b_count = 0;
    for (int i = 0; i < n + m; i++) {
        if (std::abs(x[i]) < eps) {
            N[n_count++] = i;
        } else {
            B[b_count++] = i;
        }
    }


    int n_total = n + m;
    init_gpu_workspace(n_total, m, n_total);
    std::vector<double> A_host = pack_A_column_major(m, n_total, A);

    // main variables.
    mat_cm_gpu A_B;
    vec_gpu    c_B;
    vec_gpu    x_B;

    mat_cm_gpu A_N;
    vec_gpu    c_N;

    // temporary variables
    mat_cm_gpu A_BT;
    vec_gpu y;
    mat_cm_gpu A_NT;
    vec_gpu tmp;
    vec_gpu s_N;
    vec_gpu Ajj;
    vec_gpu d;

    cudaMalloc(&A_B, sizeof(double) * m*m);
    cudaMalloc(&c_B, sizeof(double) * m);
    cudaMalloc(&x_B, sizeof(double) * m);
    cudaMalloc(&A_N, sizeof(double) * m*n);
    cudaMalloc(&c_N, sizeof(double) * n);

    cudaMalloc(&A_BT, sizeof(double) * m*m);
    cudaMalloc(&y, sizeof(double) * m);
    cudaMalloc(&A_NT, sizeof(double) * n*m);
    cudaMalloc(&tmp, sizeof(double) * n);
    cudaMalloc(&s_N, sizeof(double) * n);
    cudaMalloc(&Ajj, sizeof(double) * m);
    cudaMalloc(&d, sizeof(double) * m);

    // Extra device scalars for reductions
    double* d_min_scalar; // for s_N min
    double* d_max_scalar; // for d max
    double* d_minratio_scalar; // for min ratio
    int* d_index_scalar;

    cudaMalloc(&d_min_scalar, sizeof(double));
    cudaMalloc(&d_max_scalar, sizeof(double));
    cudaMalloc(&d_minratio_scalar, sizeof(double));
    cudaMalloc(&d_index_scalar, sizeof(int));

    for(int iter = 0; iter < 100; iter++) {
        logging::log("B", B);
        logging::log("N", N);

        // Build A_B and A_N (column-major)
        for (int col = 0; col < m; col++) {
            int src = B[col];
            cudaMemcpy(&A_B[(size_t)col*m], &A_host[(size_t)src*m], sizeof(double)*m, cudaMemcpyHostToDevice);
            cudaMemcpy(&c_B[col], &c[src], sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(&x_B[col], &x[src], sizeof(double), cudaMemcpyHostToDevice);
        }
        for (int col = 0; col < n; col++) {
            int src = N[col];
            cudaMemcpy(&A_N[(size_t)col*m], &A_host[(size_t)src*m], sizeof(double)*m, cudaMemcpyHostToDevice);
            cudaMemcpy(&c_N[col], &c[src], sizeof(double), cudaMemcpyHostToDevice);
        }

        // 1. Dual estimates.
        m_transpose_gpu(m, m, A_B, A_BT);
        mv_solve_gpu(m, A_BT, c_B, y);
        m_transpose_gpu(m, n, A_N, A_NT);
        mv_mult_gpu(n, m, A_NT, y, tmp);
        v_minus_gpu(n, c_N, tmp, s_N);

        // kernelized, compute min
        // initialize scalar
        double init_min = INFINITY;
        cudaMemcpy(d_min_scalar, &init_min, sizeof(double), cudaMemcpyHostToDevice);

        // choose kernel launch params
        int block = 256;
        int grid = (n + block - 1) / block;
        if (grid > 512) grid = 512;

        kernel_min_val<<<grid, block>>>(s_N, n, d_min_scalar);
        cudaDeviceSynchronize();

        double sN_min_val;
        cudaMemcpy(&sN_min_val, d_min_scalar, sizeof(double), cudaMemcpyDeviceToHost);

        vec s_N_h(n);
        cudaMemcpy(s_N_h.data(), s_N, sizeof(double) * n, cudaMemcpyDeviceToHost);
        logging::log("s_N_h", s_N_h);

        // 2. Check optimality.
        bool optimal = true;
        // use on-device result to determine if there's a negative reduced cost
        if (sN_min_val < -eps) optimal = false;
        if(optimal) {
            x.resize(n);
            struct result res = {.success = true, .assignment = x};
            destroy_gpu_workspace();

            // free scalars
            cudaFree(d_min_scalar);
            cudaFree(d_max_scalar);
            cudaFree(d_minratio_scalar);
            cudaFree(d_index_scalar);

            return res;
        }

        // 3. Selection of entering variable.

        //TODO: not kernelized yet
        int j_i = 0;
        for(int i = 0; i < n; i++) {
            if(s_N_h[j_i] > s_N_h[i]) j_i = i;
        }
        int jj = N[j_i];

        // 4. Compute step
        cudaMemcpy(Ajj, A[jj].data(), sizeof(double) * m, cudaMemcpyHostToDevice);
        mv_solve_gpu(m, A_B, Ajj, d);

        vec d_h(m);
        cudaMemcpy(d_h.data(), d, sizeof(double) * m, cudaMemcpyDeviceToHost);
        logging::log("d", d_h);

        // 5. Check unboundedness (kernelized)
        // compute max(d)
        double init_max = -INFINITY;
        cudaMemcpy(d_max_scalar, &init_max, sizeof(double), cudaMemcpyHostToDevice);
        int block2 = 256;
        int grid2 = (m + block2 - 1) / block2;
        if (grid2 > 512) grid2 = 512;

        kernel_max_val<<<grid2, block2>>>(d, m, d_max_scalar);
        cudaDeviceSynchronize();

        double d_max_val;
        cudaMemcpy(&d_max_val, d_max_scalar, sizeof(double), cudaMemcpyDeviceToHost);

        bool unbounded = true;
        if (d_max_val > eps) unbounded = false;
        if(unbounded) {
            struct result res;
            res.success = false;
            destroy_gpu_workspace();

            // free scalars
            cudaFree(d_min_scalar);
            cudaFree(d_max_scalar);
            cudaFree(d_minratio_scalar);
            cudaFree(d_index_scalar);

            return res;
        }

        // 6. leaving Variable selection (kernelized)
        double init_minratio = INFINITY;
        cudaMemcpy(d_minratio_scalar, &init_minratio, sizeof(double), cudaMemcpyHostToDevice);

        kernel_min_ratio<<<grid2, block2>>>(x_B, d, m, eps, d_minratio_scalar);
        cudaDeviceSynchronize();

        double minratio_val;
        cudaMemcpy(&minratio_val, d_minratio_scalar, sizeof(double), cudaMemcpyDeviceToHost);

        int init_idx = -1;
        cudaMemcpy(d_index_scalar, &init_idx, sizeof(int), cudaMemcpyHostToDevice);

        double tol = 1e-12;
        kernel_find_index_ratio<<<grid2, block2>>>(x_B, d, m, minratio_val, eps, tol, d_index_scalar);
        cudaDeviceSynchronize();

        int r_dev;
        cudaMemcpy(&r_dev, d_index_scalar, sizeof(int), cudaMemcpyDeviceToHost);

        if (r_dev == -1) {
            vec x_B_h(m);
            cudaMemcpy(x_B_h.data(), x_B, sizeof(double) * m, cudaMemcpyDeviceToHost);
            vec d_h2(m);
            cudaMemcpy(d_h2.data(), d, sizeof(double) * m, cudaMemcpyDeviceToHost);

            int r = -1;
            for(int i = 0; i < m; i++) {
                if(d_h2[i] > eps && (r == -1 || x_B_h[i]/d_h2[i] < x_B_h[r]/d_h2[r])) {
                    r = i;
                }
            }
            assert(r >= 0);
            r_dev = r;
        }

        int r = r_dev;
        // copy x_B to host for ratio and update (we need x_B_h for the actual x updates below)
        vec x_B_h(m);
        cudaMemcpy(x_B_h.data(), x_B, sizeof(double) * m, cudaMemcpyDeviceToHost);

        int ii = B[r];
        double tt = x_B_h[r]/d_h[r];
        // 7. Update variables
        x[jj] = tt;
        // x_B <== x_B - tt * d
        for(int i = 0; i < m; i++) {
            x[B[i]] = x_B_h[i] - tt * d_h[i];
        }

        // 8. Update Basis
        N[n_count == 0 ? j_i : j_i] = ii; // keep N array updated (n_count doesn't change)
        N[j_i] = ii;
        B[r] = jj;
    }

    result res;
    res.success = false;
    destroy_gpu_workspace();

    // free scalars
    cudaFree(d_min_scalar);
    cudaFree(d_max_scalar);
    cudaFree(d_minratio_scalar);
    cudaFree(d_index_scalar);

    return res;
}

