#include "types.hpp"
typedef double *vec_gpu;
typedef double *mat_cm_gpu;



/**
 * Matrix Vector solve. Solves the equation Ax = b and returns x.
 * 
 * A: nxn matrix column major order.
 * b: n vector
 * 
 * x, the solution.
 */
void mv_solve_gpu(int n, const mat_cm_gpu A, const vec_gpu b, vec_gpu x);


/**
 * Matrix vector multiplication.
 * 
 * A: mxn matrix column major order.
 * x: n vector
 * 
 * y: Ax, the solution a m vector.
 */
void mv_mult_gpu(int m, int n, const mat_cm_gpu A, const vec_gpu x, vec_gpu y);

/**
 * Returns the transposed of the matrix.
 * 
 * A: mxn matrix column major order.
 * 
 * AT: A^T, the solution nxm matrix column major order.
 */
void m_transpose_gpu(int m, int n, const mat_cm_gpu A, mat_cm_gpu AT);


/**
 * c: a - b
 */
void v_minus_gpu(int n, const vec_gpu a, const vec_gpu b, vec_gpu c);

// Initialize global GPU workspace (allocates device buffers, cuBLAS/cuSOLVER handles)
void init_gpu_workspace(int n, int m, int max_cols);
void destroy_gpu_workspace(void);
