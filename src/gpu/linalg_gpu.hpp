#include "types.hpp"
typedef vec vec_gpu;
typedef mat_cm mat_cm_gpu;



/**
 * Matrix Vector solve. Solves the equation Ax = b and returns x.
 * 
 * A: nxn matrix column major order.
 * b: n vector
 * 
 * return x, the solution.
 */
vec_gpu mv_solve_gpu(int n, mat_cm_gpu& A, vec_gpu& b);


/**
 * Matrix vector multiplication.
 * 
 * A: mxn matrix column major order.
 * x: n vector
 * 
 * returns Ax, the solution a m vector.
 */
vec_gpu mv_mult_gpu(int m, int n, mat_cm_gpu& A, vec_gpu& x);

/**
 * Returns the transposed of the matrix.
 * 
 * A: mxn matrix column major order.
 * 
 * return A^T, the solution nxm matrix column major order.
 */
mat_cm_gpu m_transpose_gpu(int m, int n, mat_cm_gpu& A);


/**
 * Returns a - b
 */
vec_gpu v_minus_gpu(int n, vec_gpu& a, vec_gpu& b);

// Initialize global GPU workspace (allocates device buffers, cuBLAS/cuSOLVER handles)
void init_gpu_workspace(int n, int m, int max_cols);
void destroy_gpu_workspace(void);
