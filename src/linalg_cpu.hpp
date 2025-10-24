#include <vector>
using std::vector;
typedef vector<double> vec;
typedef vector<vec> mat;

/**
 * Matrix column major order
 * [0, 1]
 * [2, 3]
 * 
 * Is stored as [0, 2, 1, 3].
 */ 

typedef vector<double> mat_cm;

/**
 * Matrix Vector solve. Solves the equation Ax = b and returns x.
 * 
 * A: nxn matrix column major order.
 * b: n vector
 * 
 * return x, the solution.
 */
vec mv_solve(int n, mat_cm& A, vec& b);


/**
 * Matrix vector multiplication.
 * 
 * A: mxn matrix column major order.
 * x: n vector
 * 
 * returns Ax, the solution a m vector.
 */
vec mv_mult(int m, int n, mat_cm& A, vec& x);

/**
 * Returns the transposed of the matrix.
 * 
 * A: mxn matrix column major order.
 * 
 * return A^T, the solution nxm matrix column major order.
 */
mat_cm m_transpose(int m, int n, mat_cm& A);


/**
 * Returns a - b
 */
vec v_minus(int n, vec& a, vec& b);
