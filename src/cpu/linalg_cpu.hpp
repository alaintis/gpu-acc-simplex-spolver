#pragma once
#include <vector>
using std::vector;

typedef vector<double> vec;
typedef vector<vec> mat;
typedef vector<double> mat_cm;

// Workspace Management
void init_cpu_workspace(int n);
void destroy_cpu_workspace();

/**
 * Factorize matrix A (LU decomposition).
 * Result is stored internally in the workspace.
 * A: pointer to nxn matrix (column major)
 */
void cpu_factorize(int n, const double* A);

/**
 * Solve linear system using the PREVIOUSLY computed factorization.
 * If transpose is true: Solves A^T x = b
 * If transpose is false: Solves A x = b
 * * b: pointer to n vector (input)
 * x: pointer to n vector (output)
 */
void cpu_solve_prefactored(int n, const double* b, double* x, bool transpose);