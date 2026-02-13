#pragma once
#include "../common/solver.hpp"

/**
 * Simplex Solver. Assumed interface
 *
 * min c^T x
 * Ax = b
 * x >= 0
 *
 * 0 is assumed to be a feasible solution.
 *
 * m: Total number of equations.
 * n: Total number of variables.
 *
 * A: list of columns of constraint matrix.
 * b: vector of constraints
 * c: vector of costs
 */

struct result base_solver_wrapped(int m,
                          int n,
                          const mat_csc& A,
                          const vector<double>& b,
                          const vector<double>& c);


#ifdef __cplusplus
extern "C" {
#endif

struct result base_solver(int m,
                     int n,
                     const mat_csc& A,
                     const vector<double>& b,
                     const vector<double>& c,
                     vector<double>& x,
                     const vector<int>&B
                    );

#ifdef __cplusplus
}
#endif
