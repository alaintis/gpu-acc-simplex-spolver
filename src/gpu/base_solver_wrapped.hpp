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
                          const vector<vector<double>>& A,
                          const vector<double>& b,
                          const vector<double>& c);
