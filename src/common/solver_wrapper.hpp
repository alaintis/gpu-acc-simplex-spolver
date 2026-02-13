#pragma once
#include <vector>

#include "solver.hpp"
using std::vector;

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
 * x: initial feasible solution
 */

struct result solver_wrapper(int m,
                             int n,
                             const mat_csc& A,
                             const vector<double>& b,
                             const vector<double>& c);
