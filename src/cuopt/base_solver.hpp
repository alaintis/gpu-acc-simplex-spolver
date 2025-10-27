#pragma once
#include "../solver.hpp"

/**
 * Simplex Solver. Assumed interface
 * 
 * min c^T x
 * Ax <= b
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

struct result base_solver(int m, int n, vector<vector<double>> A, vector<double> b, vector<double> c);
