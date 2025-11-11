#pragma once
#include <vector>
using std::vector;

struct result {
    bool success;
    vector<double> assignment;
};

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

 #ifdef __cplusplus
extern "C" {
#endif

struct result solver(int m, int n, vector<vector<double>> A, vector<double> b, vector<double> c, vector<double> x);

#ifdef __cplusplus
}
#endif