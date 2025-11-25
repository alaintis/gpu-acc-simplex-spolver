#pragma once
#include <vector>
using std::vector;

struct result {
    bool success;
    vector<double> assignment;
    vector<int> basis;
    bool basis_split_found; // indicates if a valid basis/non-basis split was found
};

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

#ifdef __cplusplus
extern "C" {
#endif

struct result solver(int m,
                     int n,
                     const vector<vector<double>>& A,
                     const vector<double>& b,
                     const vector<double>& c,
                     vector<double>& x,
                     const vector<int>&B
                    );

#ifdef __cplusplus
}
#endif
