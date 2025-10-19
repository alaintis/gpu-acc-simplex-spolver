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
 * n: Total number of variables.
 * m: Total number of equations.
 * 
 * A: list of columns of constraint matrix.
 * b: vector of constraints
 * c: vector of costs
 */

struct result solver(int n, int m, vector<vector<double>> A, vector<double> b, vector<double> c);
