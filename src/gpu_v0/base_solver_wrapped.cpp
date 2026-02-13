#include <iostream>
#include "base_solver_wrapped.hpp"

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
                          const mat_csc& A_in,
                          const vector<double>& b_in,
                          const vector<double>& c_in) {
    // Step 1. Build & Solve Auxiliary Problem
    int n_aux = m + n; // New number of variables
    mat_csc A_aux = A_in;
    vector<double> c_aux(n_aux, 0.0);
    vector<double> x_aux(n_aux, 0.0);
    vector<int> B_aux(m);
    int nnz = A_in.col_ptr[n];

    for (int i = 0; i < m; i++) {
        double multiplier = (b_in[i] > 0) ? 1.0 : -1.0;
        A_aux.col_ptr.push_back(nnz + 1);
        A_aux.row_idx.push_back(i);
        A_aux.values.push_back(multiplier);
        nnz += 1;
        
        c_aux[n + i] = 1.0;
        x_aux[n + i] = multiplier * b_in[i]; // To satisfy the constraints (a_i = b_i)
        B_aux[i] = n + i;
    }

    std::cout << "Solving auxiliary problem to find feasible solution x..." << std::endl;
    result res = base_solver(m, n_aux, A_aux, b_in, c_aux, x_aux, B_aux);
    if (!res.success) {
        std::cout << "Phase I solver failed." << std::endl;
        return res;
    }

    // Check Phase 1 Objective
    double phase1_obj = 0.0;
    for (int i = n; i < n_aux; ++i) {
        phase1_obj += res.assignment[i];
    }
    // If the aux score is > 0, the original problem is infeasible
    if (phase1_obj > 1e-4) {
        std::cout << "Infeasible: Phase 1 objective " << phase1_obj << " > 0." << std::endl;
        res.success = false;
        return res;
    }
    std::cout << "Auxiliary problem solved. Phase 1 Obj: " << phase1_obj << std::endl;

    // Step 2. Prepare Phase 2 Problem
    vector<int> B_phase2 = res.basis;
    bool artificial_in_basis = false;
    for (int i = 0; i < m; ++i) {
        if (B_phase2[i] >= n) {
            artificial_in_basis = true;
            // This is just a warning; Should not really happen if the solver is correct (unless
            // rounding, floating-point errors, pivot noise, etc)
            if (std::abs(res.assignment[B_phase2[i]]) > 1e-5) {
                std::cout << "Warning: Non-zero artificial " << B_phase2[i] << " in basis!"
                          << std::endl;
            }
        }
    }

    // Step 3. Solve Phase 2 Problem
    vector<double> x_phase2 = res.assignment;

    if (artificial_in_basis) {
        // This should be optimized!
        std::cout << "Phase 1 has artificials in basis. Using Big-M Method for Phase 2."
                  << std::endl;

        vector<double> c_bigM(n_aux);
        for (int i = 0; i < n; i++) {
            c_bigM[i] = c_in[i];
        }
        for (int i = n; i < n_aux; i++) {
            c_bigM[i] = 1e9; // Huge cost penalty
        }

        result res2 = base_solver(m, n_aux, A_aux, b_in, c_bigM, x_phase2, B_phase2);

        // Trim result back to n variables
        if (res2.success) {
            res2.assignment.resize(n);
        }
        return res2;
    } else {
        // Standard Phase 2: No artificials in basis, use original A and c.
        x_phase2.resize(n);
        return base_solver(m, n, A_in, b_in, c_in, x_phase2, B_phase2);
    }
}
