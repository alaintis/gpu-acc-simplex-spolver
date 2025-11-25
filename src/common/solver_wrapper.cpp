#include "solver_wrapper.hpp"

#include <algorithm>
#include <iostream>
#include <string>

#include "linprog.hpp"
#include "logging.hpp"

#ifndef SOLVER_BACKEND_NAME
#error "SOLVER_BACKEND_NAME is not defined. CMake config is broken."
#endif

/**
 * We check the backend name at compile time.
 * If the backend is "cuopt", we just pass the problem through,
 * as cuOpt has its own robust presolver and doesn't need our
 * manual Phase I logic.
 */
#if 1 // This is a bit of a hack to check string equality at compile time
#define IS_CUOPT_BACKEND (SOLVER_BACKEND_NAME[0] == 'c' && SOLVER_BACKEND_NAME[1] == 'u')
#else
#define IS_CUOPT_BACKEND (false)
#endif

/**
 * Auxiliary Problem
 *
 * Ax <= b.
 * We detect each constraint i with b[i] < 0.
 * For those cases we add a variable aux_i with initial value aux_i = -b[i].
 *
 * This gives us a new LP with a known basic feasible solution.
 * min sum(aux_i)
 * Ax - sum(e_i) <= b.
 *
 * if there is some feasible solution, we will find it and get sum(aux_i) = 0.
 */

struct result solver_wrapper(int m,
                             int n,
                             const vector<vector<double>>& A_in,
                             const vector<double>& b_in,
                             const vector<double>& c_in) {
    if (IS_CUOPT_BACKEND) {
        // The cuOpt backend is being used.
        // We skip the Phase I logic and directly call the solver.
        vector<double> x_dummy;
        vector<int> B_dummy;
        return solver(m, n, A_in, b_in, c_in, x_dummy, B_dummy);
    }

    // Step 1. Build & Solve Auxiliary Problem
    int n_aux = m + n; // New number of variables
    vector<vector<double>> A_aux = A_in;
    vector<double> c_aux(n_aux, 0.0);
    vector<double> x_aux(n_aux, 0.0);
    vector<int> B_aux(m);

    for (int i = 0; i < m; i++) {
        vector<double> e_i(m, 0);
        double multiplier = (b_in[i] > 0) ? 1.0 : -1.0;
        e_i[i] = multiplier;

        A_aux.push_back(e_i);
        c_aux[n + i] = 1.0;
        x_aux[n + i] = multiplier * b_in[i]; // To satisfy the constraints (a_i = b_i)
        B_aux[i] = n + i;
    }

    std::cout << "Solving auxiliary problem to find feasible solution x..." << std::endl;
    result res = solver(m, n_aux, A_aux, b_in, c_aux, x_aux, B_aux);
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

        result res2 = solver(m, n_aux, A_aux, b_in, c_bigM, x_phase2, B_phase2);

        // Trim result back to n variables
        if (res2.success) {
            res2.assignment.resize(n);
        }
        return res2;
    } else {
        // Standard Phase 2: No artificials in basis, use original A and c.
        x_phase2.resize(n);
        return solver(m, n, A_in, b_in, c_in, x_phase2, B_phase2);
    }
}