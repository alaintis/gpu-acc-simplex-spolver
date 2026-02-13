#include <assert.h>
#include <cmath>
#include <cuopt/linear_programming/cuopt_c.h>
#include <stdio.h>

#include "base_solver.hpp"
#include "solver.hpp"

typedef vector<cuopt_int_t> idx;
typedef vector<cuopt_float_t> vec;

/**
 * This is the weak-symbol implementation of solver().
 * When the cuopt backend is selected, this function is linked.
 * It just passes the call through to base_solver().
 *
 * NOTE: The 'x' vector is non-const (vector<double>&) to match the
 * cpu/gpu solvers, but base_solver will treat it as const.
 */
extern "C" __attribute__((weak)) result solver(int m,
                                               int n,
                                               const mat_csc& A,
                                               const vector<double>& b,
                                               const vector<double>& c,
                                               vector<double>& x,
                                               const idx& B) {
    return base_solver(m, n, A, b, c);
}

/**
 * This is the actual cuOpt implementation, called by the weak solver()
 * and by the test_runner for the reference solution.
 */
result base_solver(int m,
                   int n,
                   const mat_csc& A,
                   const vector<double>& b,
                   const vector<double>& c) {
    assert(A.col_ptr.size() == n + 1);
    
    assert(b.size() == m);
    assert(c.size() == n);
    assert(n >= m);

    vector<vec> A_scatter;
    // scatter
    A_scatter.assign(n, std::vector<double>(m, 0.0));
    for (int j = 0; j < n; ++j) {
        for (int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k) {
            int i = A.row_idx[k];     // row
            double v = A.values[k];
            A_scatter[j][i] = v;          // column-major
        }
    }

    cuOptOptimizationProblem problem = NULL;
    cuOptSolverSettings settings = NULL;
    cuOptSolution solution = NULL;

    cuopt_int_t num_variables = n;
    cuopt_int_t num_constraints = m;

    // Build a sparse CSR matrix
    idx row_offsets;
    idx column_indices;
    vec values;

    row_offsets.push_back(0); // First row always starts at 0
    cuopt_int_t nnz = 0;

    // We iterate row-by-row and only store non-zeroes
    for (int i = 0; i < m; i++) { // For each row i
        for (int j = 0; j < n; j++) { // For each column j
            double val = A_scatter[j][i]; // Get A(i, j) from column-major input

            // Only store non-zero values
            if (std::abs(val) > 1e-12) {
                values.push_back(val);
                column_indices.push_back(j);
                nnz++;
            }
        }
        row_offsets.push_back(nnz);
    }

    vec var_lower_bounds(n, 0);
    vec var_upper_bounds(n, CUOPT_INFINITY);
    vector<char> var_types(n, CUOPT_CONTINUOUS);
    vector<char> constraint_types(m, CUOPT_EQUAL);

    cuopt_int_t status;
    cuopt_float_t time;
    cuopt_int_t termination_status = -1; // Initialize to a known error state
    cuopt_float_t objective_value;

    // Result
    vec solution_values(n);

    // Create the problem
    status = cuOptCreateProblem(m, n, CUOPT_MINIMIZE,
                                0.0, // objective offset
                                c.data(), row_offsets.data(), column_indices.data(), values.data(),
                                constraint_types.data(), b.data(), var_lower_bounds.data(),
                                var_upper_bounds.data(), var_types.data(), &problem);
    if (status != CUOPT_SUCCESS) {
        printf("Error creating problem: %d\n", status);
        goto DONE;
    }

    // Create solver settings
    status = cuOptCreateSolverSettings(&settings);
    if (status != CUOPT_SUCCESS) {
        printf("Error creating solver settings: %d\n", status);
        goto DONE;
    }

    // Set solver parameters
    status = cuOptSetIntegerParameter(settings, CUOPT_LOG_TO_CONSOLE, false);
    if (status != CUOPT_SUCCESS) {
        printf("Error setting log state: %d\n", status);
        goto DONE;
    }

    status = cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 1.0e-4);
    if (status != CUOPT_SUCCESS) {
        printf("Error setting primal tolerance: %d\n", status);
        goto DONE;
    }

    status = cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_DUAL_TOLERANCE, 1.0e-4);
    if (status != CUOPT_SUCCESS) {
        printf("Error setting dual tolerance: %d\n", status);
        goto DONE;
    }

    status = cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_GAP_TOLERANCE, 1.0e-4);
    if (status != CUOPT_SUCCESS) {
        printf("Error setting gap tolerance: %d\n", status);
        goto DONE;
    }

    // Force deterministic behavior by using only one thread.
    status = cuOptSetIntegerParameter(settings, CUOPT_NUM_CPU_THREADS, 1);
    if (status != CUOPT_SUCCESS) {
        printf("Error setting num cpu threads: %d\n", status);
        goto DONE;
    }

    // Force the solver to only use the CPU-based Dual Simplex method.
    // This, combined with num_cpu_threads=1, eliminates all
    // non-determinism from the default "concurrent" method.
    status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, 2); // 2 = DUAL_SIMPLEX
    if (status != CUOPT_SUCCESS) {
        printf("Error setting solver method: %d\n", status);
        goto DONE;
    }

    // Solve the problem
    status = cuOptSolve(problem, settings, &solution);
    if (status != CUOPT_SUCCESS) {
        printf("Error solving problem: %d\n", status);
        goto DONE;
    }

    status = cuOptGetTerminationStatus(solution, &termination_status);
    if (status != CUOPT_SUCCESS) {
        printf("Error getting termination status: %d\n", status);
        goto DONE;
    }

    status = cuOptGetPrimalSolution(solution, solution_values.data());

DONE:
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);

    if (termination_status == CUOPT_TERIMINATION_STATUS_OPTIMAL) {
        result res = {.success = true, .assignment = solution_values};
        return res;
    } else {
        result res;
        res.success = false;
        return res;
    }
}
