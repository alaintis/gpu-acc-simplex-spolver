#include <assert.h>
#include <stdio.h>
#include <cuopt/linear_programming/cuopt_c.h>

#include "solver.hpp"
#include "base_solver.hpp"

typedef vector<cuopt_int_t> idx;
typedef vector<cuopt_float_t> vec;

extern "C" __attribute__((weak))
result solver(int m, int n, vector<vector<double>> A, vector<double> b, vector<double> c, vector<double> x) {
    return base_solver(m, n, A, b, c);
}

result base_solver(int m, int n, vector<vector<double>> A, vector<double> b, vector<double> c) {
    assert(A.size() == n);
    for(int i = 0; i < n; i++) {
        assert(A[i].size() == m);
    }
    assert(b.size() == m);
    assert(c.size() == n);

    cuOptOptimizationProblem problem = NULL;
    cuOptSolverSettings settings = NULL;
    cuOptSolution solution = NULL;

    cuopt_int_t num_variables = n;
    cuopt_int_t num_constraints = m;
    cuopt_int_t nnz = m * n;

    // CSR
    idx row_offsets(m + 1);
    idx column_indices(m * n);
    vec values(m * n);
    for(int i = 0; i < m+1; i++) {
        row_offsets[i] = i * n;
    }
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            column_indices[i*n+j] = j;
            values[i*n + j] = A[j][i];
        }
    }

    vec var_lower_bounds(n, 0);
    vec var_upper_bounds(n, CUOPT_INFINITY);
    vector<char> var_types(n, CUOPT_CONTINUOUS);
    vector<char> constraint_types(m, CUOPT_LESS_THAN);
    
    cuopt_int_t status;
    cuopt_float_t time;
    cuopt_int_t termination_status;
    cuopt_float_t objective_value;

    // Result
    vec solution_values(n);
    
    // Create the problem
    status = cuOptCreateProblem(m,
                                n,
                                CUOPT_MINIMIZE,
                                0.0,    // objective offset
                                c.data(),
                                row_offsets.data(),
                                column_indices.data(),
                                values.data(),
                                constraint_types.data(),
                                b.data(),
                                var_lower_bounds.data(),
                                var_upper_bounds.data(),
                                var_types.data(),
                                &problem);
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
    // Silence solver
    // This does not really work, still an open issue: https://github.com/NVIDIA/cuopt/issues/187
    status = cuOptSetIntegerParameter(settings, CUOPT_LOG_TO_CONSOLE, false);
    if (status != CUOPT_SUCCESS) {
        printf("Error setting log state: %d\n", status);
        goto DONE;
    }

    // Set tolerance
    status = cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 0.0001);
    if (status != CUOPT_SUCCESS) {
        printf("Error setting optimality tolerance: %d\n", status);
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
    if (status != CUOPT_SUCCESS) {
        printf("Error getting solution values: %d\n", status);
        goto DONE;
    }
DONE:
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);

    if (termination_status == CUOPT_TERIMINATION_STATUS_OPTIMAL) {
        result res = {
            .success = true,
            .assignment = solution_values
        };
        return res;
    } else {
        result res;
        res.success = false;
        return res;
    }
}
