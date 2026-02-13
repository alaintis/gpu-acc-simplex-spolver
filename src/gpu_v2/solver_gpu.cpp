#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "ProfileUtils.h"
#include "linalg_gpu.hpp"
#include "logging.hpp"
#include "solver.hpp"
#include "types.hpp"

using std::vector;

struct ScalingContext {
    vector<double> row_scale;
    vector<double> col_scale;
    vector<double> A_scaled;
    vector<double> b_scaled;
    vector<double> c_scaled;
    vector<double> x_scaled;
};

static ScalingContext build_scaling(int m,
                                    int n_total,
                                    const mat& A,
                                    const vec& b,
                                    const vec& c,
                                    const vec& x) {
    PROFILE_SCOPE("Scaling_Build");
    ScalingContext sc;
    sc.row_scale.assign(m, 0.0);
    sc.col_scale.assign(n_total, 0.0);

    // Row Scales: Max absolute value in each row
    for (int j = 0; j < n_total; j++) {
        for (int i = 0; i < m; i++) {
            sc.row_scale[i] = std::max(sc.row_scale[i], std::abs(A[j][i]));
        }
    }
    for (int i = 0; i < m; ++i)
        if (sc.row_scale[i] < 1e-9)
            sc.row_scale[i] = 1.0;

    // Col Scales: Max absolute value in each col (after row scaling)
    for (int j = 0; j < n_total; j++) {
        for (int i = 0; i < m; i++) {
            sc.col_scale[j] = std::max(sc.col_scale[j], std::abs(A[j][i]) / sc.row_scale[i]);
        }
    }
    for (int j = 0; j < n_total; ++j)
        if (sc.col_scale[j] < 1e-9)
            sc.col_scale[j] = 1.0;

    // A_scaled = A / (R * C)
    sc.A_scaled.assign(n_total * m, 0.0);
    for (int j = 0; j < n_total; ++j) {
        double c_s = sc.col_scale[j];
        for (int i = 0; i < m; ++i) {
            sc.A_scaled[j * m + i] = A[j][i] / (sc.row_scale[i] * c_s);
        }
    }

    // Scale b: b_new = b / R
    sc.b_scaled = b;
    for (int i = 0; i < m; ++i)
        sc.b_scaled[i] /= sc.row_scale[i];

    // Scale c: c_new = c / C
    sc.c_scaled = c;
    for (int j = 0; j < n_total; ++j)
        sc.c_scaled[j] /= sc.col_scale[j];

    // Scale initial x: x_new = x * C
    sc.x_scaled = x;
    for (int j = 0; j < n_total; ++j)
        sc.x_scaled[j] *= sc.col_scale[j];

    return sc;
}

static void unscale_solution(int m,
                             int n_total,
                             const idx& B,
                             const vector<double>& x_B,
                             const ScalingContext& sc,
                             vec& x_out) {
    PROFILE_SCOPE("Unscaling");

    // Build full x
    std::fill(x_out.begin(), x_out.end(), 0.0);
    for (int k = 0; k < m; ++k) {
        // x = x_new / C
        int col = B[k];
        x_out[col] = x_B[k] / sc.col_scale[col];
    }

    // Fix tiny negative entries to 0
    for (int j = 0; j < n_total; ++j) {
        if (x_out[j] > -1e-9 && x_out[j] < 0.0)
            x_out[j] = 0.0;
    }
}

// Implementation following the Rice Notes:
// https://www.cmor-faculty.rice.edu/~yzhang/caam378/Notes/Save/note2.pdf
// It accept as input Ax = b, with initial solution x in column major format.
extern "C" struct result solver(int m,
                                int n_total,
                                const mat_csc& A,
                                const vec& b,
                                const vec& c,
                                vec& x,
                                const idx& B_init) {
    PROFILE_SCOPE("Solver_Total");

    // ============================================================
    // 0. SANITY CHECKS
    // ============================================================
    assert(A.col_ptr.size() == n_total + 1);

    assert(b.size() == m);
    assert(c.size() == n_total);
    assert(B_init.size() == m);

    vector<vec> A_scatter;
    // scatter
    A_scatter.assign(n_total, std::vector<double>(m, 0.0));
    for (int j = 0; j < n_total; ++j) {
        for (int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k) {
            int i = A.row_idx[k]; // row
            double v = A.values[k];
            A_scatter[j][i] = v; // column-major
        }
    }

    // ============================================================
    // 1. SCALING
    // ============================================================
    ScalingContext sc = build_scaling(m, n_total, A_scatter, b, c, x);
    // ============================================================
    // 2. SOLVER INITIALIZATION
    // ============================================================
    int n = n_total - m; // Number of Non-Basic Variables
    idx B = B_init; // Basic Variables
    idx N(n); // Non-Basic Variables

    // Perform Basis Split
    std::vector<bool> is_basic(n_total, false);
    for (int i : B)
        is_basic[i] = true;
    int n_count = 0;
    for (int i = 0; i < n_total; i++) {
        if (!is_basic[i])
            N[n_count++] = i;
    }

    // Validate that the partition was successful.
    if (n_count != n) {
        std::cout << "Failed to partition Basis/Non-Basis." << std::endl;
        return {.success = false};
    }

    // Perturbation
    vector<double> b_eff = sc.b_scaled;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> dist_pert(1e-8, 1e-7);

    // Initial Perturbation
    for (int k = 0; k < m; ++k) {
        b_eff[k] += dist_pert(rng);
    }

    {
        PROFILE_SCOPE("GPU_Load");
        init_gpu_workspace(m);
        gpu_load_problem(m, n_total, sc.A_scaled.data(), b_eff.data(), sc.c_scaled.data());
        gpu_init_partition(m, n, B.data(), N.data());
    }
    cudaStream_t stream = gpu_get_stream();

    vector<double> x_B(m);
    // ============================================================
    // 3. MAIN LOOP
    // ============================================================
    const int BATCH_SIZE = 100;
    const bool DISABLE_REFACTOR = true;
    const int REFACTOR_FREQ = 50;
    int max_iter = 500 * (n_total + m);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    bool graph_created = false;

    for (int iter = 0; iter < max_iter; iter += BATCH_SIZE) {
        PROFILE_SCOPE("Solver_Iteration");

        // 1. & 2. Build and factorize Basis Matrix or update it
        bool refactor_needed = (iter % REFACTOR_FREQ == 0);
        if (iter == 0 || (refactor_needed && !DISABLE_REFACTOR)) {
            PROFILE_SCOPE("Refactor_Basis");
            // Full Rebuild: Gather -> LU -> Invert
            gpu_build_basis_and_invert(m);
        }

        if (!graph_created) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            // 3. Solve Dual: y = B^-T * c_B
            {
                PROFILE_SCOPE("Solve_Dual");
                gpu_solve_duals(m);
            }

            {
                PROFILE_SCOPE("Pricing");
                gpu_compute_reduced_costs(m, n_total);

                // Run reduction kernel & sets optimality flag
                gpu_pricing_dantzig(n);
            }

            // 6. Compute Primal Step: d = B^-1 * A_jj
            {
                PROFILE_SCOPE("Calc_Direction");
                gpu_calc_direction(m);
            }

            // 7. Unbounded Check & 9. Harris Ratio Test
            {
                PROFILE_SCOPE("Ratio_Test");
                gpu_run_ratio_test(m);
            }

            // 8. Update Basis
            {
                PROFILE_SCOPE("Update_Basis");
                gpu_update_all_state(m);
            }

            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graph_created = true;
        }

        // Launch Batch
        for (int k = 0; k < BATCH_SIZE; ++k) {
            cudaGraphLaunch(instance, stream);
        }

        int status = gpu_check_status();

        // Optimal
        if (status == 1) {
            PROFILE_SCOPE("Final_Cleanup");

            // 5. FINAL CLEANUP & UNSCALING
            gpu_download_basis(m, B.data());

            // Solve against scaled b, then unscale x.
            gpu_update_rhs_storage(m, sc.b_scaled.data());
            gpu_recalc_x_from_persistent_b(m, x_B.data());
            unscale_solution(m, n_total, B, x_B, sc, x);

            // Compute final objective value
            double obj = 0.0;
            for (int i = 0; i < n_total; ++i) {
                obj += x[i] * c[i];
            }

            std::cout << "Optimal basis found at iter " << iter << " | Obj: " << obj << std::endl;

            if (instance) {
                cudaGraphExecDestroy(instance);
            }
            if (graph) {
                cudaGraphDestroy(graph);
            }
            destroy_gpu_workspace();
            return {.success = true, .assignment = x, .basis = B, .basis_split_found = true};
        } else if (status == 2) {
            std::cout << "Unbounded or Error at approx iter " << iter << std::endl;
            break;
        }
    }

    std::cout << "Out of iterations!" << std::endl;
    if (instance) {
        cudaGraphExecDestroy(instance);
    }
    if (graph) {
        cudaGraphDestroy(graph);
    }
    destroy_gpu_workspace();
    return {.success = false};
}
