#include <algorithm>
#include <assert.h>
#include <cmath>
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

// Implementation following the Rice Notes:
// https://www.cmor-faculty.rice.edu/~yzhang/caam378/Notes/Save/note2.pdf
// It accept as input Ax = b, with initial solution x in column major format.
extern "C" struct result
solver(int m, int n_total, const mat_csc& A, const vec& b, const vec& c, vec& x, const idx& B_init) {
    PROFILE_SCOPE("Solver_Total");

    // ============================================================
    // 0. SANITY CHECKS
    // ============================================================
    assert(A.col_ptr.size() == n_total + 1);
    
    assert(b.size() == m);
    assert(c.size() == n_total);
    assert(x.size() == n_total);
    assert(n_total >= m);
    assert(B_init.size() == m);

    vector<vec> A_scatter;
    // scatter
    A_scatter.assign(n_total, std::vector<double>(m, 0.0));
    for (int j = 0; j < n_total; ++j) {
        for (int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k) {
            int i = A.row_idx[k];     // row
            double v = A.values[k];
            A_scatter[j][i] = v;          // column-major
        }
    }

    // ============================================================
    // 1. SCALING
    // ============================================================
    
    {
        PROFILE_SCOPE("GPU_Load");
        init_gpu_workspace(m);
        gpu_load_problem(m, n_total, A_scatter, b.data(), c.data(), x.data(), B_init.data());
    }

    {
        PROFILE_SCOPE("Scaling_Build");

        gpu_scale_problem(m, n_total);
    }

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
        destroy_gpu_workspace();
        return {.success = false};
    }


    {
        PROFILE_SCOPE("GPU_Load");
        gpu_init_non_basic(n, N.data());
    }

    // Perturbation
    bool is_perturbed = false;
    double prev_obj = std::numeric_limits<double>::infinity();
    int stall_counter = 0;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> dist_pert(1e-8, 1e-7);

    // ============================================================
    // 3. MAIN LOOP
    // ============================================================
    const bool DISABLE_REFACTOR = true;
    const int REFACTOR_FREQ = 50;
    int max_iter = 500 * (n_total + m);

    for (int iter = 0; iter < max_iter; iter++) {
        PROFILE_SCOPE("Solver_Iteration");

        // 1. & 2. Build and factorize Basis Matrix or update it
        gpu_set_c_B(m);

        bool refactor_needed = (iter % REFACTOR_FREQ == 0);
        if (iter == 0 || (refactor_needed && !DISABLE_REFACTOR)) {
            PROFILE_SCOPE("Refactor_Basis");
            // Full Rebuild: Gather -> LU -> Invert
            gpu_build_basis_and_invert(m, n_total, B.data());
        }

        // 3. Recompute Primal Solution
        {
            PROFILE_SCOPE("Recalc_X");
            gpu_recalc_x_from_persistent_b(m);
        }

        {
            PROFILE_SCOPE("Stall_Detection");
            double current_obj = gpu_get_obj(m, n_total);

            // We use the relative change! (NumCSE)
            if (std::abs(current_obj - prev_obj) < 1e-9 * (1.0 + std::abs(current_obj))) {
                stall_counter++;
            } else {
                stall_counter = 0;
            }
            prev_obj = current_obj;

            // Perturbation trigger
            if (stall_counter > 30 && !is_perturbed) {
                std::cout << "   >>> Stall detected (Iter " << iter << "). Perturbing..."
                          << std::endl;
                vec delta(m);
                for (int k = 0; k < m; ++k)
                    delta[k] = dist_pert(rng);
                is_perturbed = true;
                // Update the persistent b on GPU
                gpu_update_rhs_storage(m, delta.data());
                stall_counter = 0;
                // Re-solve with perturbed b immediately
                gpu_recalc_x_from_persistent_b(m);
            }

            if (iter % 500 == 0) {
                std::cout << "Iter " << iter << " | Obj: " << current_obj << std::endl;
            }
        }

        // 4. Solve Dual: y = B^-T * c_B
        {
            PROFILE_SCOPE("Solve_Dual");
            gpu_solve_duals(m);
        }

        // 5. Pricing (Dantzig's Rule for now)
        int j_i = -1;
        double min_sn = -1e-7;
        bool optimal = true;

        {
            PROFILE_SCOPE("Pricing");
            gpu_compute_reduced_costs(m, n_total);

            // Run reduction kernel
            PricingResult res = gpu_pricing_dantzig(n);

            if (res.index_in_N != -1) {
                optimal = false;
                j_i = res.index_in_N;
            }
        }

        if (optimal) {
            PROFILE_SCOPE("Final_Cleanup");
            std::cout << "Optimal basis found at iter " << iter << " | Scaled Obj: " << prev_obj
                      << std::endl;

            // 6. FINAL CLEANUP & UNSCALING
            // Solve against scaled b, then unscale x.
            if (is_perturbed) {
                gpu_reset_rhs_storage(m);
            }
            gpu_recalc_x_from_persistent_b(m);
            gpu_unscale(m, n_total, x.data());
            destroy_gpu_workspace();
            return {.success = true, .assignment = x, .basis = B, .basis_split_found = true};
        }

        // 7. Compute Primal Step: d = B^-1 * A_jj
        int jj = N[j_i]; // Entering Column Index
        {
            PROFILE_SCOPE("Calc_Direction");
            gpu_calc_direction(m, jj);
        }

        // 8. Unbounded Check & 9. Harris Ratio Test
        int r = -1;
        {
            PROFILE_SCOPE("Ratio_Test");
            r = gpu_run_ratio_test(m);
        }

        if (r < 0) {
            std::cout << "Solver failure: No leaving variable found." << std::endl;
            destroy_gpu_workspace();
            return {.success = false};
        }

        // 10. Update Basis
        {
            PROFILE_SCOPE("Update_Basis");
            int ii = B[r];
            N[j_i] = ii;
            B[r] = jj;
            gpu_update_non_basic_index(j_i, ii);
            gpu_update_basis_fast(m, r, jj);
        }
    }

    std::cout << "Out of iterations!" << std::endl;
    destroy_gpu_workspace();
    return {.success = false};
}
