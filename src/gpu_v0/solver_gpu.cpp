#include <algorithm>
#include <assert.h>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "linalg_gpu.hpp"
#include "logging.hpp"
#include "solver.hpp"
#include "types.hpp"

#include "base_solver_wrapped.hpp"

static bool contains(const idx& B, int i) { return std::find(B.begin(), B.end(), i) != B.end(); }

struct ScalingContext {
    vector<double> row_scale;
    vector<double> col_scale;

    vector<double> A_scaled; // flattened Matrix n_total * m
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

extern "C" struct result
base_solver(int m, int n_total, const mat& A, const vec& b, const vec& c, vec& x, const idx& B_init) {
    // ============================================================
    // 0. SANITY CHECKS
    // ============================================================
    assert(A.size() == n_total);
    for (int i = 0; i < n_total; i++) {
        assert(A[i].size() == m);
    }
    assert(b.size() == m);
    assert(c.size() == n_total);
    assert(x.size() == n_total);
    assert(n_total >= m);
    assert(B_init.size() == m);

    // ============================================================
    // 1. SCALING
    // ============================================================
    ScalingContext sc = build_scaling(m, n_total, A, b, c, x);

    auto& row_scale = sc.row_scale;
    auto& col_scale = sc.col_scale;
    auto& A_scaled = sc.A_scaled;
    auto& b_scaled = sc.b_scaled;
    auto& c_scaled = sc.c_scaled;
    auto& x_scaled = sc.x_scaled;

    // ============================================================
    // 2. SOLVER INITIALIZATION
    // ============================================================
    int n = n_total - m; // Number of Non-Basic Variables
    idx B = B_init; // Basic Variables
    idx N(n); // Non-Basic Variables

    // Perform Basis Split
    int n_count = 0;
    for (int i = 0; i < n_total; i++) {
        if (!contains(B, i))
            N[n_count++] = i;
    }

    // Validate that the partition was successful.
    if (n_count != n) {
        std::cout << "Failed to partition Basis/Non-Basis." << std::endl;
        return {.success = false};
    }

    init_gpu_workspace_v0(m);
    gpu_load_problem_v0(m, n_total, A_scaled.data(), b_scaled.data(), c_scaled.data());

    // Buffers
    vector<double> A_B(m * m);
    vector<double> c_B(m);
    vector<double> x_B(m);
    vector<double> y(m);
    vector<double> d(m);

    // Perturbation
    vector<double> b_eff = b_scaled;
    bool is_perturbed = false;
    double prev_obj = std::numeric_limits<double>::infinity();
    int stall_counter = 0;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> dist_pert(1e-8, 1e-7);

    // ============================================================
    // 3. MAIN LOOP
    // ============================================================
    int max_iter = 500 * (n_total + m);

    for (int iter = 0; iter < max_iter; iter++) {
        // 1. & 2. Build and factorize Basis Matrix
        for (int i = 0; i < m; i++) {
            c_B[i] = c_scaled[B[i]];
        }

        gpu_build_basis_and_factorize_v0(m, n_total, B.data());

        // 3. Recompute Primal Solution instead of incremental updates (Anti-Drift) & do stall
        // detection. We solve A_B * x_B = b_scaled explicitly every iteration.
        gpu_solve_from_persistent_b_v0(m, x_B.data());

        // Calculate whole x_scaled for stall detection
        std::fill(x_scaled.begin(), x_scaled.end(), 0.0);
        for (int i = 0; i < m; ++i) {
            x_scaled[B[i]] = x_B[i];
        }

        // Stall Detection
        double current_obj = 0.0;
        for (int i = 0; i < n_total; ++i) {
            current_obj += x_scaled[i] * c_scaled[i];
        }

        // We use the relative change! (NumCSE)
        if (std::abs(current_obj - prev_obj) < 1e-9 * (1.0 + std::abs(current_obj))) {
            stall_counter++;
        } else {
            stall_counter = 0;
        }
        prev_obj = current_obj;

        // Perturbation trigger
        // We only have to do this once per run!
        if (stall_counter > 30 && !is_perturbed) {
            std::cout << "   >>> Stall detected (Iter " << iter << "). Perturbing..." << std::endl;
            for (int k = 0; k < m; ++k)
                b_eff[k] += dist_pert(rng);
            is_perturbed = true;
            // Update the persistent b on GPU
            gpu_update_rhs_storage_v0(m, b_eff.data());
            stall_counter = 0;
            // Re-solve with perturbed b immediately
            gpu_solve_from_persistent_b_v0(m, x_B.data());
            for (int i = 0; i < m; ++i) {
                x_scaled[B[i]] = x_B[i];
            }
        }

        if (iter % 500 == 0) {
            std::cout << "Iter " << iter << " | Obj: " << current_obj << std::endl;
        }

        // 4. Solve Dual: A_B^T y = c_B
        gpu_solve_prefactored_v0(m, c_B.data(), y.data(), true);

        // 5. Pricing (Dantzig's Rule for now)
        // We calculate reduced costs for ALL variables (Basic and Non-Basic) at once on GPU.
        // It's faster to do one giant matrix mult than iterating just N on CPU.
        const double* all_sn = gpu_compute_reduced_costs_v0(m, n_total, y.data());

        int j_i = -1;
        double min_sn = -1e-7;
        bool optimal = true;

        // Iterate over Non-Basic variables to find the entering variable
        for (int i = 0; i < n; i++) {
            int original_idx = N[i];

            // Read pre-calculated value from the host buffer returned by GPU
            double sn = all_sn[original_idx];

            if (sn < min_sn) {
                optimal = false;
                min_sn = sn;
                j_i = i;
            }
        }

        if (optimal) {
            std::cout << "Optimal basis found at iter " << iter << " | Scaled Obj: " << current_obj
                      << std::endl;

            // 6. FINAL CLEANUP & UNSCALING
            // Solve against scaled b, then unscale x.
            if (is_perturbed) {
                gpu_update_rhs_storage_v0(m, b_scaled.data());
            }

            gpu_solve_from_persistent_b_v0(m, x_B.data());

            unscale_solution(m, n_total, B, x_B, sc, x);

            destroy_gpu_workspace_v0();
            return {.success = true, .assignment = x, .basis = B, .basis_split_found = true};
        }

        // 7. Compute Primal Step: A_B d = A_jj
        int jj = N[j_i];
        gpu_solve_from_resident_col_v0(m, jj, d.data());

        // 8. Unbounded Check
        bool unbounded = true;
        for (int i = 0; i < m; i++) {
            if (d[i] > 1e-9) {
                unbounded = false;
                break;
            }
        }
        if (unbounded) {
            std::cout << "Problem is unbounded." << std::endl;
            destroy_gpu_workspace_v0();
            return {.success = false};
        }

        // 9. Harris Ratio Test
        double harris_tol = 1e-7;
        double best_theta = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; i++) {
            if (d[i] > 1e-9) {
                double theta = (x_B[i] + harris_tol) / d[i];
                if (theta < best_theta)
                    best_theta = theta;
            }
        }

        int r = -1;
        double max_pivot = -1.0;

        for (int i = 0; i < m; i++) {
            if (d[i] > 1e-9) {
                double theta = x_B[i] / d[i];
                if (theta <= best_theta) {
                    if (d[i] > max_pivot) {
                        max_pivot = d[i];
                        r = i;
                    }
                }
            }
        }

        if (r < 0) {
            std::cout << "Solver failure: No leaving variable found." << std::endl;
            destroy_gpu_workspace_v0();
            return {.success = false};
        }

        // 10. Update Basis
        int ii = B[r];
        N[j_i] = ii;
        B[r] = jj;
    }
    std::cout << "Out of iterations!" << std::endl;
    destroy_gpu_workspace_v0();
    return {.success = false};
}


extern "C" __attribute__((weak)) struct result
solver(int m, int n_total, const mat& A, const vec& b, const vec& c, vec& x, const idx& B_init) {
    return base_solver(m, n_total, A, b, c, x, B_init);
}
