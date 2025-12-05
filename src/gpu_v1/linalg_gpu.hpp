#pragma once
#include <vector>
using std::vector;

typedef vector<double> vec;
typedef vector<vec> mat;
typedef vector<double> mat_cm;

// Workspace Management
void init_gpu_workspace(int n);
void destroy_gpu_workspace();

// Upload static problem data (A and c) to GPU once
void gpu_load_problem(int m, int n_total, const double* A_flat, const double* b, const double* c);

// Upload initial N indices to GPU
void gpu_init_non_basic(int n_count, const int* N_indices);

// Update a single index in N (used after pivot)
// offset: index in the N array (0 to n-1)
// new_val: the new variable index (ii)
void gpu_update_non_basic_index(int offset, int new_val);

// Update only the RHS storage (used during perturbation)
void gpu_update_rhs_storage(int m, const double* b_new);

// Runs the pricing kernel and returns the best candidate
struct PricingResult {
    int index_in_N; // The index inside N array (0 to n-1), corresponds to j_i
    double min_val;
};
PricingResult gpu_pricing_dantzig(int n_count);

// Compute reduced costs: sn = c - A^T * y
void gpu_compute_reduced_costs(int m, int n_total);

/**
 * 1. Uploads the basis indices (B) to the GPU.
 * 2. Gathers the corresponding columns from A_full_d into B_inv_d.
 * 3. Factorizes (LU) AND Computes the Explicit Inverse (B_inv_d becomes B^-1).
 */
void gpu_build_basis_and_invert(int m, int n_total, const int* B_indices);

/**
 * Updates the Inverse Basis Matrix (B_inv_d) in-place using the Sherman-Morrison formula.
 * pivot_row: The index of the variable leaving the basis (j).
 * Assumes the direction vector 'd' (aka B^-1 * A_entering) is already
 * residing in the GPU workspace 'ws.b_d' (leftover from the previous step).
 */
void gpu_update_basis_fast(int m, int pivot_row);

// Solves x = B^-1 * b. Result copied to Host (for ratio test).
void gpu_solve_primal(int m, const double* b_host, double* x_host);

// Solves y = B^-T * c_B. Result stays on Device in ws.y_d.
void gpu_solve_duals(int m, const double* c_B_host);

/**
 * Computes d = B^-1 * A_column[col_idx]
 * Uses the Explicit Inverse currently stored in AB_d.
 * The result is left in GPU memory (ws.b_d) for the subsequent update step,
 * and also copied to d_out (Host) for the ratio test.
 */
void gpu_calc_direction(int m, int col_idx, double* d_out);

// Computes x = B^-1 * b_persistent
void gpu_recalc_x_from_persistent_b(int m, double* x_out);