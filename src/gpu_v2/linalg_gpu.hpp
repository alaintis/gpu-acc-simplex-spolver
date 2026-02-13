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

// Uploads both B (Basis) and N (Non-Basic) indices once
void gpu_init_partition(int m, int n_count, const int* B_indices, const int* N_indices);

// Download current B array from GPU (Called only at optimality)
void gpu_download_basis(int m, int* B_out);

// Update only the RHS storage (used during perturbation)
void gpu_update_rhs_storage(int m, const double* b_new);

/**
 * Runs the pricing kernel, unpacks the result on the GPU, and sets the Optimality Flag.
 * Returns: true if optimal, false otherwise.
 */
void gpu_pricing_dantzig(int n_count);

// Compute reduced costs: sn = c - A^T * y
void gpu_compute_reduced_costs(int m, int n_total);

/**
 * Uses the GPU-resident B array to gather columns, factorize, and invert.
 */
void gpu_build_basis_and_invert(int m);

/**
 * Performs the complete Basis, Solution, and State update step .
 * 1. Snapshot: Copies the pivot row to a temp buffer to prevent Race Conditions & Cache Thrashing
 * (Read-After-Write).
 * 2. Matrix Update: Updates the Inverse Basis (B_inv) using the safe snapshot (Sherman-Morrison).
 * 3. State Update: Updates the solution vector x (incrementally), swaps Basis/Non-Basis indices,
 * and updates the Basis costs (c_B).
 */
void gpu_update_all_state(int m);

// Solves y = B^-T * c_B. Result stays on Device in ws.y_d.
void gpu_solve_duals(int m);

/**
 * Extracts the column indicated by 'ws.entering_col_d' into a temp buffer
 * and computes d = B^-1 * A_entering
 */
void gpu_calc_direction(int m);

// Computes x = B^-1 * b_persistent
void gpu_recalc_x_from_persistent_b(int m, double* x_out);

// Runs the ratio test and stores the leaving index on Device
void gpu_run_ratio_test(int m);

// Returns 0 (Running), 1 (Optimal), or 2 (Error)
int gpu_check_status();

// Returns the stream used by the solver (for Graph Capture)
cudaStream_t gpu_get_stream();