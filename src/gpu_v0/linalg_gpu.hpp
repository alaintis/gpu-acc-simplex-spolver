#pragma once
#include <vector>
using std::vector;

typedef vector<double> vec;
typedef vector<vec> mat;
typedef vector<double> mat_cm;

// Workspace Management
void init_gpu_workspace_v0(int n);
void destroy_gpu_workspace_v0();

/**
 * Solve linear system using the PREVIOUSLY computed factorization.
 * If transpose is true: Solves A^T x = b
 * If transpose is false: Solves A x = b
 * b: pointer to n vector (input)
 * x: pointer to n vector (output)
 */
void gpu_solve_prefactored(int n, const double* b, double* x, bool transpose);

// Upload static problem data (A and c) to GPU once
void gpu_load_problem_v0(int m, int n_total, const double* A_flat, const double* b, const double* c);

// Compute reduced costs: sn = c - A^T * y
// Returns a pointer to host memory containing all reduced costs
const double* gpu_compute_reduced_costs(int m, int n_total, const double* y_host);

/**
 * Solves A_Basis * d = A_column[col_idx]
 * Uses the A matrix already stored on the GPU from gpu_load_problem_v0.
 * col_idx: The index of the entering variable (column to fetch)
 * d_out:   Host pointer to store the result
 */
void gpu_solve_from_resident_col(int m, int col_idx, double* d_out);

/**
 * 1. Uploads the basis indices (B) to the GPU.
 * 2. Gathers the corresponding columns from A_full_d into AB_d.
 * 3. Factorizes AB_d inplace (LU decomposition).
 */
void gpu_build_basis_and_factorize(int m, int n_total, const int* B_indices);

// Update only the RHS storage (used during perturbation)
void gpu_update_rhs_storage_v0(int m, const double* b_new);

// Uses the b stored on the GPU. Result is copied to x_out (Host).
void gpu_solve_from_persistent_b(int m, double* x_out);
