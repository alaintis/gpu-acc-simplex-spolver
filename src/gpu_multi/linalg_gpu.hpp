#pragma once
#include <vector>
using std::vector;

#include "workspace.hpp"

typedef vector<double> vec;
typedef vector<vec> mat;
typedef vector<double> mat_cm;


void gpu_build_basis_and_invert(SharedWorkspace &ws, const PrivateWorkspace &pws);

// Collect Sherman-Morrison logic.
void gpu_update_basis_fast(PrivateWorkspace &pws, int pivot_row, int new_row);

// Uses the Sherman-Morrison kernel to update (A)B_inv_d in-place.
// This relies on 'ws.b_d' containing the direction vector 'd'!
void gpu_apply_sparse_basis(SharedWorkspace &ws, PrivateWorkspace &pws);


// Apply the sparse updated B_inv matrix.
// Calculates x = B^-1 * y, consumes y in the process.
void gpu_apply_B_inv(const PrivateWorkspace &pws, cublasOperation_t op, const double *y_d, double *x_d);


// This computes sn = c - A^T * y
// (we do this for the whole A, I think it's faster than building the non-basic Matrix)
void gpu_compute_reduced_costs(const PrivateWorkspace &pws);

// Helper: Update the stored RHS b (used when applying perturbation)
void gpu_init_non_basic(const PrivateWorkspace &pws, const int* N_indices);
void gpu_update_non_basic_index(PrivateWorkspace &pws, int offset, int new_val);


// Runs the pricing kernel and returns the best candidate
struct PricingResult {
    int index_in_N; // The index inside N array (0 to n-1), corresponds to j_i
    double min_val;
};
PricingResult gpu_pricing_dantzig(PrivateWorkspace &pws);
PricingResult gpu_pricing_random(PrivateWorkspace &pws, int seed);
int gpu_run_ratio_test(const PrivateWorkspace &pws);

void gpu_vec_add(const PrivateWorkspace &pws, int n, const double *a, const double *b, double *c);

double gpu_get_obj(const PrivateWorkspace &pws);
void gpu_set_c_B(const PrivateWorkspace &pws);

void gpu_scatter_A(SharedWorkspace &ws, cudaStream_t stream);
