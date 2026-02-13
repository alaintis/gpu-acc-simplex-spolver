#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>


#include <thread>
#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <atomic>

#include "ProfileUtils.h"
#include "workspace.hpp"
#include "scale.hpp"
#include "linalg_gpu.hpp"
#include "logging.hpp"
#include "solver.hpp"
#include "types.hpp"

using std::vector;

#define WORKERS 4
static SharedWorkspace  ws;
static PrivateWorkspace pws[WORKERS];


static void gpu_load_original(
    SharedWorkspace &ws, cudaStream_t stream,
    const mat_csc &A, const double *b, const double *c,
    const double *x, const int *B
) {
    int m = ws.m;
    int n = ws.n;
    int nnz = A.col_ptr[n];

    cudaMemcpyAsync(ws.origin.A_col_ptr, A.col_ptr.data(), (n+1) * sizeof(int),    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.origin.A_row_idx, A.row_idx.data(), (nnz) * sizeof(int),    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.origin.A_values,  A.values.data(),  (nnz) * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ws.origin.c_d, c, n * sizeof(double), cudaMemcpyHostToDevice,     stream);
    cudaMemcpyAsync(ws.origin.b_d, b, m * sizeof(double), cudaMemcpyHostToDevice,     stream);
    cudaMemcpyAsync(ws.origin.x_d, x, n * sizeof(double), cudaMemcpyHostToDevice,     stream);
    cudaMemcpyAsync(ws.B_d, B, m * sizeof(int), cudaMemcpyHostToDevice, stream);

    gpu_scatter_A(ws, stream);
}

static void gpu_set_B(PrivateWorkspace& pws) {
    int m = pws.ws->m;
    cudaMemcpyAsync(pws.B_d, pws.ws->B_d, m * sizeof(int), cudaMemcpyDeviceToDevice, pws.stream);
}

void inner_iterations(PrivateWorkspace &pws, int inner_iter_count, int id) {
    pws.sp_u.count = 0;
    int end = pws.sol.iter + inner_iter_count;
    for(; pws.sol.iter < end; pws.sol.iter++) {
        gpu_set_c_B(pws);

        // 2. Apply Stall logic if necessary
        {
            PROFILE_SCOPE("Stall_Detection apply");
            // Perturbation trigger
            if (pws.sol.stall_counter >= 30 && !pws.sol.is_perturbed) {
                std::cout << "   >>> Stall detected (Iter " << pws.sol.iter << "). Perturbing..."
                        << std::endl;
                pws.sol.is_perturbed = true;
                pws.b_active = pws.ws->b_perturbe_d;
                pws.sol.stall_counter = 0;
            }
        }

        // 3. Recompute Primal Solution
        {
            PROFILE_SCOPE("Recalc_X");
            gpu_apply_B_inv(pws, CUBLAS_OP_N, pws.b_active, pws.x_B_d);
        }

        {
            PROFILE_SCOPE("Stall_Detection");
            double current_obj = gpu_get_obj(pws);

            // We use the relative change! (NumCSE)
            if (std::abs(current_obj - pws.sol.prev_obj) < 1e-9 * (1.0 + std::abs(current_obj))) {
                pws.sol.stall_counter++;
            } else {
                pws.sol.stall_counter = 0;
            }
            pws.sol.prev_obj = current_obj;
        }

        // 4. Solve Dual: y = B^-T * c_B
        {
            PROFILE_SCOPE("Solve_Dual");
            gpu_apply_B_inv(pws, CUBLAS_OP_T, pws.c_B_d, pws.y_d);
        }

        // 5. Pricing (Dantzig's Rule for now)
        int j_i = -1;
        double min_sn = -1e-7;
        bool optimal = true;

        {
            PROFILE_SCOPE("Pricing");
            gpu_compute_reduced_costs(pws);

            PricingResult res;
            // Run reduction kernel
            if(id == 0) {
                res = gpu_pricing_dantzig(pws);
            }
            else {
                res = gpu_pricing_random(pws, id ^ pws.sol.iter);
            }

            if (res.index_in_N != -1) {
                optimal = false;
                j_i = res.index_in_N;
            }
        }

        if (optimal) {
            pws.sol.state = ExitState::Optimal;
            return;
        }

        // 7. Compute Primal Step: d = B^-1 * A_jj
        int jj = pws.sol.N_h[j_i]; // Entering Column Index
        {
            PROFILE_SCOPE("Calc_Direction");
            double* A_col_ptr = pws.ws->scale.A_d + (static_cast<size_t>(jj) * pws.ws->m);
        
            // This computes d = B^-1 * A_j
            gpu_apply_B_inv(pws, CUBLAS_OP_N, A_col_ptr, pws.d_d);
        }

        // 8. Unbounded Check & 9. Harris Ratio Test
        int r = -1;
        {
            PROFILE_SCOPE("Ratio_Test");
            r = gpu_run_ratio_test(pws);
        }

        if (r < 0) {
            std::cout << "Solver failure: No leaving variable found." << std::endl;
            pws.sol.state = ExitState::Unbounded;
            return;
        }

        // 10. Update Basis
        {
            PROFILE_SCOPE("Update_Basis");
            int ii = pws.sol.B_h[r];

            pws.sol.N_h[j_i] = ii;
            pws.sol.B_h[r] = jj;
            gpu_update_non_basic_index(pws, j_i, ii);
            gpu_update_basis_fast(pws, r, jj);
        }
    }
    pws.sol.state = ExitState::None;
    cudaStreamSynchronize(pws.stream);
}




struct Sync {
    std::mutex m;
    std::condition_variable cv;
    std::condition_variable cv_controller;
    int phase = -1; // shared generation
    int remaining_workers = 0;
};

void worker(Sync& sync, int id, std::atomic_bool& stop_flag, int inner_iter_count) {
    int my_phase = -1;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        // Wait for next phase
        std::unique_lock<std::mutex> lock(sync.m);
        sync.cv.wait(lock, [&] {
            return sync.phase != my_phase;
        });

        my_phase = sync.phase;
        lock.unlock();

        inner_iterations(pws[id], inner_iter_count, id);

        {
            std::lock_guard<std::mutex> lock(sync.m);
            sync.remaining_workers--;
            if (sync.remaining_workers == 0) {
                sync.cv_controller.notify_one(); // all workers done
            }
        }
    }
}



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

    // ============================================================
    // 1. SCALING
    // ============================================================

    {
        PROFILE_SCOPE("GPU_Load");
        gpu_shared_workspace_init(ws, m, n_total, A.col_ptr[n_total]);
        for(int i = 0; i < WORKERS; i++) {
            gpu_private_workspace_init(pws[i], ws, B, N, 20);
        }
        gpu_load_original(ws, pws[0].stream, A, b.data(), c.data(), x.data(), B_init.data());
    }

    {
        PROFILE_SCOPE("Scaling_Build");
        gpu_scale_problem(ws, pws[0].stream);
        cudaStreamSynchronize(pws[0].stream);
    }

    // ============================================================
    // 2. SOLVER INITIALIZATION
    // ============================================================
    {
        PROFILE_SCOPE("GPU_Load");
        for(int i = 0; i < WORKERS; i++) {
            gpu_set_B(pws[i]);
            gpu_init_non_basic(pws[i], N.data());
            cudaStreamSynchronize(pws[i].stream);
        }
    }

    // Perturbation
    bool is_perturbed = false;
    double prev_obj = std::numeric_limits<double>::infinity();
    int stall_counter = 0;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> dist_pert(1e-8, 1e-7);

    // Create perturbed b in advance.
    vec delta(m);
    for (int k = 0; k < m; ++k)
        delta[k] = dist_pert(rng);
    // Update the persistent b on GPU
    gpu_vec_add(pws[0], m, delta.data(), ws.scale.b_d, ws.b_perturbe_d);
    cudaStreamSynchronize(pws[0].stream);


    // ============================================================
    // 3. MAIN LOOP
    // ============================================================
    const bool DISABLE_REFACTOR = true;
    const int REFACTOR_FREQ = 100;
    int max_iter = 500 * (n_total + m);
    const int inner_iter_count = pws[0].sp_u.capacity;

    // Create multiple threads.
    Sync sync;
    const int thread_count = WORKERS-1;
    std::atomic<bool> stop_flag{false};


    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    // Launch worker threads
    for (int i = 0; i < thread_count; ++i) {
        threads.emplace_back(worker, std::ref(sync), i+1, std::ref(stop_flag), inner_iter_count);
    }

    for (int iter = 0; iter < max_iter; iter+=inner_iter_count) {
        PROFILE_SCOPE("Solver_Iteration");

        if (iter % 500 == 0) {
            std::cout << "Iter " << iter << " | Obj: " << pws[0].sol.prev_obj << std::endl;
        }

        // 1. Build and factorize Basis Matrix or update it
        bool refactor_needed = (iter % REFACTOR_FREQ == 0);
        if (iter == 0 || (refactor_needed && !DISABLE_REFACTOR)) {
            PROFILE_SCOPE("Refactor_Basis");
            // Full Rebuild: Gather -> LU -> Invert
            gpu_build_basis_and_invert(ws, pws[0]);
        } else if (pws[0].sp_u.count > 0) {
            gpu_apply_sparse_basis(ws, pws[0]);
        }

        cudaStreamSynchronize(pws[0].stream);

        {
            std::lock_guard<std::mutex> lock(sync.m);
            sync.remaining_workers = thread_count;
            sync.phase = iter;

        }
        sync.cv.notify_all();

        inner_iterations(pws[0], inner_iter_count, 0);
        
        {
            // Wait for next phase
            std::unique_lock<std::mutex> lock(sync.m);
            sync.cv_controller.wait(lock, [&] {
                return sync.remaining_workers == 0;
            });
        }

        
        std::cout << "Iter " << iter << " | Obj: ";
        for(int i = 0; i < WORKERS; i++) {
            std::cout << i << ": " << pws[i].sol.prev_obj << ", ";
        }
        std::cout << std::endl;

        if(pws[0].sol.state != ExitState::None) break;
    }

    {
        PROFILE_SCOPE("Final Cleanup");

        std::cout << "Final Cleanup" << std::endl;

        // Stop all threads
        stop_flag.store(true, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(sync.m);
            sync.phase = max_iter;
        }
        sync.cv.notify_all();

        for (auto& t : threads) {
            t.join();
        }


        result res;
        if (pws[0].sol.state == ExitState::None) {
            std::cout << "Out of iterations!" << std::endl;
            res.success = false;
        } else if (pws[0].sol.state == ExitState::Unbounded) {
            std::cout << "Unbounded!" << std::endl;
            res.success = false;
        } else if (pws[0].sol.state == ExitState::Optimal) {
            std::cout << "Optimal basis found at iter " << pws[0].sol.iter << " | Scaled Obj: " << pws[0].sol.prev_obj
                        << std::endl;
            gpu_apply_B_inv(pws[0], CUBLAS_OP_N, ws.scale.b_d, pws[0].x_B_d);
            gpu_unscale(ws, pws[0], x.data());

            res.success = true;
            res.assignment = x;
            res.basis = pws[0].sol.B_h;
            res.basis_split_found = true;
        }
    
        gpu_shared_workspace_destroy(ws);
        for(int i = 0; i < WORKERS; i++) {
            gpu_private_workspace_destroy(pws[i]);
        }
        return res;
    }
}
