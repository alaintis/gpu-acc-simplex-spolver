#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../gpu_v0/base_solver_wrapped.hpp" // cuOpt reference solver
#include "linprog.hpp" // score
#include "logging.hpp"
#include "problem_reader.hpp"
#include "solver_wrapper.hpp" // backend (cpu/gpu/cuopt)

namespace fs = std::filesystem;

// Timer utility for benchmarking
struct Timer {
    std::chrono::high_resolution_clock::time_point time_point;

    // Returns duration in milliseconds
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - time_point;
        return ms.count();
    }
    void start() {
        auto t = std::chrono::high_resolution_clock::now();
        time_point = t;
    }
};

/**
 * Runs the solver for a single problem and prints the result.
 * Returns TRUE if the test passed (solvers agree),
 * Returns FALSE if the test failed (solvers disagree).
 */
bool run_solver_test(const Problem& p, const std::string& problem_name) {
    // ... (This function remains unchanged)
    std::cout << "Comparing on: " << problem_name << " (m=" << p.m << ", n=" << p.n << ")"
              << std::endl;

    result r_backend; // Result from backend (cpu/gpu)
    result r_cuopt; // Result from cuOpt (base)
    Timer timer_backend;
    Timer timer_cuopt;
    double time_backend = 0.0;
    double time_cuopt = 0.0;

    // Run selected backend solver (via wrapper)
    try {
        timer_backend.start();

        r_backend = solver_wrapper(p.m, p.n, p.A, p.b, p.c);

        time_backend = timer_backend.stop();
        std::cout << "  Backend: " << (r_backend.success ? "Success" : "Failed")
                  << " (Time: " << time_backend << " ms)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  Backend threw exception: " << e.what() << std::endl;
        r_backend.success = false;
    }

    // Run the cuOpt base solver (reference)
    try {
        timer_cuopt.start();

        r_cuopt = base_solver_wrapped(p.m, p.n, p.A, p.b, p.c);

        time_cuopt = timer_cuopt.stop();
        std::cout << "  cuOpt:   " << (r_cuopt.success ? "Success" : "Failed")
                  << " (Time: " << time_cuopt << " ms)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  cuOpt threw exception: " << e.what() << std::endl;
        r_cuopt.success = false;
    }

    // Compare results
    if (r_backend.success && r_cuopt.success) {
        // Both successful, compare objective scores
        double score_backend = score(p.n, r_backend.assignment, p.c);
        double score_cuopt = score(p.n, r_cuopt.assignment, p.c);
        double delta = std::fabs(score_backend - score_cuopt);

        std::cout << "Printing the following for statistics collection ( time_all_problems.sh)"
                  << std::endl;
        std::cout << "optimal value BACKEND, optimal value CUOPT, delta" << std::endl;
        std::cout << "time BACKEND (ms), time GPUv0 (ms)" << std::endl;

        std::cout << "START" << std::endl;
        // std::cout << std::fixed << std::setprecision(6);
        // std::cout << "  Scores -> Backend: " << score_backend << " | cuOpt: " << score_cuopt
        //           << " | Delta: " << delta << std::endl;
        // std::cout << "  Times  -> Backend: " << time_backend << " ms | cuOpt: " << time_cuopt <<
        // " ms" << std::endl;

        // for statistics only data is outputed
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "" << score_backend << "," << score_cuopt << "," << delta << std::endl;
        std::cout << "" << time_backend << "," << time_cuopt << std::endl;
        std::cout << "END" << std::endl;

        if (delta > 0.001) {
            std::cout << "  !! WARNING: Objective values differ significantly!" << std::endl;
            std::cout << std::endl;
            return false;
        }
    } else if (r_backend.success != r_cuopt.success) {
        // One succeeded and one failed
        std::cout << "  !! WARNING: Solvers disagree on success!" << std::endl;
        std::cout << std::endl;
        return false;
    } else {
        std::cout << "  (Both solvers failed or were infeasible)" << std::endl;
    }

    std::cout << std::endl;
    return true; // TEST PASSED
}

/**
 * Helper function to read and test a single problem file.
 * Updates counters for processed and failed files.
 */
void process_problem_file(const fs::path& path, int& files_processed, int& files_failed) {
    // Check extension
    const std::string extension = path.extension().string();
    if (extension != ".txt" && extension != ".csc") {
        std::cout << "Skipping non-problem file: " << path.string() << std::endl;
        return;
    }

    files_processed++;

    // Run test for this file
    try {
        Problem p = read_problem(path.string());
        bool test_passed = run_solver_test(p, path.string());
        if (!test_passed) {
            files_failed++;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to read or test file: " << path.string() << std::endl;
        std::cerr << "  Error: " << e.what() << std::endl << std::endl;
        files_failed++;
    }
}

int main(int argc, char** argv) {
    // Set global logging state once for the entire test run
    logging::active = false;

    // Path to test (can be a file or directory)
    fs::path input_path = "test";
    if (argc > 1) {
        input_path = argv[1];
    }

    std::cout << "Processing path: " << input_path << std::endl;
    std::cout << "---------------------------------" << std::endl;

    int files_processed = 0;
    int files_failed = 0;

    try {
        // Check if the path is a directory
        if (fs::is_directory(input_path)) {
            std::cout << "Path is a directory, iterating..." << std::endl << std::endl;
            for (const auto& entry : fs::directory_iterator(input_path)) {
                // Only process regular files
                if (entry.is_regular_file()) {
                    process_problem_file(entry.path(), files_processed, files_failed);
                }
            }
        }
        // Check if the path is a single file
        else if (fs::is_regular_file(input_path)) {
            std::cout << "Path is a single file, processing..." << std::endl << std::endl;
            process_problem_file(input_path, files_processed, files_failed);
        }
        // Handle cases where the path doesn't exist or isn't a file/directory
        else {
            std::cerr << "Fatal Error: Path is not a valid file or directory: " << input_path
                      << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: Error while processing path: " << e.what() << std::endl;
        return 1;
    }

    // Print a final summary
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Test run complete." << std::endl;
    std::cout << "Processed: " << files_processed << " files" << std::endl;
    std::cout << "Failed:    " << files_failed << " files" << std::endl;

    return (files_failed > 0) ? 1 : 0; // Return error code if any tests failed
}
