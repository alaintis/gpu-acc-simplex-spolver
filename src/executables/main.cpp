#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

// #include "base_solver.hpp" // REMOVED
#include "linprog.hpp" // for score()
#include "logging.hpp"
#include "problem_reader.hpp"
#include "solver_wrapper.hpp" // Your backend

namespace fs = std::filesystem;

// Timer utility
struct Timer {
    std::chrono::high_resolution_clock::time_point time_point;

    void start() { time_point = std::chrono::high_resolution_clock::now(); }

    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - time_point;
        return ms.count();
    }
};

/**
 * Runs the backend solver for a single problem.
 * Returns TRUE if the solver reported success, FALSE otherwise.
 */
bool run_backend_solver(const Problem& p, const std::string& problem_name) {
    std::cout << "Solving: " << problem_name << " (m=" << p.m << ", n=" << p.n << ")" << std::endl;

    result r;
    Timer timer;
    double elapsed_ms = 0.0;
    bool crashed = false;

    // Run Backend
    try {
        timer.start();
        r = solver_wrapper(p.m, p.n, p.A, p.b, p.c);
        elapsed_ms = timer.stop();
    } catch (const std::exception& e) {
        std::cerr << "  !! Exception during solve: " << e.what() << std::endl;
        crashed = true;
        r.success = false;
    }

    // Report Results
    if (r.success) {
        double objective_value = score(p.n, r.assignment, p.c);

        std::cout << "  Status:      SUCCESS" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Objective:   " << objective_value << std::endl;
        std::cout << "  Time:        " << elapsed_ms << " ms" << std::endl;
        std::cout << "---------------------------------" << std::endl;
        return true;
    } else {
        std::cout << "  Status:      FAILED" << std::endl;
        if (!crashed) {
            std::cout << "  Time:        " << elapsed_ms << " ms" << std::endl;
        }
        std::cout << "---------------------------------" << std::endl;
        return false;
    }
}

/**
 * Helper to read and solve a file.
 */
void process_problem_file(const fs::path& path, int& files_processed, int& files_solved) {
    const std::string extension = path.extension().string();
    if (extension != ".txt" && extension != ".csc") {
        return; // Skip non-problem files silently
    }

    files_processed++;

    try {
        Problem p = read_problem(path.string());
        bool success = run_backend_solver(p, path.filename().string());
        if (success) {
            files_solved++;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to read file: " << path.string() << std::endl;
        std::cerr << "  Error: " << e.what() << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
}

int main(int argc, char** argv) {
    logging::active = false;

    fs::path input_path = "test";
    if (argc > 1) {
        input_path = argv[1];
    }

    std::cout << "=================================" << std::endl;
    std::cout << "Solver Backend Test" << std::endl;
    std::cout << "Target: " << input_path << std::endl;
    std::cout << "=================================" << std::endl;

    int files_processed = 0;
    int files_solved = 0;

    try {
        if (fs::is_directory(input_path)) {
            // Iterate over all files in directory
            for (const auto& entry : fs::directory_iterator(input_path)) {
                if (entry.is_regular_file()) {
                    process_problem_file(entry.path(), files_processed, files_solved);
                }
            }
        } else if (fs::is_regular_file(input_path)) {
            // Process single file
            process_problem_file(input_path, files_processed, files_solved);
        } else {
            std::cerr << "Fatal Error: Invalid path: " << input_path << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error during processing: " << e.what() << std::endl;
        return 1;
    }

    // Final Summary
    std::cout << std::endl;
    std::cout << "======== SUMMARY ========" << std::endl;
    std::cout << "Total Processed: " << files_processed << std::endl;
    std::cout << "Solved Success:  " << files_solved << std::endl;
    std::cout << "Failed/Infeas:   " << (files_processed - files_solved) << std::endl;
    std::cout << "=========================" << std::endl;

    return (files_solved == files_processed) ? 0 : 1;
}
