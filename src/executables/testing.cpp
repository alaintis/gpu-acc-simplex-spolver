#include <iostream>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <iomanip>

#include "solver_wrapper.hpp" // backend (cpu/gpu/cuopt)
#include "base_solver.hpp"    // cuOpt reference solver
#include "linprog.hpp"        // score
#include "logging.hpp"
#include "problem_reader.hpp"

namespace fs = std::filesystem;

/**
 * Runs the solver for a single problem and prints the result.
 */
void run_solver_test(const Problem &p, const std::string &problem_name)
{
    std::cout << "Comparing on: " << problem_name
              << " (m=" << p.m << ", n=" << p.n << ")" << std::endl;

    result r_backend; // Result from backend (cpu/gpu)
    result r_cuopt;   // Result from cuOpt (base)

    // Run selected backend solver (via wrapper)
    try
    {
        r_backend = solver_wrapper(p.m, p.n, p.A, p.b, p.c);
        std::cout << "  Backend: "
                  << (r_backend.success ? "Success" : "Failed") << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "  Backend threw exception: " << e.what() << std::endl;
        r_backend.success = false;
    }

    // Run the cuOpt base solver (reference)
    try
    {
        r_cuopt = base_solver(p.m, p.n, p.A, p.b, p.c);
        std::cout << "  cuOpt:   "
                  << (r_cuopt.success ? "Success" : "Failed") << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "  cuOpt threw exception: " << e.what() << std::endl;
        r_cuopt.success = false;
    }

    // Compare results
    if (r_backend.success && r_cuopt.success)
    {
        // Both successful, compare objective scores
        double score_backend = score(p.n, r_backend.assignment, p.c);
        double score_cuopt = score(p.n, r_cuopt.assignment, p.c);
        double delta = std::fabs(score_backend - score_cuopt);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Scores -> Backend: " << score_backend
                  << " | cuOpt: " << score_cuopt
                  << " | Delta: " << delta << std::endl;

        std::cout.unsetf(std::ios_base::floatfield | std::ios_base::fixed);

        if (delta > 0.001)
        {
            std::cout << "  !! WARNING: Objective values differ significantly!" << std::endl;
        }
    }
    else if (r_backend.success != r_cuopt.success)
    {
        // One succeeded and one failed
        std::cout << "  !! WARNING: Solvers disagree on success!" << std::endl;
    }
    else
    {
        std::cout << "  (Both solvers failed or were infeasible)" << std::endl;
    }

    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    // Set global logging state once for the entire test run
    logging::active = false;

    // Test directory
    fs::path test_dir = "test";
    if (argc > 1)
    {
        test_dir = argv[1];
    }

    std::cout << "Testing problems in: " << test_dir << std::endl;
    std::cout << "---------------------------------" << std::endl;

    int files_processed = 0;
    int files_failed = 0;

    // Iterate through the directory
    try
    {
        for (const auto &entry : fs::directory_iterator(test_dir))
        {

            // Filter for valid problem files
            if (!entry.is_regular_file())
            {
                continue; // Skip weird files
            }

            const auto &path = entry.path();
            const std::string extension = path.extension().string();

            if (extension == ".txt" || extension == ".presolved")
            {
                files_processed++;

                // Run test for this file
                try
                {
                    Problem p = read_problem(path.string());
                    run_solver_test(p, path.string());
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Failed to read or test file: " << path.string() << std::endl;
                    std::cerr << "  Error: " << e.what() << std::endl
                              << std::endl;
                    files_failed++;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal Error: Error while iterating test directory: " << e.what() << std::endl;
        return 1;
    }

    // Print a final summary
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Test run complete." << std::endl;
    std::cout << "Processed: " << files_processed << " files" << std::endl;
    std::cout << "Failed:    " << files_failed << " files" << std::endl;

    return (files_failed > 0) ? 1 : 0; // Return error code if any tests failed
}