#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>    
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>


#include "logging.hpp"
#include "problem_reader.hpp"
#include "solver_wrapper.hpp"

namespace fs = std::filesystem;

// Record structure to hold benchmark results
struct record {
    double total_time;
    double avg_time;
    double min_time;
    double max_time;
    bool all_successful;
};

// Function to output records to CSV
void output_csv(const std::vector<std::filesystem::path>& files_to_benchmark,
                const std::vector<record>& records,
                const std::string& output_filename = "benchmark_results.csv")
{
    // Sanity check: both vectors must have the same size
    

    // Open CSV file for writing
    std::ofstream csv_file(output_filename);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open output CSV file: " << output_filename << "\n";
        return;
    }

    // Write CSV header
    csv_file << "File,Total Time (s),Average Time (s),Min Time (s),Max Time (s),All Successful\n";

    // Write each row
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& path = files_to_benchmark[i];
        const auto& rec = records[i];

        csv_file << '"' << path.string() << '"' << ','      // Quote path to handle commas
                 << rec.total_time << ','
                 << rec.avg_time << ','
                 << rec.min_time << ','
                 << rec.max_time << ','
                 << (rec.all_successful ? "true" : "false")
                 << '\n';
    }

    csv_file.close();
    std::cout << "CSV file written successfully: " << output_filename << "\n";
}; 


struct Timer {
    std::chrono::high_resolution_clock::time_point time_point;
    
    // Returns duration in milliseconds
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - time_point;
        return ms.count();
    }
    void  start() {
        auto t = std::chrono::high_resolution_clock::now();
        time_point = t;
    }
};

record run_benchmark(const Problem& p,
                   const std::string& problem_name,
                   int warmup_runs,
                   int timed_runs) {
    std::cout << "Benchmarking: " << problem_name << " (m=" << p.m << ", n=" << p.n << ")"
              << std::endl;

    // Warm-up runs
    for (int i = 0; i < warmup_runs; ++i) {
        solver_wrapper(p.m, p.n, p.A, p.b, p.c);
    }

    // Timed runs
    std::vector<double> timings;
    timings.reserve(timed_runs);
    bool all_successful = true;

    for (int i = 0; i < timed_runs; ++i) {
        Timer timer;
        timer.start();
        result r = solver_wrapper(p.m, p.n, p.A, p.b, p.c);
        timings.push_back(timer.stop());

        if (!r.success) {
            all_successful = false;
        }
    }

    // Calculate and report statistics
    if (timings.empty()) {
        std::cout << "  No timed runs executed." << std::endl;
        return record{0, 0, 0, 0, false};
    }

    double total_time = std::accumulate(timings.begin(), timings.end(), 0.0);
    double avg_time = total_time / timings.size();
    double min_time = *std::min_element(timings.begin(), timings.end());
    double max_time = *std::max_element(timings.begin(), timings.end());
    // Create and return record
    record rec{total_time, avg_time, min_time, max_time, all_successful};

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Runs: " << timed_runs << " (after " << warmup_runs << " warm-ups)" << std::endl;
    std::cout << "  Avg:  " << avg_time << " ms" << std::endl;
    std::cout << "  Min:  " << min_time << " ms (Fastest)" << std::endl;
    std::cout << "  Max:  " << max_time << " ms" << std::endl;

    if (!all_successful) {
        std::cout << "  !! WARNING: At least one run failed!" << std::endl;
    }

    std::cout.unsetf(std::ios_base::floatfield | std::ios_base::fixed);
    std::cout << std::endl;
    return rec;
}

int main(int argc, char** argv) {
    // Benchmarking should be silent
    logging::active = false;

    const int WARMUP_RUNS = 5;
    const int TIMED_RUNS = 10;

    std::vector<fs::path> files_to_benchmark;
    std::string run_description;

    try {
        if (argc > 1) {
            // User provided an argument
            fs::path user_path = argv[1];
            if (!fs::exists(user_path)) {
                std::cerr << "Error: Path not found: " << user_path << std::endl;
                return 1;
            }

            if (fs::is_directory(user_path)) {
                // Arg is a DIRECTORY
                run_description = "Benchmarking files in: " + user_path.string();
                for (const auto& entry : fs::directory_iterator(user_path)) {
                    if (entry.is_regular_file()) {
                        const std::string ext = entry.path().extension().string();
                        if (ext == ".txt" || ext == ".presolved") {
                            files_to_benchmark.push_back(entry.path());
                        }
                    }
                }
            } else if (fs::is_regular_file(user_path)) {
                // Arg is a FILE
                run_description = "Benchmarking single file: " + user_path.string();
                files_to_benchmark.push_back(user_path);
            } else {
                std::cerr << "Error: Path is not a file or directory: " << user_path << std::endl;
                return 1;
            }
        } else {
            // Path 2: No argument
            fs::path test_dir = "test";
            run_description = "Benchmarking files in default: " + test_dir.string();
            if (!fs::exists(test_dir)) {
                std::cerr << "Error: Default test directory not found: " << test_dir << std::endl;
                return 1;
            }
            for (const auto& entry : fs::directory_iterator(test_dir)) {
                if (entry.is_regular_file()) {
                    const std::string ext = entry.path().extension().string();
                    if (ext == ".txt" || ext == ".presolved") {
                        files_to_benchmark.push_back(entry.path());
                    }
                }
            }
        }

        // run the benchmark loop
        std::cout << "Starting benchmark run..." << std::endl;
        std::cout << "Config: " << WARMUP_RUNS << " warm-up, " << TIMED_RUNS << " timed runs."
                  << std::endl;
        std::cout << run_description << std::endl;
        std::cout << "---------------------------------" << std::endl;

        if (files_to_benchmark.empty()) {
            std::cout << "No problem files found to benchmark." << std::endl;
        } else {
            std::cout << "Found " << files_to_benchmark.size() << " problem files." << std::endl
                      << std::endl;
        }
        int count = 0;
        int total = files_to_benchmark.size();
        std::vector<record> records;
        for (const auto& path : files_to_benchmark) {
            try {
                Problem p = read_problem(path.string());
                std::cout << "----is running number " << count++ << " of  " << total << ": " << path.string() << " ---------------------------" << std::endl;
                // Run benchmark and store record
                records.push_back(run_benchmark(p, path.string(), WARMUP_RUNS, TIMED_RUNS));

            // Limit to 10 files for quick testing
            if(count > 10){
                std::cout << "------Attention!!!---------------------------" << std::endl;
                std::cout << " There is a break after 10 files for quick testing purposes " << std::endl;
                break;
            }           

            } catch (const std::exception& e) {
                // Catch errors from read_problem or solver
                std::cerr << "Failed to benchmark file: " << path.string() << std::endl;
                std::cerr << "  Error: " << e.what() << std::endl << std::endl;
            }
        }
        // You can now use the 'records' vector for further processing or reporting
        output_csv(files_to_benchmark, records);

    } catch (const std::exception& e) {
        // Catch fatal errors like directory iteration
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Benchmark run complete. Processed " << files_to_benchmark.size() << " files."
              << std::endl;
    return 0;
}