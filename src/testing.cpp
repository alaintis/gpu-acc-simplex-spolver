#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

#include "solver.hpp"
#include "logging.hpp"

namespace fs = std::filesystem;

// Container for a single LP problem
struct Problem {
    int m; // constraints (rows)
    int n; // variables (columns)
    vector<vector<double>> A; // m x n (rows)
    vector<double> b; // size m
    vector<double> c; // size n
};

// Read a problem file in the format:
// m n
// (m lines with n entries each) -- rows of A
// (one line with m numbers) -- b
// (one line with n numbers) -- c
Problem read_problem(const std::string &path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Could not open file: " + path);

    int m, n;
    in >> m >> n;
    Problem p;
    p.m = m;
    p.n = n;
    p.A.assign(n, vector<double>(m, 0.0));

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            in >> p.A[j][i];
        }
    }

    p.b.assign(m, 0.0);
    for(int i = 0; i < m; ++i) in >> p.b[i];

    p.c.assign(n, 0.0);
    for(int j = 0; j < n; ++j) in >> p.c[j];

    return p;
}

// Run solver on a Problem and print results
void test(const Problem &p, const std::string &name = "") {
    std::cout << "Running test" << (name.empty() ? "" : (": " + name)) << " (m=" << p.m << ", n=" << p.n << ")" << std::endl;
    try {
        std::cout << "Invoking solver..."<< "A.size=" << p.A.size() << " m=" << p.m << " n=" << p.n << std::endl;

        logging::active = false;
        struct result r = solver(p.m, p.n, p.A, p.b, p.c);
        if(r.success) {
            std::cout << "success";
        }
    }
    catch(const std::exception &e) {
        std::cerr << "Solver threw exception: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    std::string test_dir = "src/test";
    if(argc > 1) test_dir = argv[1];

    std::cout << "Testing problems in: " << test_dir << std::endl;

    try {
        for(const auto &entry : fs::directory_iterator(test_dir)) {
            if(!entry.is_directory()) continue;
            // find first file starting with "A_" inside the problem directory
            for(const auto &f : fs::directory_iterator(entry.path())) {
                std::string fname = f.path().filename().string();
                // assumined!! problem files start with "A_
                // no longer the case --- if(fname.rfind("A_", 0) == 0) {
                if(true) {
                    Problem p = read_problem(f.path().string());
                    test(p, f.path().string());
                    break; // one problem file per folder
                }
            }
        }
    }
    catch(const std::exception &e) {
        std::cerr << "Error while running tests: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
