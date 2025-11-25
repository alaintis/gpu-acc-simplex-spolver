#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
using std::string;

#include "problem_reader.hpp"

Problem read_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Could not open file: " + path);

    string name, fmt;
    int m, n, nnz;
    in >> name >> fmt >> m >> n >> nnz;

    if (fmt != "csc") {
        throw std::runtime_error("Expected A with 'csc' format");
    }

    Problem p;
    p.m = m;
    p.n = n;
    // Allocate A as n columns, each of size m
    p.A.assign(n, std::vector<double>(m, 0.0));

    // Read CSC structure
    std::vector<int> col_ptr(n + 1);
    for (int j = 0; j <= n; ++j) {
        in >> col_ptr[j];
    }

    std::vector<int> row_idx(nnz);
    for (int k = 0; k < nnz; ++k) {
        in >> row_idx[k];
    }

    std::vector<double> values(nnz);
    for (int k = 0; k < nnz; ++k) {
        in >> values[k];
    }

    // scatter
    for (int j = 0; j < n; ++j) {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            int i = row_idx[k];     // row
            double v = values[k];
            p.A[j][i] = v;          // column-major
        }
    }

    int bm;
    in >> name >> fmt >> bm;
    if (fmt != "dense" || bm != m) {
        throw std::runtime_error("Expected b with 'dense' format and m entries");
    }

    // Read b vector (size m)
    p.b.assign(m, 0.0);
    for (int i = 0; i < m; ++i) {
        in >> p.b[i];
    }


    int cn;
    in >> name >> fmt >> cn;
    if (fmt != "dense" || cn != n) {
        throw std::runtime_error("Expected c with 'dense' format and n entries");
    }

    // Read c vector (size n)
    p.c.assign(n, 0.0);
    for (int j = 0; j < n; ++j) {
        in >> p.c[j];
    }

    return p;
}
