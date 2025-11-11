#include "problem_reader.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>

Problem read_problem(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Could not open file: " + path);

    int m, n;
    in >> m >> n;
    Problem p;
    p.m = m;
    p.n = n;

    // Allocate A as n columns, each of size m
    p.A.assign(n, std::vector<double>(m, 0.0));

    // Read A matrix (row-major) and store as column-major
    for (int i = 0; i < m; ++i)
    { // iterate rows
        for (int j = 0; j < n; ++j)
        {                    // iterate columns
            in >> p.A[j][i]; // Store in A[col][row]
        }
    }

    // Read b vector (size m)
    p.b.assign(m, 0.0);
    for (int i = 0; i < m; ++i)
        in >> p.b[i];

    // Read c vector (size n)
    p.c.assign(n, 0.0);
    for (int j = 0; j < n; ++j)
        in >> p.c[j];

    return p;
}