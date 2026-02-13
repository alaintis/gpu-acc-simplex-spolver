#pragma once

#include <string>
#include <vector>

#include "types.hpp"

struct Problem {
    int m;
    int n;
    mat_csc A;
    vec b;
    vec c;
};

/**
 * Reads a problem file in the format:
 * m n
 * (m lines with n entries each) -- rows of A (row-major)
 * (one line with m numbers) -- b
 * (one line with n numbers) -- c
 * * The function loads the A matrix into the column-major format required
 * by the solver.
 */
Problem read_problem(const std::string& path);
