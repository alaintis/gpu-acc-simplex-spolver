#include <iostream>
#include <cmath>
#include "linprog.hpp"
#include "types.hpp"

static double eps = 1e-6;

// Private helper for column-major matrix-vector multiply (y = Ax)
static vec Ax_mult_col_major(int m, int n, mat &A, vec &x)
{
    vec y(m, 0.0); // Initialize y to all zeros
    for (int j = 0; j < n; ++j)
    { // Iterate over columns (variables)
        for (int i = 0; i < m; ++i)
        { // Iterate over rows (constraints)
            y[i] += A[j][i] * x[j];
        }
    }
    return y;
}

bool feasible(int n, int m, mat &A, vec &x, vec &b)
{
    // 1. Check x >= 0
    for (int i = 0; i < n; i++)
    {
        if (x[i] < -eps)
            return false;
    }

    // 2. Check Ax <= b
    vec y = Ax_mult_col_major(m, n, A, x);
    for (int i = 0; i < m; i++)
    {
        if (y[i] > b[i] + eps)
            return false;
    }

    return true;
}

double score(int n, const vec &x, const vec &c)
{
    double s = 0;
    for (int i = 0; i < n; i++)
    {
        s += x[i] * c[i];
    }
    return s;
}