#include "linprog.hpp"

#include <cmath>
#include <assert.h>
#include <iostream>

#include "types.hpp"

// Private helper for column-major matrix-vector multiply (y = Ax)
static vec Ax_mult_col_major(int m, int n, const mat& A, const vec& x) {
    vec y(m, 0.0); // Initialize y to all zeros
    for (int j = 0; j < n; ++j) { // Iterate over columns (variables)
        for (int i = 0; i < m; ++i) { // Iterate over rows (constraints)
            y[i] += A[j][i] * x[j];
        }
    }
    return y;
}

bool feasible(int n, int m, const mat& A, const vec& x, const vec& b) {
    // 1. Check x >= 0
    for (int i = 0; i < n; i++) {
        if (x[i] < -1e-6)
            return false;
    }

    // 2. Check Ax = b
    vec y = Ax_mult_col_major(m, n, A, x);
    for (int i = 0; i < m; i++) {
        if (std::abs(y[i] - b[i]) > 1e-3) {
            std::cout << std::abs(y[i] - b[i]) << std::endl;
            return false;
        }
    }

    return true;
}

double score(int n, const vec& x, const vec& c) {
    double s = 0;
    for (int i = 0; i < n; i++) {
        s += x[i] * c[i];
    }
    return s;
}

mat convert(int m, int n, const mat &A) {
    assert(A.size() == n);
    for(int i = 0; i < n; i++) {
        assert(A[i].size() == m);
    }

    mat A_dash(m + n);
    for(int i = 0; i < n; i++) {
        A_dash[i] = A[i];
    }
    for(int i = 0; i < m; i++) {
        A_dash[n+i] = vec(m, 0);
        A_dash[n+i][i] = 1;
    }
    return A_dash;
}
