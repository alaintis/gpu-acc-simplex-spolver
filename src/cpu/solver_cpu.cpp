#include <assert.h>
#include <cmath>

#include "linalg_cpu.hpp"
#include "logging.hpp"
#include "solver.hpp"

typedef vector<int> idx;

double eps = 1e-6;

vec Ax_mult(int m, int n, mat& A, vec& x) {
    vec y(m);

    for (int i = 0; i < m; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[j][i] * x[j];
        }
        y[i] = sum;
    }

    return y;
}

// Implementation directly following the Rice Notes:
// https://www.cmor-faculty.rice.edu/~yzhang/caam378/Notes/Save/note2.pdf
// It accept as input Ax <= b, with initial solution x in column major format.
extern "C" struct result solver(int m, int n, mat A, vec b, vec c, vec x) {
    assert(A.size() == n);
    for (int i = 0; i < n; i++) {
        assert(A[i].size() == m);
    }
    assert(b.size() == m);
    assert(c.size() == n);
    assert(x.size() == n);

    vec y = Ax_mult(m, n, A, x);

    // Adding slack variables to obtain Ax = b
    for (int i = 0; i < m; i++) {
        vec e_i(m, 0);
        e_i[i] = 1;
        A.push_back(e_i);
        c.push_back(0);
        x.push_back(b[i] - y[i]);
    }

    // Selection of the Basis and Non Basis, currently we don't accept initial conditions
    // with zeros in the basis. This is because we otherwise would have to solve some linear
    // systems to get started.
    int zero_count = 0;
    for (int i = 0; i < n + m; i++)
        if (std::abs(x[i]) < eps)
            zero_count += 1;
    if (zero_count != n) {
        std::cout << "No Non-Basis / Basis split found." << std::endl;
        result res;
        res.success = false;
        return res;
    }

    idx B(m);
    idx N(n);
    int n_count = 0;
    int b_count = 0;
    for (int i = 0; i < n + m; i++) {
        if (std::abs(x[i]) < eps) {
            N[n_count++] = i;
        } else {
            B[b_count++] = i;
        }
    }

    mat_cm A_B(m * m);
    vec c_B(m);
    vec x_B(m);

    mat_cm A_N(m * n);
    vec c_N(n);

    for (int i = 0; i < 100; i++) {
        logging::log("B", B);
        logging::log("N", N);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                A_B[j + i * m] = A[B[i]][j];
            }
            c_B[i] = c[B[i]];
            x_B[i] = x[B[i]];
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                A_N[j + i * m] = A[N[i]][j];
            }
            c_N[i] = c[N[i]];
        }

        // 1. Dual estimates.
        mat_cm A_BT = m_transpose(m, m, A_B);
        vec y = mv_solve(m, A_BT, c_B);
        mat_cm A_NT = m_transpose(m, n, A_N);
        vec tmp = mv_mult(n, m, A_NT, y);
        vec s_N = v_minus(n, c_N, tmp);

        logging::log("s_N", s_N);

        // 2. Check optimality.
        bool optimal = true;
        for (int i = 0; i < n; i++) {
            optimal &= s_N[i] >= -eps;
        }
        if (optimal) {
            x.resize(n);
            struct result res = {.success = true, .assignment = x};
            return res;
        }

        // 3. Selection of entering variable.
        // For now we choose the most negative entry.
        int j_i = 0;
        for (int i = 0; i < n; i++) {
            if (s_N[j_i] > s_N[i])
                j_i = i;
        }
        int jj = N[j_i];

        // 4. Compute step
        vec d = mv_solve(m, A_B, A[jj]);

        logging::log("d", d);

        // 5. Check unboundedness
        bool unbounded = true;
        for (int i = 0; i < m; i++) {
            unbounded &= d[i] <= eps;
        }
        if (unbounded) {
            struct result res;
            res.success = false;
            return res;
        }

        // 6. Leaving Variable selection
        int r = -1;
        for (int i = 0; i < m; i++) {
            if (d[i] > eps && (r == -1 || x_B[i] / d[i] < x_B[r] / d[r])) {
                r = i;
            }
        }
        assert(r >= 0);
        int ii = B[r];
        double tt = x_B[r] / d[r];

        // 7. Update variables
        x[jj] = tt;
        // x_B <== x_B - tt * d
        for (int i = 0; i < m; i++) {
            x[B[i]] = x_B[i] - tt * d[i];
        }

        // 8. Update Basis
        N[j_i] = ii;
        B[r] = jj;
    }

    result res;
    res.success = false;
    return res;
}
