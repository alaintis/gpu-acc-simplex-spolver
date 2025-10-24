#include <assert.h>
#include <cmath>

#include "solver.hpp"
#include "linalg.hpp"
#include "logging.hpp"

typedef vector<int> idx;


double eps = 1e-6;

// Implementation directly following the Rice Notes:
// https://www.cmor-faculty.rice.edu/~yzhang/caam378/Notes/Save/note2.pdf

struct result solver(int m, int n, mat A, vec b, vec c) {
    assert(A.size() == n);
    for(int i = 0; i < n; i++) {
        assert(A[i].size() == m);
    }
    assert(b.size() == m);
    assert(c.size() == n);

    idx B(m);
    vec x(m + n, 0);
    for(int i = 0; i < m; i++){
        vec e_i(m, 0);
        e_i[i] = 1;
        A.push_back(e_i);
        c.push_back(0);
        B[i] = n+i;
        x[i+n] = b[i];
    }

    idx N(n);
    for(int i = 0; i < n; i++) N[i] = i;

    mat_cm A_B(m*m);
    vec    c_B(m);
    vec    x_B(m);

    mat_cm A_N(m*n);
    vec    c_N(n);

    while(true) {
        logging::log("B", B);
        logging::log("N", N);

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < m; j++) {
                A_B[j + i * m] = A[B[i]][j];
            }
            c_B[i] = c[B[i]]; 
            x_B[i] = x[B[i]];
        }

        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                A_N[j + i*m] = A[N[i]][j];
            }
            c_N[i] = c[N[i]];
        }

        // 1. Dual estimates.
        vec y = mv_solve(m, A_B, c_B);
        mat_cm A_NT = m_transpose(m, n, A_N);
        vec tmp = mv_mult(n, m, A_NT, y);
        vec s_N = v_minus(n, c_N, tmp);
        
        logging::log("s_N", s_N);

        // 2. Check optimality.
        bool optimal = true;
        for(int i = 0; i < n; i++) {
            optimal &= s_N[i] >= -eps;
        }
        if(optimal) {
            x.resize(n);
            struct result res = {
                .success = true,
                .assignment = x
            };
            return res;
        }

        // 3. Selection of entering variable.
        // For now we choose the most negative entry.
        int j_i = 0;
        for(int i = 0; i < n; i++) {
            if(s_N[j_i] > s_N[i]) j_i = i;
        }
        int jj = N[j_i];

        // 4. Compute step
        vec d = mv_solve(m, A_B, A[jj]);

        logging::log("d", d);

        // 5. Check unboundedness
        bool unbounded = true;
        for(int i = 0; i < m; i++) {
            unbounded &= d[i] <= eps;
        }
        if(unbounded) {
            struct result res;
            res.success = false;
            return res;
        }

        // 6. Leaving Variable selection
        int r = -1;
        for(int i = 0; i < m; i++) {
            if(d[i] > 0 && (r == -1 || x_B[i]/d[i] < x_B[r]/d[r])) {
                r = i;
            }
        }
        assert(r >= 0);
        int ii = B[r];
        double tt = x_B[r]/d[r];

        // 7. Update variables
        x[jj] = tt;
        // x_B <== x_B - tt * d
        for(int i = 0; i < m; i++) {
            x[B[i]] = x_B[i] - tt * d[i];
        }

        // 8. Update Basis
        N[j_i] = ii;
        B[r]   = jj;
    }
}
