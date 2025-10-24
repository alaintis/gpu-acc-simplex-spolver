#include "linprog.hpp"
#include "linalg_cpu.hpp"

bool feasible(int n, int m, mat &A, vec &x, vec &b) {
    for(int i = 0; i < n; i++) {
        if(x[i] < 0) return false;
    }


    mat_cm A_dense(n*m);

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A_dense[i + j * m] = A[j][i];
        }
    }

    vec y = mv_mult(m, n, A_dense, x);
    for(int i = 0; i < m; i++) {
        if(y[i] > b[i]) return false;
    }

    return true;
}

double score(int n, vec &x, vec &c) {
    double s = 0;
    for(int i = 0; i < n; i++) {
        s += x[i] * c[i];
    }
    return s;
}
