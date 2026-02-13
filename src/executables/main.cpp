#include <iostream>
#include <cmath>
#include "types.hpp"


#include "linprog.hpp"
#include "solver_wrapper.hpp"

int main() {
    int n = 2;
    int m = 2;

    vector<vector<double>> A = {{2, 5}, {8, 2}};
    vector<double> b = {60, 60};
    vector<double> c = {-40, -88};

    // [TMP] Convertion needed until we adapted our problems.
    int n_dash = n + m;
    mat A_dash = convert(m, n, A);
    vec c_dash(n + m, 0);
    for(int i = 0; i < n; i++) {
        c_dash[i] = c[i];
    }

    mat_csc A_final = {
        .col_ptr = idx(n_dash + 1),
        .row_idx = idx(),
        .values  = vec()
    };

    int nnz = 0;
    for(int i = 0; i < n_dash; i++) {
        A_final.col_ptr[i] = nnz;
        for(int j = 0; j < m; j++) {
            if (std::fabs(A_dash[i][j]) > 1e-9) {
                A_final.row_idx.push_back(j);
                A_final.values.push_back(A_dash[i][j]);
                nnz += 1;
            }
        }
    }
    A_final.col_ptr[n_dash] = nnz;

    struct result r = solver_wrapper(m, n_dash, A_final, b, c_dash);

    std::cout << "Result: " << ((r.success) ? "Success" : "Failed") << std::endl;

    if (r.success) {
        std::cout << "Feasible: " << ((feasible(n, m, A, r.assignment, b)) ? "Yes" : "No")
                  << std::endl;
        std::cout << "Score: " << score(n, r.assignment, c) << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << i << ": " << r.assignment[i] << std::endl;
        }
    }
}
