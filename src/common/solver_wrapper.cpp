#include <iostream>
#include "solver_wrapper.hpp"
#include "linprog.hpp"

/**
 * Auxiliary Problem
 * 
 * Ax <= b.
 * We detect each constraint i with b[i] < 0.
 * For those cases we add a variable aux_i with initial value aux_i = -b[i].
 * 
 * This gives us a new LP with a known basic feasible solution.
 * min sum(aux_i)
 * Ax - sum(e_i) <= b.
 * 
 * if there is some feasible solution, we will find it and get sum(aux_i) = 0.
 */


struct result solver_wrapper(int m, int n, vector<vector<double>> A, vector<double> b, vector<double> c) {
    vector<double> x(n, 0);
    
    bool positive = true;
    for(int i = 0; i < m; i++) {
        if(b[i] < 0) positive = false;
    }

    if(!positive) {
        vector<vector<double>> A_aux = A;
        vector<double> c_aux(n, 0);
        vector<double> x_aux = x;

        // A being column major, we add rows
        for(int i = 0; i < m; i++) {
            if(b[i] < 0) {
                vector<double> e_i(m, 0);
                e_i[i] = -1;

                A_aux.push_back(e_i);
                x_aux.push_back(-b[i]);
                c_aux.push_back(1);
            }
        }

        result res = solver(m, x_aux.size(), A_aux, b, c_aux, x_aux);
        if(!res.success) return res;
        
        x = res.assignment;
        x.resize(n);

        if(!feasible(n, m, A, x, b)) {
            std::cout << "Still not feasible" << std::endl;
            result res;
            res.success = false;
            return res;
        }
    }

    return solver(m, n, A, b, c, x);
}
