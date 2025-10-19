#include <iostream>

#include "solver.hpp"
#include "linprog.hpp"

int main() {
    int n = 2;
    int m = 2;

    vector<vector<double>> A = {
        {2, 5},
        {8, 2}
    };
    vector<double> b = { 60, 60 };
    vector<double> c = { -40, -88 };

    struct result r = solver(n, m, A, b, c);

    std::cout << "Result: " << ((r.success) ? "Success" : "Failed") << std::endl;


    if(r.success) {
        std::cout << "Feasible: " << ((feasible(n, m, A, r.assignment, b)) ? "Yes" : "No") << std::endl;
        std::cout << "Score: " << score(n, r.assignment, c) << std::endl;
        for(int i = 0; i < n; i++) {
            std::cout << i << ": " << r.assignment[i] << std::endl;
        }
    }
}
