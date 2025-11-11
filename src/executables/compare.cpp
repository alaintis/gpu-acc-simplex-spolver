#include <iostream>
#include <cmath>

#include "solver_wrapper.hpp"
#include "base_solver.hpp"
#include "linprog.hpp"
#include "logging.hpp"

double fRand(double min, double max) {
    return (((double) rand())/RAND_MAX) * (max - min) + min;
}

void test(int seed) {
    srand(seed);
    int n = 100 + (rand() % 30);
    int m = rand() % 200;

    vector<vec> A(n);
    vec b(m);
    vec c(n);
    
    for(int i = 0; i < n; i++) {
        A[i] = vec(m);
        for(int j = 0; j < m; j++) {
            A[i][j] = fRand(-1, 10);
        }
    }

    for(int i = 0; i < m; i++) b[i] = fRand(0, 1);
    for(int i = 0; i < n; i++) c[i] = fRand(-2, -1);

    // Set a single problem negative.
    b[rand() % m] = -0.01;

    logging::log("A", A[0]);
    logging::log("b", b);
    logging::log("c", c);

    std::cout << "Test(seed: " << seed << ")" << std::endl;
    
    vec x(n, 0);
    if(!feasible(n, m, A, x, b)) {
        std::cout << "No 0 solution!" << std::endl;
    }

    struct result r = solver_wrapper(m, n, A, b, c);
    struct result base_r = base_solver(m, n, A, b, c);
    
    double delta = 0.0;
    if(r.success && r.success == base_r.success) {
        delta = std::fabs(score(n, r.assignment, c) - score(n, base_r.assignment, c));
    }
    
    if (r.success == base_r.success && delta < 0.001) {
        std::cout << "success: " << r.success;
        if(r.success) std::cout << " " << score(n, r.assignment, c);
        std::cout << std::endl;
    } else {
        std::cout << "failure" << std::endl;
        std::cout << "n: " << n << ", m: " << m << std::endl;
        std::cout << "Expected:" << std::endl;
        std::cout << base_r.success << std::endl;
        if(base_r.success) {
            for(int i = 0; i < n; i++) std::cout << base_r.assignment[i] << ", ";
            std::cout << std::endl;
            std::cout << "Score: " << score(n, base_r.assignment, c) << std::endl;
        }
        
        std::cout << "Result:" << std::endl;
        std::cout << r.success << std::endl;
        if(r.success) {
            for(int i = 0; i < n; i++) std::cout << r.assignment[i] << ", ";
            std::cout << std::endl;
            std::cout << "Score: " << score(n, r.assignment, c) << std::endl;
        }
    }
}

int main() {
    logging::active = false;
    for(int i = 0; i < 100; i++) {
        test(i);
    }
}
