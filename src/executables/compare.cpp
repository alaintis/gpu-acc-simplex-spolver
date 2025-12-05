#include <chrono>
#include <cmath>
#include <iostream>

#include "../gpu_v0/base_solver_wrapped.hpp"
#include "linprog.hpp"
#include "logging.hpp"
#include "solver_wrapper.hpp"

double fRand(double min, double max) { return (((double)rand()) / RAND_MAX) * (max - min) + min; }

struct Timer {
    std::chrono::high_resolution_clock::time_point time_point;

    // Returns duration in milliseconds
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - time_point;
        return ms.count();
    }
    void start() {
        auto t = std::chrono::high_resolution_clock::now();
        time_point = t;
    }
};

void test(int seed) {
    srand(seed);
    int n = 30 + (rand() % 50);
    int m = 30 + (rand() % 50);

    vector<vec> A(n);
    vec b(m);
    vec c(n);

    for (int i = 0; i < n; i++) {
        A[i] = vec(m);
        for (int j = 0; j < m; j++) {
            A[i][j] = fRand(-1, 10);
        }
    }

    for (int i = 0; i < m; i++)
        b[i] = fRand(0, 1);
    for (int i = 0; i < n; i++)
        c[i] = fRand(-2, -1);

    // Set a single problem slightly negative.
    b[rand() % m] = -0.001;

    logging::log("A", A[0]);
    logging::log("b", b);
    logging::log("c", c);

    std::cout << "Test(seed: " << seed << ")" << std::endl;

    // [TMP] Convertion needed until we adapted our problems.
    int n_dash = n + m;
    mat A_dash = convert(m, n, A);
    vec c_dash(n_dash, 0);
    for (int i = 0; i < n; i++) {
        c_dash[i] = c[i];
    }

    vec x(n_dash);

    Timer timer_backend;
    Timer timer_base;
    timer_backend.start();
    struct result r = solver_wrapper(m, n_dash, A_dash, b, c_dash);
    double time_backend = timer_backend.stop();
    timer_base.start();
    struct result base_r = base_solver_wrapped(m, n_dash, A_dash, b, c_dash);
    double time_base = timer_base.stop();
    std::cout << "Base solver result:" << base_r.success << std::endl;
    std::cout << "Backend success? Base success?" << (r.success && base_r.success) << std::endl;
    std::cout << "Backend time (ms): " << time_backend << std::endl;
    std::cout << "Base time (ms): " << time_base << std::endl;

    double score_backend = 0, score_base = 0;
    ;
    double delta = 0.0;
    if (r.success && r.success == base_r.success) {
        score_backend = score(n, r.assignment, c);
        score_base = score(n, base_r.assignment, c);
        delta = std::fabs(score_backend - score_base);
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Data for statistics collection only enter if both are succesfull "
                  << std::endl;
        std::cout << "Backend time (ms), Base time (ms)" << std::endl;
        std::cout << "Delta" << std::endl;
        std::cout << "START" << std::endl;
        std::cout << "" << time_backend << "," << time_base << std::endl;
        std::cout << "" << delta << std::endl;
        std::cout << "END" << std::endl;
    }

    if (r.success == base_r.success && delta < 0.001) {
        std::cout << "success: " << r.success << std::endl;
        std::cout << "score backend: " << score_backend;
        std::cout << " score base: " << score_base;
        if (r.success)
            std::cout << " " << score(n, r.assignment, c);
        std::cout << std::endl;
    } else {
        std::cout << "failure" << std::endl;
        std::cout << "n: " << n << ", m: " << m << std::endl;
        std::cout << "Expected:" << std::endl;
        std::cout << base_r.success << std::endl;
        if (base_r.success) {
            for (int i = 0; i < n + m; i++)
                std::cout << base_r.assignment[i] << ", ";
            std::cout << std::endl;
            std::cout << "Score: " << score(n, base_r.assignment, c) << std::endl;
        }

        std::cout << "Result:" << std::endl;
        std::cout << r.success << std::endl;
        if (r.success) {
            for (int i = 0; i < n + m; i++)
                std::cout << r.assignment[i] << ", ";
            std::cout << std::endl;
            std::cout << "Score: " << score(n, r.assignment, c) << std::endl;
        }
    }
}

int main() {
    logging::active = false;
    for (int i = 0; i < 100; i++) {
        test(i);
    }
}
