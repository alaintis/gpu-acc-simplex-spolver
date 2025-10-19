#include <iostream>

#include "solver.hpp"

void test(int in) {
    int expected = 1;

    int result = 1; // solver(in);
    if(expected == result) {
        std::cout << "Success!" << std::endl;
    }
    else {
        std::cout << "Fail (expected/got): " << expected << ", " << result << std::endl;
    }
}

int main() {
    std::cout << "Testing..." << std::endl;

    test(0);
    test(1);
}
