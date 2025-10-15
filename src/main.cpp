#include <iostream>

#include "solver.hpp"

int main() {
    std::cout << "Enter a number to solve." << std::endl;
    
    int num = 0;
    std::cin >> num;

    std::cout << "Result: " << solver(num) << std::endl;
}
