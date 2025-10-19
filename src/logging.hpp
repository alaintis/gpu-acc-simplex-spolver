#pragma once
#include <iostream>
#include <string>
#include <vector>
using std::string;
using std::vector;

namespace logging {
template<typename T>
void log(string name, vector<T>& v){
    std::cout << name << "(" << v.size() << "): ";
    for(int i = 0; i < v.size(); i++) {
        std::cout << v[i] << ", ";
    }
    std::cout << std::endl;
}
}

