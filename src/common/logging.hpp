#pragma once
#include <iostream>
#include <string>
#include <vector>
using std::string;
using std::vector;

namespace logging {
extern bool active;
template<typename T>
void log(string name, const vector<T>& v) {
    if(!active) return;
    std::cout << name << "(" << v.size() << "): ";
    for(int i = 0; i < v.size(); i++) {
        std::cout << v[i] << ", ";
    }
    std::cout << std::endl;
}
}

