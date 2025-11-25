#pragma once
#include <vector>
using std::vector;
typedef vector<int> idx;
typedef vector<double> vec;
typedef vector<vec> mat;

/**
 * Matrix column major order
 * [0, 1]
 * [2, 3]
 *
 * Is stored as [0, 2, 1, 3].
 */

typedef vector<double> mat_cm;
static double eps = 1e-6;
