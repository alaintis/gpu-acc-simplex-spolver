/**
 * Utility for simple Linear Programming questions.
 */
#include "types.hpp"

/**
 * Given an x, is it feasible in Ax <= b?
 */
bool feasible(int n, int m, mat &A, vec &x, vec &b);

/**
 * Given an x, evaluate its score.
 */
double score(int n, const vec &x, const vec &c);
