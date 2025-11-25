/**
 * Utility for simple Linear Programming questions.
 */
#include "types.hpp"

/**
 * Given an x, is it feasible in Ax <= b?
 */
bool feasible(int n, int m, const mat& A, const vec& x, const vec& b);

/**
 * Given an x, evaluate its score.
 */
double score(int n, const vec& x, const vec& c);

/**
 * Converts a leq problem into an equality problem.
 * Input A, output A'
 * such that Ax <= b has a solution iff Ax' = b has one. where x' = [x, slack]
 */
mat convert(int m, int n, const mat &A);
