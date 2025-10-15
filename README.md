# Simplex Solver

Solve regularly shaped Linear Programs optimally by moving through basic feasible solutions.

## Code Organization

The `src` repository contains all source files, both those defining the different solvers and
any helpers we need to test and execute it.

### Makefile

The `Makefile` is core for both executing and building the executables. There are two main
commands.
1. `make run` runs the solver as an executable.
2. `make test` runs the solver in a testing environment where it is exposed to different problems.

To choose which solver to use we can use the `BACKEND` flag. Currently there is either `cpu` or `cuda`(default).
A full run command would therefore be `make run BACKEND=cuda`.
