#!/bin/bash

CUOPT_INCLUDE=~/dphpc/venv/lib/python3.12/site-packages/libcuopt/include/
CUOPT_LIB=~/dphpc/venv/lib/python3.12/site-packages/libcuopt/lib64/
gcc -I $CUOPT_INCLUDE -L $CUOPT_LIB -c -o lp_example.o lp_example.c -lcuopt
g++ -L $CUOPT_LIB -o lp_example lp_example.o -lcuopt
# LD_LIBRARY_PATH=~/dphpc/venv/lib/python3.12/site-packages/libcuopt/lib64/:$LD_LIBRARY_PATH ./lp_example