#!/bin/bash

# If this doesn't match your code then we have to find a better solution.
CUOPT_INCLUDE=~/.venv/lib/python3.12/site-packages/libcuopt/include/

if [[ ! -d "$CUOPT_INCLUDE" ]]; then
    echo "[compiler] ERROR: libcuopt was not found at the expected position ($CUOPT_INCLUDE)."
    echo "Please refer to $PWD/README.md to install the library."
    exit 1
fi

g++ $@
