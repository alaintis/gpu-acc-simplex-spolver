Because `cuopt` still is an experimental library by NVidia it is not preinstalled.
Instead we have to install it using python, as it has a decent package manager.

# Installation
```bash
# Create and activate the venv in our home directory.
python3 -m venv ~/.venv
. ~/.venv/bin/activate

# Install the cuopt library this will take a bit.
pip install --pre --extra-index-url=https://pypi.nvidia.com 'libcuopt-cu12==25.10.*'
```

# VSCode or any Intellisense
Add `~/.venv/lib/python3.12/site-packages/libcuopt/include/**` to the includePaths.


# Usage
For each use of the library we need to link it into the compiler.
I hope to include this in the `Makefile` but for now we do it manually.
This is the snippet you need to compile it.
```bash
CUOPT_INCLUDE=~/.venv/lib/python3.12/site-packages/libcuopt/include/
CUOPT_LIB=~/.venv/lib/python3.12/site-packages/libcuopt/lib64/
gcc -I $CUOPT_INCLUDE -L $CUOPT_LIB lp_example.c -o lp_example
LD_LIBRARY_PATH=$CUOPT_LIB:LD_LIBRARY_PATH ./lp_example
```
