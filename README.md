# Simplex Solver

Solve regularly shaped Linear Programs optimally by moving through basic feasible solutions. This solver is written in C++ and CUDA. The project is designed for modularity and speed, allowing you to swap between different compute backends (CPU, GPU) and provides a full suite of tools for correctness testing and performance profiling.The solver is designed to solve LPs in the form: ```min c^T x``` ```s.t. Ax <= b``` ```x >= 0```.

## 1. Project Structure
The codebase is organized into modular libraries and distinct executables:
```
├── CMakeLists.txt               # Root CMake file, defines SOLVER_BACKEND
├── README.md                    # This project guide
├── executables/                 # All main programs
│   ├── CMakeLists.txt           # Defines the executable targets
│   ├── testing.cpp              # Source for 'test_runner' (Correctness)
|   |── compare.cpp              # Source for 'compare_tool' (Correctness)
│   └── benchmarking.cpp         # Source for 'benchmark_runner' (Performance)
├── src/
│   ├── common/                  # Shared code for all backends
│   │   ├── problem_reader.hpp   # Loads problems from files
│   │   ├── linprog.hpp          # Utility functions (score, feasible)
│   │   |── solver.hpp           # Interface for solvers
│   │   └── solver_wrapper.hpp   # Handles auxiliary problem setup
│   ├── cpu/                     # CPU Backend
│   │   └── solver_cpu.cpp
│   ├── gpu/                     # GPU Backend (Optimized)
│   │   ├── solver_gpu.cpp
│   │   └── linalg_gpu.cu
│   └── cuopt/                   # NVIDIA cuOpt Backend (Reference)
│       ├── solver_cuopt.cpp     # Wrapper for cuOpt library
│       └── base_solver.hpp      # Header for the reference solver
└── test/                        # Default directory for small test problems
```

## 2. Prerequisites & Setup
Before building, you need to set up the required dependencies.
Dependencies
- CMake (v3.20+)
- C++ Compiler (supporting C++20)
- NVIDIA CUDA Toolkit (v12.0+). Must include:
  - nvcc (CUDA compiler)
  - cusolver
  - cublas
- NVIDIA cuOpt (This is a separate library)

### Installing cuOpt (Required)
The project uses NVIDIA's cuOpt library as a "golden reference" for correctness testing. It must be installed via pip.
```bash
# Create and activate the venv in our home directory.
python3 -m venv ~/.venv
. ~/.venv/bin/activate

# Install the cuopt library this will take a bit.
pip install --pre --extra-index-url=https://pypi.nvidia.com 'libcuopt-cu12==25.10.*'

# VSCode or any Intellisense
Add `~/.venv/lib/python3.12/site-packages/libcuopt/include/**` to the includePaths.
```

Important: The build system assumes cuOpt is installed at: ```$HOME/.venv/lib/python3.12/site-packages/libcuopt```

If your path is different (e.g., a different Python version), you must tell CMake where to find it by setting the ```cuOpt_DIR``` variable during the configure step (see section 3).

## 3. How to Build
The build is controlled by the ```SOLVER_BACKEND``` CMake variable, which selects the implementation for your main solver.

### Step 1: Configure CMake
Create a build directory and run cmake. You must specify which backend you want to test.

**Option 1: Build for GPU (Default for Benchmarking)** This builds the gpu solver as the main backend.
```bash
cmake -S . -B build -DSOLVER_BACKEND=gpu
```

**Option 2: Build for CPU** This builds the cpu solver as the main backend.
```bash
cmake -S . -B build -DSOLVER_BACKEND=cpu
```

**Option 3: Build with cuOpt** This builds the cuopt solver as the main backend.
```bash
cmake -S . -B build -DSOLVER_BACKEND=cuopt
```

**If your cuOpt path is different:** Add the ```-DcuOpt_DIR``` flag to your cmake command:
```bash
cmake -S . -B build -DSOLVER_BACKEND=gpu -DcuOpt_DIR=/path/to/your/libcuopt
```

### Step 2: Build the Executables
After configuring, you can build all targets or just the ones you need.

```bash
# Build all executables
cmake --build build

# Or (the superior method)
cd build
make

# Or, build a specific executable
cmake --build build --target benchmark_runner
cmake --build build --target test_runner
```

This creates the executables in the ```build/``` directory (e.g., ```build/benchmark_runner```).

## 4. How to Use The Executables
There are two main executables: one for **correctness** and one for **performance**.

### Correctness Testing with test_runner
**Purpose:** Compares your selected backend (SOLVER_BACKEND) against the cuOpt base_solver to ensure the results are correct.

```bash
# 1. Run the tester on a directory (e.g., netlib)
./build/test_runner ./netlib/

# 2. Run on the default 'test/' directory
./build/test_runner
```

**Example Output:**
```
Comparing on: ./netlib/afiro.presolved (m=35, n=32)
  Backend: Success (4.12 ms)
  cuOpt:   Success (1.89 ms)
  Scores -> Backend: -0.465153 | cuOpt: -0.465153 | Delta: 0.000000
```

### Performance Benchmarking with ```benchmark_runner```
**Purpose:** Measures the execution speed of *only* your selected backend (```SOLVER_BACKEND```) using warm-up runs and statistical analysis. This is the main tool for profiling.

```bash
# 1. Run on a whole directory
./build/benchmark_runner ./netlib/

# 2. Run on a single problem file
./build/benchmark_runner ./netlib/afiro.presolved
```

**Example Output:**
```
Benchmarking: ./netlib/afiro.presolved (m=35, n=32)
  Runs: 10 (after 5 warm-ups)
  Avg:  4.081 ms
  Min:  4.052 ms (Fastest)
  Max:  4.119 ms
```

# 5. Performance Profiling Workflow
This is the primary goal of the project. Here is the step-by-step workflow to find and fix bottlenecks.

## Step 1. Build the GPU solver
Configure with the ```gpu``` backend and build the ```benchmark_runner```.

```bash
cmake -S . -B build -DSOLVER_BACKEND=gpu
cmake --build build --target benchmark_runner
```

## Step 2. Run the NSight Systems Profiler (```nsys```)
```nsys``` gives you a high-level timeline of your entire application (CPU threads, CUDA calls, GPU kernels). This is the best place to start.
Run your ```benchmark_runner``` under nsys on a single, representative problem.

```bash
nsys profile -o gpu_report ./build/benchmark_runner ./netlib/afiro.presolved
```

This will run the benchmark and create a ```gpu_report.nsys-rep``` file.

## Step 3. Analyze in NSight Systems GUI
Open the ```gpu_report.nsys-rep``` file in the **NVIDIA NSight Systems GUI**. 
Look for:
- **Long Gaps:** Is the CPU waiting for the GPU? Is the GPU waiting for the CPU?
- **Memory Transfers:** Are cudaMemcpy calls taking a long time?
- **Slow Kernels:** Which GPU kernels are taking the most time?

## Step 4. (Optional) Deep-Dive with NSight Compute (```ncu```)
If ```nsys``` shows that a specific kernel (e.g., ```vec_sub_kernel```) is your bottleneck, use ```ncu``` to analyze *just that kernel* in extreme detail.

```bash
ncu --set full -o compute_report ./build/benchmark_runner ./netlib/afiro.presolved
```

Open the ```compute_report.ncu-rep``` file in the NVIDIA NSight Compute GUI to see details like memory stalls, instruction counts, and cache miss rates for your kernel.