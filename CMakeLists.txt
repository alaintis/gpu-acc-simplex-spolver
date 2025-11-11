cmake_minimum_required(VERSION 3.20)

project(SimplexSolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 12.0 REQUIRED
    COMPONENTS
        cusolver
        cublas
        cudart
        cusparse
)

# This can be overridden (e.g., -DcuOpt_DIR=...)
set(cuOpt_DIR "$ENV{HOME}/.venv/lib/python3.12/site-packages/libcuopt"
    CACHE PATH "Path to the root of the cuOpt installation")

message(STATUS "Using cuOpt from: ${cuOpt_DIR}")
if(NOT EXISTS "${cuOpt_DIR}/include/cuopt/linear_programming/cuopt_c.h")
    message(FATAL_ERROR "cuOpt not found at the specified path. "
                        "Please set -DcuOpt_DIR to the correct location.")
endif()

# Backend Selection
set(SOLVER_BACKEND "gpu" CACHE STRING "Which solver backend to use (cpu, gpu, cuopt)")
set_property(CACHE SOLVER_BACKEND PROPERTY STRINGS "cpu" "gpu" "cuopt")
message(STATUS "Selected solver backend: ${SOLVER_BACKEND}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
