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
        nvToolsExt
)

# This can be overridden (e.g., -DcuOpt_DIR=...)
set(cuOpt_DIR "$ENV{HOME}/.venv/lib/python3.12/site-packages/libcuopt"
    CACHE PATH "Path to the root of the cuOpt installation")

message(STATUS "Using cuOpt from: ${cuOpt_DIR}")
if(NOT EXISTS "${cuOpt_DIR}/include/cuopt/linear_programming/cuopt_c.h")
    message(FATAL_ERROR "cuOpt not found at the specified path. "
                        "Please set -DcuOpt_DIR to the correct location.")
endif()

# This can be overridden (e.g., -DcuDSS_DIR=...)
set(cuDSS_DIR "$ENV{HOME}/dphpc/venv/lib/python3.12/site-packages/nvidia/cu12"
    CACHE PATH "Path to the root of the cuDSS installation")

message(STATUS "Using cuDSS from: ${cuDSS_DIR}")
if(NOT EXISTS "${cuDSS_DIR}/include/cudss.h")
    message(FATAL_ERROR "cuDSS not found at the specified path ${cuDSS_DIR}. "
                        "Please set -DcuDSS_DIR to the correct location.")
endif()

# Backend Selection
set(SOLVER_BACKEND "gpuv0" CACHE STRING "Which solver backend to use (cpu, gpuv0, gpuv1 cuopt, sparse, gpu_multi)")
set_property(CACHE SOLVER_BACKEND PROPERTY STRINGS "cpu" "gpuv0" "gpuv1" "cuopt" "sparse" "gpu_multi")
message(STATUS "Selected solver backend: ${SOLVER_BACKEND}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
