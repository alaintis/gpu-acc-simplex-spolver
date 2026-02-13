if(SOLVER_BACKEND STREQUAL "cpu")
    set(SOLVER_LIB simplex_cpu)
elseif(SOLVER_BACKEND STREQUAL "gpuv0")
    set(SOLVER_LIB simplex_gpu_v0)
elseif(SOLVER_BACKEND STREQUAL "gpuv1")
    set(SOLVER_LIB simplex_gpu_v1)
elseif(SOLVER_BACKEND STREQUAL "gpu_multi")
    set(SOLVER_LIB gpu_multi)
elseif(SOLVER_BACKEND STREQUAL "cuopt")
    set(SOLVER_LIB simplex_cuopt)
elseif(SOLVER_BACKEND STREQUAL "sparse")
    set(SOLVER_LIB simplex_sparse)
else()
    message(FATAL_ERROR "Unknown SOLVER_BACKEND: ${SOLVER_BACKEND}")
endif()

message(STATUS "Tools will link against: ${SOLVER_LIB}")

add_executable(solver
    main.cpp
)

target_link_libraries(solver PRIVATE
    simplex_common
    ${SOLVER_LIB}
)

add_executable(testing_execution
    testing.cpp
)

target_link_libraries(testing_execution PRIVATE
    simplex_common
    ${SOLVER_LIB}

)

add_executable(test_runner
    testing.cpp
)

target_link_libraries(test_runner PRIVATE
    simplex_common
    ${SOLVER_LIB}
    simplex_gpu_v0
)

add_executable(benchmark_runner
    benchmarking.cpp
)

target_link_libraries(benchmark_runner PRIVATE
    simplex_common
    ${SOLVER_LIB}
)

add_executable(compare_tool
    compare.cpp
)

target_link_libraries(compare_tool
    PRIVATE
        simplex_common
        ${SOLVER_LIB}
        simplex_gpu_v0
)
