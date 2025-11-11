add_library(simplex_cuopt STATIC
    solver_cuopt.cpp
)

target_include_directories(simplex_cuopt
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${cuOpt_DIR}/include
)

target_link_libraries(simplex_cuopt
    PUBLIC
        simplex_common
        cuopt
        CUDA::cusparse
        CUDA::cudart
)

target_link_directories(simplex_cuopt
    PUBLIC
        ${cuOpt_DIR}/lib64
)