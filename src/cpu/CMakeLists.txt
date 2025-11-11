add_library(simplex_cpu STATIC
    linalg_cpu.cu
    linalg_cpu.hpp
    solver_cpu.cpp
)

target_include_directories(simplex_cpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(simplex_cpu
    PUBLIC
        simplex_common
        CUDA::cusolver
        CUDA::cudart
)