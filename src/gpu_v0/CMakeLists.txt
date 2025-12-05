add_library(simplex_gpu_v0 STATIC
    linalg_gpu.cu
    solver_gpu.cpp
    base_solver_wrapped.cpp
)

target_include_directories(simplex_gpu_v0
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(simplex_gpu_v0
    PUBLIC
        simplex_common
        CUDA::cusolver
        CUDA::cublas
        CUDA::cudart
)
