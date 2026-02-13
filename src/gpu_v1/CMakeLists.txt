add_library(simplex_gpu_v1 STATIC
    linalg_gpu.cu
    solver_gpu.cpp
)

target_include_directories(simplex_gpu_v1
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        /opt/cuda/include
)

target_compile_options(simplex_gpu_v1 PRIVATE)

target_link_libraries(simplex_gpu_v1
    PUBLIC
        simplex_common
        CUDA::cusolver
        CUDA::cublas
        CUDA::cudart
)

target_compile_options(simplex_gpu_v1 PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)