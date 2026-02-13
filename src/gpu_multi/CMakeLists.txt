add_library(gpu_multi STATIC
    linalg_gpu.cu
    scale.cu
    workspace.cpp
    solver_gpu.cpp
)

target_include_directories(gpu_multi
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        /opt/cuda/include
)

target_link_libraries(gpu_multi
    PUBLIC
        simplex_common
        CUDA::cusolver
        CUDA::cublas
        CUDA::cudart
)

target_compile_options(gpu_multi PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    # CPU optimization
    #$<$<COMPILE_LANGUAGE:CXX>:-march=native>
)
