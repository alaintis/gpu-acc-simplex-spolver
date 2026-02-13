add_library(simplex_gpu_v2 STATIC
    linalg_gpu.cu
    solver_gpu.cpp
)

set_target_properties(simplex_gpu_v0 PROPERTIES
    CUDA_ARCHITECTURES native   # e.g. 75, 80, 86, 89, 90
)



target_include_directories(simplex_gpu_v2
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        /opt/cuda/include
)


target_compile_options(simplex_gpu_v2 PRIVATE

    # -------- Normal C++ (.cpp) --------
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>

    # -------- CUDA Device Code --------
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>

    # -------- CUDA Host Code (inside .cu) --------
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native>

    # -------- Debug Handling --------
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -lineinfo>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-O0,-g>

    # -------- Warnings --------
    -Wall -Wextra

)


target_link_libraries(simplex_gpu_v2
    PUBLIC
        simplex_common
        CUDA::cusolver
        CUDA::cublas
        CUDA::cudart
)

