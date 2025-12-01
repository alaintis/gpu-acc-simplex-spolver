add_library(simplex_sparse STATIC
    linalg_sparse.cu
    solver_sparse.cpp
)

target_include_directories(simplex_sparse
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${cuDSS_DIR}/include
)

target_link_libraries(simplex_sparse
    PUBLIC
        simplex_common
        CUDA::cusolver
        CUDA::cublas
        CUDA::cudart
)


target_link_libraries(simplex_sparse
    PUBLIC
        ${cuDSS_DIR}/lib/libcudss.so.0
)
