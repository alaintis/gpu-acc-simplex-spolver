add_library(simplex_common STATIC
    logging.cpp
    linprog.cpp
    solver_wrapper.cpp
    problem_reader.cpp
)

target_include_directories(simplex_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(simplex_common PUBLIC "SOLVER_BACKEND_NAME=\"${SOLVER_BACKEND}\"")