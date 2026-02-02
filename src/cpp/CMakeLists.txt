# C++ source directory CMakeLists.txt

# Core library - common utilities and base classes
add_library(quant_core INTERFACE)
target_include_directories(quant_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if(TARGET Eigen3::Eigen)
    target_link_libraries(quant_core INTERFACE Eigen3::Eigen)
    target_compile_definitions(quant_core INTERFACE QUANT_USE_EIGEN)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(quant_core INTERFACE OpenMP::OpenMP_CXX)
endif()

# Math utilities library
add_library(quant_math STATIC
    core/math_utils.cpp
)
target_include_directories(quant_math PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
target_link_libraries(quant_math PUBLIC quant_core)

# Models library - Heston, SABR, OU implementations
add_library(quant_models STATIC
    models/heston.cpp
    models/sabr.cpp
    models/ou_process.cpp
)
target_include_directories(quant_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(quant_models PUBLIC quant_core quant_math)

# PDE Solvers library - Black-Scholes, Heston, HJB solvers
add_library(quant_solvers INTERFACE)
target_include_directories(quant_solvers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(quant_solvers INTERFACE quant_core quant_models)

# Combined library for easy linking
add_library(quant_trading_cpp INTERFACE)
target_link_libraries(quant_trading_cpp INTERFACE
    quant_core
    quant_math
    quant_models
    quant_solvers
)

# Export include directories for bindings
set(QUANT_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers
    PARENT_SCOPE
)

# Python bindings (if pybind11 is available)
if(pybind11_FOUND)
    message(STATUS "Building Python bindings with pybind11")

    pybind11_add_module(quant_cpp
        bindings/quant_cpp.cpp
        bindings/heston_bindings.cpp
        bindings/sabr_bindings.cpp
        bindings/ou_bindings.cpp
        bindings/pde_bindings.cpp
    )

    target_link_libraries(quant_cpp PRIVATE
        quant_models
        quant_math
        quant_core
        quant_solvers
    )

    target_include_directories(quant_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_CURRENT_SOURCE_DIR}/solvers
    )

    # Set output directory for the Python module
    set_target_properties(quant_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python/quant_trading/cpp"
    )

    # Install target for pip install
    install(TARGETS quant_cpp
        LIBRARY DESTINATION quant_trading/cpp
    )
else()
    message(STATUS "pybind11 not found - Python bindings will not be built")
endif()
