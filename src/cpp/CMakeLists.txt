# C++ source directory CMakeLists.txt

# Core library - common utilities and base classes
add_library(quant_core INTERFACE)
target_include_directories(quant_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
if(TARGET Eigen3::Eigen)
    target_link_libraries(quant_core INTERFACE Eigen3::Eigen)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(quant_core INTERFACE OpenMP::OpenMP_CXX)
endif()

# Models library - Heston, SABR, OU implementations
# Will be populated in later tasks

# Solvers library - PDE solvers, FFT, numerical methods
# Will be populated in later tasks

# Bindings - pybind11 Python bindings
# Will be populated when models are implemented

# Placeholder: Create a simple test library to verify build works
add_library(quant_math STATIC
    core/math_utils.cpp
)
target_include_directories(quant_math PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
if(TARGET Eigen3::Eigen)
    target_link_libraries(quant_math PUBLIC Eigen3::Eigen)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(quant_math PUBLIC OpenMP::OpenMP_CXX)
endif()
