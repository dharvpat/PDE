cmake_minimum_required(VERSION 3.18)
project(quant_trading VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(ENABLE_LTO "Enable link-time optimization" ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic)

    # Build type specific flags
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

    # Enable native architecture optimization in Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            add_compile_options(-march=native)
        endif()
    endif()
endif()

# Link-time optimization
if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
    if(lto_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO enabled")
    else()
        message(STATUS "LTO not supported: ${lto_error}")
    endif()
endif()

# OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(STATUS "OpenMP not found - parallel features disabled")
    endif()
endif()

# Find Eigen
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    # Try to find Eigen in common locations
    find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core
        PATHS
        /usr/include/eigen3
        /usr/local/include/eigen3
        /opt/homebrew/include/eigen3
        ${CMAKE_SOURCE_DIR}/third_party/eigen
    )
    if(EIGEN3_INCLUDE_DIR)
        message(STATUS "Eigen3 found at: ${EIGEN3_INCLUDE_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    else()
        message(STATUS "Eigen3 not found - will be installed via third_party or package manager")
    endif()
else()
    message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
endif()

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11 found: ${pybind11_VERSION}")
else()
    message(STATUS "pybind11 not found via CMake - will use pip-installed version")
endif()

# Add source directories
add_subdirectory(src/cpp)

# Testing
if(BUILD_TESTS)
    enable_testing()

    # Try to find Google Test
    find_package(GTest QUIET)
    if(GTest_FOUND)
        message(STATUS "Google Test found")
    else()
        # Check if we have it in third_party
        if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
            add_subdirectory(third_party/googletest)
            message(STATUS "Using bundled Google Test")
        else()
            message(STATUS "Google Test not found - C++ tests will not be built")
        endif()
    endif()

    # Add test directory if GTest is available
    if(GTest_FOUND OR TARGET gtest)
        add_subdirectory(tests/cpp)
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP: ${ENABLE_OPENMP}")
message(STATUS "LTO: ${ENABLE_LTO}")
message(STATUS "Tests: ${BUILD_TESTS}")
message(STATUS "=============================")
