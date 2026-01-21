# C++ Tests CMakeLists.txt

include(GoogleTest)

# Test math utilities
add_executable(test_math_utils test_math_utils.cpp)
target_link_libraries(test_math_utils PRIVATE
    quant_math
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(test_math_utils)
