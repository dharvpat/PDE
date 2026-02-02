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

# Test Heston model
add_executable(test_heston test_heston.cpp)
target_link_libraries(test_heston PRIVATE
    quant_models
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(test_heston)

# Test SABR model
add_executable(test_sabr test_sabr.cpp)
target_link_libraries(test_sabr PRIVATE
    quant_models
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(test_sabr)

# Test OU process
add_executable(test_ou_process test_ou_process.cpp)
target_link_libraries(test_ou_process PRIVATE
    quant_models
    GTest::gtest
    GTest::gtest_main
)
gtest_discover_tests(test_ou_process)
