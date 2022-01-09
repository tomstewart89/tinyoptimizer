cmake_minimum_required(VERSION 3.14)
project(tiny_sqp_tests)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/..")

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix gtest_main)

include(GoogleTest)

gtest_discover_tests(test_matrix)
