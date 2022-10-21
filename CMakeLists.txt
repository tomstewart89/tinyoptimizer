cmake_minimum_required(VERSION 3.14)
project(tinyoptimizer)

set(CMAKE_CXX_STANDARD 17)

include_directories("${PROJECT_SOURCE_DIR}")

add_executable(tinyoptimizer tinyoptimizer.cpp)
