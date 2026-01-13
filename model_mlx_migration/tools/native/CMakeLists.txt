cmake_minimum_required(VERSION 3.20)
project(sota_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MLX (installed via Homebrew)
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
find_package(MLX REQUIRED)

# Find pybind11 for Python bindings
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

# Main C++ library (ECAPA-TDNN + AST)
add_library(sota_native_core SHARED
    src/ecapa_inference.cpp
    src/ast_inference.cpp
)

target_include_directories(sota_native_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sota_native_core
    mlx
)

# Benchmark executable
add_executable(benchmark_sota_native
    src/benchmark.cpp
)

target_link_libraries(benchmark_sota_native
    sota_native_core
    mlx
)

# Python bindings (if pybind11 found)
if(pybind11_FOUND)
    pybind11_add_module(sota_native
        src/python_bindings.cpp
    )

    target_link_libraries(sota_native PRIVATE
        sota_native_core
        mlx
    )

    message(STATUS "pybind11 found - Python bindings will be built")
else()
    message(WARNING "pybind11 not found - Python bindings will NOT be built")
    message(STATUS "Install with: pip install pybind11")
endif()

# Print configuration
message(STATUS "MLX found: ${MLX_FOUND}")
message(STATUS "MLX include dirs: ${MLX_INCLUDE_DIRS}")
message(STATUS "Python found: ${Python3_FOUND}")
message(STATUS "Python version: ${Python3_VERSION}")
