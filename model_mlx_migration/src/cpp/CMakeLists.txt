# Copyright 2024-2025 Andrew Yates
# Zipformer C++ MLX Inference Engine
#
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.18)
project(zipformer_mlx VERSION 1.0.0 LANGUAGES CXX)

# Require C++17 for MLX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MLX (installed via homebrew)
find_library(MLX_LIBRARY mlx PATHS /opt/homebrew/lib REQUIRED)
find_path(MLX_INCLUDE_DIR mlx/mlx.h PATHS /opt/homebrew/include REQUIRED)

message(STATUS "MLX library: ${MLX_LIBRARY}")
message(STATUS "MLX include: ${MLX_INCLUDE_DIR}")

# Metal framework (required for MLX on macOS)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# Source files
set(ZIPFORMER_SOURCES
    src/array_utils.cpp
    src/scaling.cpp
    src/encoder.cpp
    src/zipformer2.cpp
    src/decoder.cpp
    src/joiner.cpp
    src/features.cpp
    src/zipformer.cpp
    src/inference.cpp
)

# Header files
set(ZIPFORMER_HEADERS
    include/zipformer/array_utils.hpp
    include/zipformer/scaling.hpp
    include/zipformer/encoder.hpp
    include/zipformer/zipformer2.hpp
    include/zipformer/decoder.hpp
    include/zipformer/joiner.hpp
    include/zipformer/features.hpp
    include/zipformer/zipformer.hpp
    include/zipformer/inference.hpp
)

# Main library
add_library(zipformer_mlx STATIC ${ZIPFORMER_SOURCES})

target_include_directories(zipformer_mlx
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MLX_INCLUDE_DIR}
)

target_link_libraries(zipformer_mlx
    PUBLIC
        ${MLX_LIBRARY}
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
)

# Compiler warnings
target_compile_options(zipformer_mlx PRIVATE
    -Wall -Wextra -Wpedantic -Wno-unused-parameter
)

# Optimization for release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(zipformer_mlx PRIVATE -O3 -march=native)
endif()

# Test executable
add_executable(test_zipformer tests/test_zipformer.cpp)
target_link_libraries(test_zipformer PRIVATE zipformer_mlx)

# Inference CLI executable
add_executable(zipformer_infer src/main.cpp)
target_link_libraries(zipformer_infer PRIVATE zipformer_mlx)

# Install
install(TARGETS zipformer_mlx zipformer_infer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
