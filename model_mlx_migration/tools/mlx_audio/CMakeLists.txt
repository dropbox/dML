cmake_minimum_required(VERSION 3.18)
project(mlx_audio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Python
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Find FFmpeg libraries via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(SWRESAMPLE REQUIRED libswresample)
pkg_check_modules(AVUTIL REQUIRED libavutil)

# nanobind (added as submodule in extern/)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/nanobind/CMakeLists.txt")
    add_subdirectory(extern/nanobind)
else()
    message(FATAL_ERROR "nanobind not found. Run: git submodule add https://github.com/wjakob/nanobind extern/nanobind")
endif()

# Build the Python extension module
nanobind_add_module(mlx_audio_native
    src/audio_loader.cpp
    src/bindings.cpp
)

target_include_directories(mlx_audio_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${SWRESAMPLE_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
)

target_link_directories(mlx_audio_native PRIVATE
    ${AVFORMAT_LIBRARY_DIRS}
    ${AVCODEC_LIBRARY_DIRS}
    ${SWRESAMPLE_LIBRARY_DIRS}
    ${AVUTIL_LIBRARY_DIRS}
)

target_link_libraries(mlx_audio_native PRIVATE
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${SWRESAMPLE_LIBRARIES}
    ${AVUTIL_LIBRARIES}
)

# Compiler flags
target_compile_options(mlx_audio_native PRIVATE
    -Wall
    -Wextra
    -O3
    -ffast-math
)

# Install the module to the python package directory
install(TARGETS mlx_audio_native
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python
)

# Output where the module will be built
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "FFmpeg libavformat: ${AVFORMAT_VERSION}")
message(STATUS "FFmpeg libavcodec: ${AVCODEC_VERSION}")
