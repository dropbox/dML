#!/bin/bash
# Build C++ test executables for MPS parallel testing
#
# This script compiles standalone C++ tests against the local pytorch-mps-fork build.
#
# Usage:
#   ./tests/build_cpp_tests.sh [test_name]
#
# Examples:
#   ./tests/build_cpp_tests.sh                    # Build all tests
#   ./tests/build_cpp_tests.sh minimal_mps_contiguous_race  # Build specific test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FORK_DIR="${REPO_ROOT}/pytorch-mps-fork"
BUILD_DIR="${REPO_ROOT}/tests/build"

# Create build directory
mkdir -p "${BUILD_DIR}"

# Get PyTorch include and library paths from the local build
PYTORCH_BUILD="${FORK_DIR}/build"
PYTORCH_INCLUDE="${FORK_DIR}"
C10_INCLUDE="${FORK_DIR}/c10"
ATEN_INCLUDE="${FORK_DIR}/aten/src"
BUILD_INCLUDE="${PYTORCH_BUILD}/aten/src"

# Check required directories exist
if [ ! -d "${PYTORCH_BUILD}" ]; then
    echo "ERROR: PyTorch build directory not found: ${PYTORCH_BUILD}"
    echo "Run 'python setup.py develop' in pytorch-mps-fork first"
    exit 1
fi

# Compiler flags
CXX="clang++"
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra"
CXXFLAGS="${CXXFLAGS} -fPIC"
CXXFLAGS="${CXXFLAGS} -DUSE_MPS -DATEN_MPS_ENABLED"

# Include paths
INCLUDES="-I${PYTORCH_INCLUDE}"
INCLUDES="${INCLUDES} -I${PYTORCH_BUILD}"
INCLUDES="${INCLUDES} -I${ATEN_INCLUDE}"
INCLUDES="${INCLUDES} -I${BUILD_INCLUDE}"
INCLUDES="${INCLUDES} -I${C10_INCLUDE}"
INCLUDES="${INCLUDES} -I${PYTORCH_INCLUDE}/torch/csrc/api/include"

# Library paths
LDFLAGS="-L${PYTORCH_BUILD}/lib"
LDFLAGS="${LDFLAGS} -Wl,-rpath,${PYTORCH_BUILD}/lib"

# Libraries to link
LIBS="-ltorch -ltorch_cpu -lc10"
LIBS="${LIBS} -framework Foundation -framework Metal -framework MetalPerformanceShaders"

# Build function
build_test() {
    local test_name="$1"
    local source_file="${SCRIPT_DIR}/${test_name}.mm"
    local output_file="${BUILD_DIR}/${test_name}"

    if [ ! -f "${source_file}" ]; then
        echo "ERROR: Source file not found: ${source_file}"
        return 1
    fi

    echo "Building ${test_name}..."
    echo "  Source: ${source_file}"
    echo "  Output: ${output_file}"

    # Compile with Objective-C++ for .mm files
    ${CXX} ${CXXFLAGS} \
        ${INCLUDES} \
        -fobjc-arc \
        -x objective-c++ \
        "${source_file}" \
        ${LDFLAGS} \
        ${LIBS} \
        -o "${output_file}"

    echo "  Done: ${output_file}"
    return 0
}

# Main logic
if [ $# -gt 0 ]; then
    # Build specific test
    build_test "$1"
else
    # Build all .mm test files
    echo "Building all C++ tests..."
    for source in "${SCRIPT_DIR}"/*.mm; do
        if [ -f "${source}" ]; then
            test_name=$(basename "${source}" .mm)
            build_test "${test_name}" || true
        fi
    done
fi

echo ""
echo "Build complete. Test executables are in: ${BUILD_DIR}/"
echo "To run:"
echo "  ${BUILD_DIR}/minimal_mps_contiguous_race"
