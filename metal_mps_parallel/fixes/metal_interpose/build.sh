#!/bin/bash
# Build the Metal interposition library
#
# Usage:
#   ./build.sh
#
# The library will be built as libmetal_interpose.dylib

set -e

cd "$(dirname "$0")"

echo "Building Metal interposition library..."

clang -dynamiclib \
    -o libmetal_interpose.dylib \
    metal_interpose.m \
    -framework Foundation \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -fobjc-arc \
    -O2

echo "Built: libmetal_interpose.dylib"
echo ""
echo "Usage:"
echo "  DYLD_INSERT_LIBRARIES=$(pwd)/libmetal_interpose.dylib \\"
echo "      METAL_INTERPOSE_LOG=1 \\"
echo "      python3 tests/mps_sync_comparison.py"
