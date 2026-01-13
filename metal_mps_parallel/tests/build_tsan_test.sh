#!/bin/bash
# Build TSan test for MPS parallel inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PYTORCH_DIR="$ROOT_DIR/pytorch-mps-fork"
BUILD_DIR="$PYTORCH_DIR/build"

echo "=== Building TSan MPS Test ==="
echo "PyTorch dir: $PYTORCH_DIR"
echo "Build dir: $BUILD_DIR"

# Check for required files
if [ ! -f "$BUILD_DIR/lib/libtorch_cpu.dylib" ]; then
    echo "ERROR: libtorch_cpu.dylib not found. Build PyTorch first."
    exit 1
fi

# Include paths
INCLUDES=(
    "-I$PYTORCH_DIR/torch/include"
    "-I$PYTORCH_DIR/torch/csrc/api/include"
    "-I$PYTORCH_DIR"
    "-I$BUILD_DIR/aten/src"
    "-I$BUILD_DIR"
    "-I$PYTORCH_DIR/c10"
    "-I$PYTORCH_DIR/aten/src"
)

# Library paths
LIBS=(
    "-L$BUILD_DIR/lib"
    "-Wl,-rpath,$BUILD_DIR/lib"
    "-ltorch_cpu"
    "-lc10"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
)

# Compiler flags
CXXFLAGS=(
    "-std=c++17"
    "-fsanitize=thread"
    "-g"
    "-fno-omit-frame-pointer"
    "-O1"  # Some optimization for reasonable perf
    "-DUSE_MPS"
)

echo "Compiling with TSan..."
# Note: Framework flags must come after source file
clang++ -std=c++17 -fsanitize=thread -g -fno-omit-frame-pointer -O1 -DUSE_MPS \
    -I"$PYTORCH_DIR/torch/include" \
    -L"$BUILD_DIR/lib" \
    -Wl,-rpath,"$BUILD_DIR/lib" \
    "$SCRIPT_DIR/tsan_mps_test.mm" \
    -ltorch_cpu -lc10 \
    -framework Foundation \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -o "$SCRIPT_DIR/tsan_mps_test" 2>&1

if [ $? -eq 0 ]; then
    echo "Build successful: $SCRIPT_DIR/tsan_mps_test"
    echo ""
    echo "Usage: ./tsan_mps_test [OPTIONS] [num_threads] [iterations]"
    echo "  Options: -t/--threads=N, -i/--iterations=N, -h/--help"
    echo "  Default: 8 threads, 50 iterations"
    echo "  Examples:"
    echo "    ./tsan_mps_test 31 100"
    echo "    ./tsan_mps_test --threads=31 --iterations=100"
    echo "    ./tsan_mps_test -t 8 -i 50"
    echo ""
    echo "To run with TSan:"
    echo "  TSAN_OPTIONS='suppressions=$SCRIPT_DIR/tsan_suppressions.txt:halt_on_error=0' $SCRIPT_DIR/tsan_mps_test"
else
    echo "Build failed"
    exit 1
fi
