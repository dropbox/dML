#!/bin/bash
# Build recordStream test for MPS parallel inference
# Tests External Audit Gap #4: record-stream semantics coverage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PYTORCH_DIR="$ROOT_DIR/pytorch-mps-fork"
BUILD_DIR="$PYTORCH_DIR/build"

echo "=== Building recordStream Test ==="
echo "PyTorch dir: $PYTORCH_DIR"
echo "Build dir: $BUILD_DIR"

# Check for required files
if [ ! -f "$BUILD_DIR/lib/libtorch_cpu.dylib" ]; then
    echo "ERROR: libtorch_cpu.dylib not found. Build PyTorch first."
    exit 1
fi

# Parse arguments for TSan option
USE_TSAN=0
for arg in "$@"; do
    if [ "$arg" = "--tsan" ] || [ "$arg" = "-t" ]; then
        USE_TSAN=1
    fi
done

TSAN_FLAGS=""
OUTPUT_NAME="record_stream_test"
if [ "$USE_TSAN" -eq 1 ]; then
    TSAN_FLAGS="-fsanitize=thread"
    OUTPUT_NAME="record_stream_test_tsan"
    echo "Building with ThreadSanitizer..."
else
    echo "Building without TSan (use --tsan to enable)..."
fi

echo "Compiling..."
clang++ -std=c++17 $TSAN_FLAGS -g -fno-omit-frame-pointer -O1 -DUSE_MPS \
    -I"$PYTORCH_DIR/torch/include" \
    -L"$BUILD_DIR/lib" \
    -Wl,-rpath,"$BUILD_DIR/lib" \
    "$SCRIPT_DIR/test_record_stream.mm" \
    -ltorch_cpu -lc10 \
    -framework Foundation \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -o "$SCRIPT_DIR/$OUTPUT_NAME" 2>&1

if [ $? -eq 0 ]; then
    echo "Build successful: $SCRIPT_DIR/$OUTPUT_NAME"
    echo ""
    echo "Run: ./$OUTPUT_NAME"
    if [ "$USE_TSAN" -eq 1 ]; then
        echo "TSan: TSAN_OPTIONS='suppressions=$SCRIPT_DIR/tsan_suppressions.txt:halt_on_error=0' ./$OUTPUT_NAME"
    fi
else
    echo "Build failed"
    exit 1
fi
