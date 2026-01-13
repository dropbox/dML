#!/bin/bash
# rebuild_tts.sh - Rebuild stream-tts-cpp with current venv's PyTorch
# Worker #486: Created to handle libtorch version mismatches
#
# Usage:
#   ./scripts/rebuild_tts.sh           # Full rebuild
#   ./scripts/rebuild_tts.sh --fix     # Fix symlinks only (if already built)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
TTS_DIR="$PROJECT_ROOT/stream-tts-cpp"
VENV_TORCH="$PROJECT_ROOT/.venv/lib/python3.11/site-packages/torch"

echo "=========================================="
echo "  Stream-TTS-CPP Rebuild Script"
echo "=========================================="
echo ""

# Check venv torch exists
if [ ! -d "$VENV_TORCH" ]; then
    echo "❌ PyTorch not found in venv at: $VENV_TORCH"
    echo "Install with: pip install torch"
    exit 1
fi

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "📦 PyTorch version in venv: $TORCH_VERSION"

# Fix symlinks
echo ""
echo "🔗 Updating libtorch-mps symlinks..."
cd "$TTS_DIR/external"
rm -rf libtorch-mps
mkdir libtorch-mps
cd libtorch-mps
ln -s "$VENV_TORCH/include" include
ln -s "$VENV_TORCH/lib" lib
ln -s "$VENV_TORCH/share" share
echo "   ✅ Symlinks updated to: $VENV_TORCH"

if [ "$1" = "--fix" ]; then
    echo ""
    echo "✅ Symlinks fixed. Run without --fix to do full rebuild."
    exit 0
fi

# Full rebuild
echo ""
echo "🔨 Rebuilding stream-tts-cpp..."
cd "$TTS_DIR"
rm -rf build
mkdir build
cd build

echo ""
echo "⚙️  Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_LIBTORCH=ON \
    -DCMAKE_PREFIX_PATH="$TTS_DIR/external/libtorch-mps" \
    2>&1 | grep -E "✅|⚠️|❌|Found|Torch" || true

echo ""
echo "🔨 Building (this may take a few minutes)..."
cmake --build . -j8 --target stream-tts-cpp 2>&1 | tail -5

# Verify binary runs
echo ""
if ! "$TTS_DIR/build/stream-tts-cpp" --version > /dev/null 2>&1; then
    echo "❌ Build failed - binary doesn't run"
    exit 1
fi

echo "✅ Binary runs!"
"$TTS_DIR/build/stream-tts-cpp" --version

# Quick smoke test - synthesize a short phrase
echo ""
echo "🔊 Running smoke test (English TTS)..."
TEST_OUTPUT="/tmp/tts_smoke_test.wav"
rm -f "$TEST_OUTPUT"

START_TIME=$(python3 -c "import time; print(time.time())")

if "$TTS_DIR/build/stream-tts-cpp" --speak "Test" --lang en --save-audio "$TEST_OUTPUT" 2>/dev/null; then
    END_TIME=$(python3 -c "import time; print(time.time())")
    DURATION=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")

    if [ -f "$TEST_OUTPUT" ] && [ -s "$TEST_OUTPUT" ]; then
        FILE_SIZE=$(ls -la "$TEST_OUTPUT" | awk '{print $5}')
        echo "✅ Smoke test passed!"
        echo "   Generated: $TEST_OUTPUT ($FILE_SIZE bytes)"
        echo "   Time: ${DURATION}s"
        rm -f "$TEST_OUTPUT"
    else
        echo "❌ Smoke test failed - no audio generated"
        exit 1
    fi
else
    echo "❌ Smoke test failed - TTS synthesis error"
    exit 1
fi

echo ""
echo "=========================================="
echo "  ✅ All checks passed!"
echo "=========================================="
