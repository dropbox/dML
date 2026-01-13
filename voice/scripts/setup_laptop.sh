#!/bin/bash
# Setup script for deploying Voice TTS to a new laptop
# Usage: ./scripts/setup_laptop.sh [--models-from <host>]
#
# This script:
# 1. Checks prerequisites (cmake, python3)
# 2. Initializes git submodules
# 3. Builds the C++ binary
# 4. Sets up Python venv for tests
# 5. Optionally syncs models from another machine

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Parse arguments
MODELS_SOURCE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --models-from)
            MODELS_SOURCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--models-from <user@host>]"
            echo ""
            echo "Options:"
            echo "  --models-from <user@host>  Sync models from another machine via rsync"
            echo ""
            echo "Example:"
            echo "  $0 --models-from ayates@main-macbook.local"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

info "Setting up Voice TTS on $(hostname)"
echo ""

# Check prerequisites
info "Checking prerequisites..."

if ! command -v cmake &> /dev/null; then
    error "cmake not found. Install with: brew install cmake"
fi
info "  cmake: $(cmake --version | head -1)"

if ! command -v python3 &> /dev/null; then
    error "python3 not found. Install with: brew install python@3.11"
fi
info "  python3: $(python3 --version)"

if ! command -v git &> /dev/null; then
    error "git not found. Install with: brew install git"
fi
info "  git: $(git --version)"

# Check for Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    warn "Not running on Apple Silicon - MPS acceleration won't be available"
fi

echo ""

# Initialize submodules
info "Initializing git submodules..."
git submodule update --init --recursive
info "  Submodules initialized"

echo ""

# Build C++ binary
info "Building C++ binary (Release mode)..."
BUILD_DIR="$PROJECT_ROOT/stream-tts-cpp/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)

if [[ -f "$BUILD_DIR/stream-tts-cpp" ]]; then
    info "  Binary built: $BUILD_DIR/stream-tts-cpp"
else
    error "Build failed - binary not found"
fi

cd "$PROJECT_ROOT"
echo ""

# Setup Python venv
info "Setting up Python virtual environment..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    info "  Created .venv"
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q 2>/dev/null || pip install pytest pyyaml numpy scipy -q
info "  Python dependencies installed"

echo ""

# Sync models if source provided
if [[ -n "$MODELS_SOURCE" ]]; then
    info "Syncing models from $MODELS_SOURCE..."
    mkdir -p models
    rsync -avP --compress "$MODELS_SOURCE:~/voice/models/" models/
    info "  Models synced"
    echo ""
fi

# Check models
info "Checking models..."
MODELS_OK=true

if [[ -f "models/kokoro/kokoro_mps.pt" ]]; then
    SIZE=$(stat -f%z "models/kokoro/kokoro_mps.pt" 2>/dev/null || stat -c%s "models/kokoro/kokoro_mps.pt" 2>/dev/null)
    info "  Kokoro TTS: $(echo "scale=1; $SIZE/1024/1024/1024" | bc)GB"
else
    warn "  Kokoro TTS: NOT FOUND - models/kokoro/kokoro_mps.pt"
    MODELS_OK=false
fi

if [[ -f "models/nllb/nllb-encoder-mps.pt" ]]; then
    SIZE=$(stat -f%z "models/nllb/nllb-encoder-mps.pt" 2>/dev/null || stat -c%s "models/nllb/nllb-encoder-mps.pt" 2>/dev/null)
    info "  NLLB Translation: $(echo "scale=1; $SIZE/1024/1024/1024" | bc)GB"
else
    warn "  NLLB Translation: NOT FOUND (optional)"
fi

if [[ "$MODELS_OK" == "false" ]]; then
    echo ""
    warn "Models missing! Sync them with:"
    echo "  rsync -avP user@source-machine:~/voice/models/ models/"
    echo ""
fi

echo ""

# Run smoke test
info "Running smoke test..."
if .venv/bin/pytest tests/smoke/test_smoke.py::TestBinaryExists -v -q 2>/dev/null; then
    info "  Smoke test passed"
else
    warn "  Smoke test failed (models may be missing)"
fi

echo ""

# Summary
echo "=============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "To start the daemon:"
echo "  ./stream-tts-cpp/build/stream-tts-cpp --daemon stream-tts-cpp/config/kokoro-mps-en.yaml &"
echo ""
echo "To speak:"
echo "  ./stream-tts-cpp/build/stream-tts-cpp --speak \"Hello world\""
echo ""
echo "To run full test suite:"
echo "  make test-smoke"
echo ""

if [[ "$MODELS_OK" == "false" ]]; then
    echo -e "${YELLOW}NOTE: Models are missing. Sync them first:${NC}"
    echo "  ./scripts/sync_models.sh <user@source-machine>"
    echo ""
fi
