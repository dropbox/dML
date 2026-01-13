#!/bin/bash
# export_model.sh - Reliable Kokoro model export with isolated environment
#
# This script creates an isolated Python environment with pinned dependencies
# to ensure reproducible model exports, avoiding version conflicts.
#
# Usage:
#   ./scripts/export_model.sh              # Export MPS model (default)
#   ./scripts/export_model.sh --cpu        # Export CPU model
#   ./scripts/export_model.sh --both       # Export both MPS and CPU
#   ./scripts/export_model.sh --clean      # Remove export venv and recreate
#
# Copyright 2025 Andrew Yates. All rights reserved.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR/.."
EXPORT_VENV="$PROJECT_DIR/.venv-export"
REQUIREMENTS="$SCRIPT_DIR/requirements-export.txt"
OUTPUT_DIR="$PROJECT_DIR/models/kokoro"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
DEVICE="mps"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --mps)
            DEVICE="mps"
            shift
            ;;
        --both)
            DEVICE="both"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--cpu|--mps|--both] [--clean]"
            echo ""
            echo "Options:"
            echo "  --mps    Export MPS model (default, Metal GPU)"
            echo "  --cpu    Export CPU model"
            echo "  --both   Export both MPS and CPU models"
            echo "  --clean  Remove export venv and recreate"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true ] && [ -d "$EXPORT_VENV" ]; then
    echo -e "${YELLOW}Removing existing export venv...${NC}"
    rm -rf "$EXPORT_VENV"
fi

# Find Python 3.11 (kokoro requires 3.10-3.12)
PYTHON311=""
for p in python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11; do
    if command -v "$p" &> /dev/null; then
        PYTHON311="$p"
        break
    fi
done

if [ -z "$PYTHON311" ]; then
    echo -e "${RED}Python 3.11 not found. Kokoro requires Python 3.10-3.12.${NC}"
    echo "Install with: brew install python@3.11"
    exit 1
fi

echo -e "${YELLOW}Using Python: $PYTHON311 ($($PYTHON311 --version))${NC}"

# Create isolated venv if needed
if [ ! -d "$EXPORT_VENV" ]; then
    echo -e "${YELLOW}Creating isolated export environment...${NC}"
    "$PYTHON311" -m venv "$EXPORT_VENV"

    echo -e "${YELLOW}Installing pinned dependencies...${NC}"
    "$EXPORT_VENV/bin/pip" install --upgrade pip
    "$EXPORT_VENV/bin/pip" install -r "$REQUIREMENTS"

    # Patch misaki to fix phonemizer API compatibility
    # misaki calls set_data_path() but phonemizer 3.x uses data_path property
    echo -e "${YELLOW}Patching misaki for phonemizer compatibility...${NC}"
    MISAKI_ESPEAK="$EXPORT_VENV/lib/python3.11/site-packages/misaki/espeak.py"
    if [ -f "$MISAKI_ESPEAK" ]; then
        # Replace set_data_path() with data_path assignment
        sed -i.bak 's/EspeakWrapper.set_data_path(/EspeakWrapper.data_path = (/g' "$MISAKI_ESPEAK"
        # Use system espeak-ng if available (more reliable than espeakng_loader)
        if [ -f "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib" ]; then
            cat > "$MISAKI_ESPEAK" << 'PATCH_EOF'
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from typing import Tuple
import os
import phonemizer
import re

# Use system espeak-ng (homebrew) - more reliable than espeakng_loader
_HOMEBREW_ESPEAK = '/opt/homebrew/opt/espeak-ng'
_ESPEAK_LIB = os.path.join(_HOMEBREW_ESPEAK, 'lib', 'libespeak-ng.dylib')
_ESPEAK_DATA = os.path.join(_HOMEBREW_ESPEAK, 'share', 'espeak-ng-data')

if os.path.exists(_ESPEAK_LIB):
    EspeakWrapper.set_library(_ESPEAK_LIB)
    EspeakWrapper.data_path = _ESPEAK_DATA
else:
    import espeakng_loader
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.data_path = espeakng_loader.get_data_path()

PATCH_EOF
            # Append the rest of the original file (after line 10)
            tail -n +11 "$MISAKI_ESPEAK.bak" >> "$MISAKI_ESPEAK"
        fi
        rm -f "$MISAKI_ESPEAK.bak"
        # Clear Python cache
        find "$EXPORT_VENV/lib/python3.11/site-packages/misaki" -name "*.pyc" -delete 2>/dev/null
        find "$EXPORT_VENV/lib/python3.11/site-packages/misaki" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        echo -e "${GREEN}Misaki patched${NC}"
    fi

    echo -e "${GREEN}Export environment created${NC}"
fi

# Verify environment
echo ""
echo -e "${YELLOW}Verifying export environment...${NC}"
"$EXPORT_VENV/bin/python" -c "
import torch
import kokoro
import phonemizer
print(f'  PyTorch: {torch.__version__}')
print(f'  Kokoro: {kokoro.__version__ if hasattr(kokoro, \"__version__\") else \"0.9.4\"}')
print(f'  Phonemizer: {phonemizer.__version__}')
print('  All dependencies OK')
"

# Export model(s)
cd "$PROJECT_DIR"

export_model() {
    local device=$1
    echo ""
    echo -e "${YELLOW}Exporting Kokoro model for ${device}...${NC}"

    "$EXPORT_VENV/bin/python" scripts/export_kokoro_torchscript.py \
        --device "$device" \
        --output "$OUTPUT_DIR"

    # Verify output
    local model_file="$OUTPUT_DIR/kokoro_${device}.pt"
    if [ -f "$model_file" ]; then
        local size=$(du -h "$model_file" | cut -f1)
        echo -e "${GREEN}Exported: $model_file ($size)${NC}"
    else
        echo -e "${RED}Export failed: $model_file not found${NC}"
        exit 1
    fi
}

case $DEVICE in
    mps)
        export_model mps
        ;;
    cpu)
        export_model cpu
        ;;
    both)
        export_model mps
        export_model cpu
        ;;
esac

echo ""
echo -e "${GREEN}Export complete!${NC}"
echo ""
echo "Models exported to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.pt 2>/dev/null | head -5
