#!/bin/bash
# export_cosyvoice_model.sh - Reliable CosyVoice2 model export with isolated environment
#
# This script creates an isolated Python environment with pinned dependencies
# to ensure reproducible model exports, avoiding version conflicts.
#
# Usage:
#   ./scripts/export_cosyvoice_model.sh              # Export all components
#   ./scripts/export_cosyvoice_model.sh --components # Export TorchScript components only
#   ./scripts/export_cosyvoice_model.sh --flow       # Export Flow + HiFT only
#   ./scripts/export_cosyvoice_model.sh --clean      # Remove export venv and recreate
#
# Copyright 2025 Andrew Yates. All rights reserved.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR/.."
EXPORT_VENV="$PROJECT_DIR/.venv-cosyvoice-export"
REQUIREMENTS="$SCRIPT_DIR/requirements-cosyvoice-export.txt"
COSYVOICE_REPO="$PROJECT_DIR/cosyvoice_repo"
MODEL_DIR="$PROJECT_DIR/models/cosyvoice/CosyVoice2-0.5B"
OUTPUT_DIR="$PROJECT_DIR/models/cosyvoice"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
EXPORT_MODE="all"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --components)
            EXPORT_MODE="components"
            shift
            ;;
        --flow)
            EXPORT_MODE="flow"
            shift
            ;;
        --all)
            EXPORT_MODE="all"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--components|--flow|--all] [--clean]"
            echo ""
            echo "Options:"
            echo "  --components  Export TorchScript components (llm_decoder, speech_embedding)"
            echo "  --flow        Export Flow encoder + HiFT vocoder"
            echo "  --all         Export all components (default)"
            echo "  --clean       Remove export venv and recreate"
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
    echo -e "${YELLOW}Removing existing CosyVoice export venv...${NC}"
    rm -rf "$EXPORT_VENV"
fi

# Check CosyVoice repo exists
if [ ! -d "$COSYVOICE_REPO" ]; then
    echo -e "${RED}CosyVoice repo not found: $COSYVOICE_REPO${NC}"
    echo "Clone it with: git clone https://github.com/FunAudioLLM/CosyVoice.git cosyvoice_repo"
    exit 1
fi

# Check model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}CosyVoice2-0.5B model not found: $MODEL_DIR${NC}"
    echo "Download from: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B"
    exit 1
fi

# Find Python 3.11 (CosyVoice works best with 3.10-3.11)
PYTHON311=""
for p in python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11; do
    if command -v "$p" &> /dev/null; then
        PYTHON311="$p"
        break
    fi
done

if [ -z "$PYTHON311" ]; then
    echo -e "${RED}Python 3.11 not found. CosyVoice requires Python 3.10-3.11.${NC}"
    echo "Install with: brew install python@3.11"
    exit 1
fi

echo -e "${YELLOW}Using Python: $PYTHON311 ($($PYTHON311 --version))${NC}"

# Create isolated venv if needed
if [ ! -d "$EXPORT_VENV" ]; then
    echo -e "${YELLOW}Creating isolated CosyVoice export environment...${NC}"
    "$PYTHON311" -m venv "$EXPORT_VENV"

    echo -e "${YELLOW}Installing pinned dependencies...${NC}"
    "$EXPORT_VENV/bin/pip" install --upgrade pip
    "$EXPORT_VENV/bin/pip" install -r "$REQUIREMENTS"

    echo -e "${GREEN}CosyVoice export environment created${NC}"
fi

# Verify environment
echo ""
echo -e "${YELLOW}Verifying CosyVoice export environment...${NC}"
"$EXPORT_VENV/bin/python" -c "
import torch
import transformers
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print('  All dependencies OK')
"

# Create output directories
mkdir -p "$OUTPUT_DIR/torchscript"
mkdir -p "$OUTPUT_DIR/exported"

# Export functions
export_components() {
    echo ""
    echo -e "${YELLOW}Exporting CosyVoice2 TorchScript components...${NC}"

    PYTHONPATH="$COSYVOICE_REPO:$PYTHONPATH" "$EXPORT_VENV/bin/python" "$SCRIPT_DIR/export_cosyvoice_components.py"

    # Verify outputs
    for f in llm_decoder.pt speech_embedding.pt llm_embedding.pt; do
        if [ -f "$OUTPUT_DIR/torchscript/$f" ]; then
            local size=$(du -h "$OUTPUT_DIR/torchscript/$f" | cut -f1)
            echo -e "${GREEN}Exported: $OUTPUT_DIR/torchscript/$f ($size)${NC}"
        else
            echo -e "${RED}Export failed: $OUTPUT_DIR/torchscript/$f not found${NC}"
            exit 1
        fi
    done
}

export_flow() {
    echo ""
    echo -e "${YELLOW}Exporting CosyVoice2 Flow + HiFT models...${NC}"

    PYTHONPATH="$COSYVOICE_REPO:$PYTHONPATH" "$EXPORT_VENV/bin/python" "$SCRIPT_DIR/export_flow_components.py"

    # Verify outputs
    for f in flow_encoder_traced.pt hift_traced.pt; do
        if [ -f "$OUTPUT_DIR/exported/$f" ]; then
            local size=$(du -h "$OUTPUT_DIR/exported/$f" | cut -f1)
            echo -e "${GREEN}Exported: $OUTPUT_DIR/exported/$f ($size)${NC}"
        else
            echo -e "${RED}Export failed: $OUTPUT_DIR/exported/$f not found${NC}"
            exit 1
        fi
    done
}

# Run exports
cd "$PROJECT_DIR"

case $EXPORT_MODE in
    components)
        export_components
        ;;
    flow)
        export_flow
        ;;
    all)
        export_components
        export_flow
        ;;
esac

echo ""
echo -e "${GREEN}CosyVoice2 export complete!${NC}"
echo ""
echo "Models exported to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/torchscript"/*.pt 2>/dev/null | head -5
ls -la "$OUTPUT_DIR/exported"/*.pt 2>/dev/null | head -5
