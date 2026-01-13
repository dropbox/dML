#!/bin/bash
#
# Speaker Embedding Model Export - Isolated Environment
#
# This script creates an isolated Python virtual environment with pinned
# dependencies for reproducible ECAPA-TDNN speaker embedding model export.
#
# Usage:
#   ./scripts/export_speaker_model.sh          # Export model
#   ./scripts/export_speaker_model.sh --clean  # Remove venv and re-export
#
# Copyright 2025 Andrew Yates. All rights reserved.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv-speaker-export"
REQUIREMENTS="$SCRIPT_DIR/requirements-speaker-export.txt"
EXPORT_SCRIPT="$SCRIPT_DIR/export_ecapa_tdnn.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Speaker Model Export (Isolated Environment)"
echo "=============================================="

# Handle --clean flag
if [[ "$1" == "--clean" ]]; then
    echo -e "${YELLOW}Cleaning existing venv...${NC}"
    rm -rf "$VENV_DIR"
    shift
fi

# Create venv if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${GREEN}Creating isolated venv at: $VENV_DIR${NC}"
    python3.11 -m venv "$VENV_DIR"

    echo "Installing dependencies from: $REQUIREMENTS"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS"
    deactivate
else
    echo "Using existing venv at: $VENV_DIR"
fi

# Activate venv and run export
echo ""
echo "Running export script..."
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"
python "$EXPORT_SCRIPT" "$@"

deactivate

echo ""
echo -e "${GREEN}Export complete!${NC}"
echo "Output: $PROJECT_DIR/models/speaker/ecapa_tdnn.pt"
