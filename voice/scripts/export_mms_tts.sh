#!/bin/bash
#
# MMS-TTS Model Export - Isolated Environment
#
# This script creates an isolated Python virtual environment with pinned
# dependencies for reproducible MMS-TTS (Arabic, Turkish, Persian) model export.
#
# Usage:
#   ./scripts/export_mms_tts.sh              # Export all languages
#   ./scripts/export_mms_tts.sh --lang ar    # Export Arabic only
#   ./scripts/export_mms_tts.sh --lang tr    # Export Turkish only
#   ./scripts/export_mms_tts.sh --lang fa    # Export Persian only
#   ./scripts/export_mms_tts.sh --clean      # Remove venv and re-export
#
# Copyright 2025 Andrew Yates. All rights reserved.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv-mms-tts-export"
REQUIREMENTS="$SCRIPT_DIR/requirements-mms-tts-export.txt"
EXPORT_SCRIPT="$SCRIPT_DIR/export_mms_tts_torchscript.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "MMS-TTS Model Export (Isolated Environment)"
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

# Default to --all if no arguments
if [[ $# -eq 0 ]]; then
    python "$EXPORT_SCRIPT" --all
else
    python "$EXPORT_SCRIPT" "$@"
fi

deactivate

echo ""
echo -e "${GREEN}Export complete!${NC}"
echo "Output: $PROJECT_DIR/models/mms-tts/"
