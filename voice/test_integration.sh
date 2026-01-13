#!/bin/bash
# Canonical integration test runner - uses pytest targets in Makefile

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  INTEGRATION TESTS (pytest)"
echo "=============================================="
echo ""

echo "[1/4] Unit tests (Python + C++)..."
make test-unit

echo ""
echo "[2/4] Integration tests..."
make test-integration

echo ""
echo "[3/4] WER/CER aggregate assertions..."
make test-wer

echo ""
echo "[4/4] Quality/LLM judge (optional)..."
if [ -n "${OPENAI_API_KEY:-}" ] && [[ "${OPENAI_API_KEY}" != "sk-..." ]]; then
    make test-quality
else
    echo "Skipping LLM quality tests (OPENAI_API_KEY not configured)."
fi

echo ""
echo "=============================================="
echo "  INTEGRATION SUITE COMPLETE"
echo "=============================================="
