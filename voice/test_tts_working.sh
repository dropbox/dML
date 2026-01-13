#!/bin/bash
# Quick end-to-end TTS sanity using the unified pytest smoke suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  TTS WORKING CHECK"
echo "=============================================="
echo ""

# Delegate to the fast multilingual smoke marker inside tests/smoke/test_smoke.py
make test-multilingual-smoke

echo ""
echo "TTS smoke completed via pytest."
