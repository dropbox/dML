#!/bin/bash
# Canonical smoke test wrapper - delegates to the pytest/Makerflow in tests/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  SMOKE TEST (pytest)"
echo "=============================================="
echo ""

echo "[1/3] Running pytest smoke suite..."
make test-smoke

echo ""
echo "[2/3] Verifying golden audio quality..."
python3 tests/audio_quality.py tests/golden/hello.wav >/dev/null
echo "Audio quality check: PASS"

echo ""
echo "[3/3] Fast multilingual TTS smoke..."
make test-multilingual-smoke

echo ""
echo "=============================================="
echo "  SMOKE TESTS PASSED"
echo "=============================================="
