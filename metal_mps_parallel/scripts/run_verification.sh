#!/bin/bash
# Run transformer verification rounds with crash-log checking.
#
# IMPORTANT: All MPS tests must be run with crash log checking. A run that
# produces new crash logs is NOT passing, even if Python exits 0.
#
# This script uses:
#   scripts/run_test_with_crash_check.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUNNER="$SCRIPT_DIR/run_test_with_crash_check.sh"

echo "Running comprehensive transformer test (10 rounds)..."
for i in $(seq 1 10); do
    set +e
    output="$("$RUNNER" timeout 60 python3 "$REPO_ROOT/scripts/test_transformer_threads.py" 2>&1)"
    rc=$?
    set -e

    result="$(echo "$output" | grep -E "RESULT:" | tail -1 || true)"
    if [ -z "$result" ]; then
        result="FAIL (rc=$rc)"
    fi
    echo "Round $i: $result"
done
echo "Final crash count: $(python3 \"$REPO_ROOT/scripts/check_crashes.py\" --count)"
