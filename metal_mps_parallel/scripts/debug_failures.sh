#!/bin/bash
# Debug failing verification runs
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

for i in 1 2 3 4 5; do
    echo "=== Run $i ==="
    set +e
    result="$("$RUNNER" python3 "$REPO_ROOT/tests/complete_story_test_suite.py" 2>&1)"
    rc=$?
    set -e

    if [ "$rc" -eq 0 ] && echo "$result" | grep -q "ALL CLAIMS VERIFIED"; then
        echo "PASS"
    else
        echo "FAIL (rc=$rc) - Full output:"
        echo "$result"
        echo "---"
    fi
done
