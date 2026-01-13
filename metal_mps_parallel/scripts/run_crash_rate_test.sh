#!/bin/bash
# Test crash rate for complete_story test with crash-log checking.
#
# IMPORTANT: All MPS tests must be run with crash log checking. A run that
# produces new crash logs is NOT passing, even if Python exits 0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUNNER="$SCRIPT_DIR/run_test_with_crash_check.sh"

PASSES=0
FAILS=0

for i in 1 2 3 4 5; do
    echo "=== Run $i ==="

    set +e
    output="$("$RUNNER" timeout 90 python3 "$REPO_ROOT/tests/complete_story_test_suite.py" 2>&1)"
    rc=$?
    set -e

    if [ "$rc" -eq 0 ] && echo "$output" | grep -q "ALL CLAIMS VERIFIED"; then
        PASSES=$((PASSES + 1))
        echo "RESULT: PASS"
    else
        FAILS=$((FAILS + 1))
        echo "RESULT: FAIL (rc=$rc)"
        echo "$output" | tail -30 || true
    fi
    echo ""
done

echo "=========================================="
echo "Summary: $PASSES passes, $FAILS fails out of 5 runs"
echo "Crash count: $(python3 \"$REPO_ROOT/scripts/check_crashes.py\" --count)"
