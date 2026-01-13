#!/bin/bash
# Run verification rounds with crash-log checking.
#
# IMPORTANT: All MPS tests must be run with crash log checking. A run that
# produces new crash logs is NOT passing, even if Python exits 0.
#
# This script uses:
#   scripts/run_test_with_crash_check.sh
#
# Which injects:
# - DYLD_INSERT_LIBRARIES (AGX fix dylib, default: v2.9; falls back to v2.8/v2.7)
# - MPS_FORCE_GRAPH_PATH=1 (graph API path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUNNER="$SCRIPT_DIR/run_test_with_crash_check.sh"

NUM_ROUNDS=${1:-25}
passed=0
failed=0
total_ops=0

echo "Starting $NUM_ROUNDS verification rounds at $(date)"
echo ""

for i in $(seq 1 $NUM_ROUNDS); do
    set +e
    result="$("$RUNNER" python3 "$REPO_ROOT/tests/complete_story_test_suite.py" 2>&1)"
    rc=$?
    set -e

    if [ "$rc" -eq 0 ] && echo "$result" | grep -q "ALL CLAIMS VERIFIED"; then
        passed=$((passed + 1))
        total_ops=$((total_ops + 160))
        echo "Round $i: PASS"
    else
        failed=$((failed + 1))
        echo "Round $i: FAIL (rc=$rc)"
        # Show the summary and any errors
        if [ -z "$result" ]; then
            echo "  (empty output - likely crashed)"
        else
            echo "$result" | grep -E "(PASSED|FAILED|CONFIRMED|ERROR|Traceback|Exception|crash)" | head -10
        fi
    fi
    # Small delay between iterations to allow Metal cleanup
    sleep 0.5
done

echo ""
echo "=== Summary ==="
echo "Passed: $passed/$NUM_ROUNDS"
echo "Failed: $failed/$NUM_ROUNDS"
echo "Total ops: $total_ops (8 threads x 20 iterations x $passed rounds)"
echo "Pass rate: $(echo "scale=1; $passed * 100 / $NUM_ROUNDS" | bc)%"
echo "Completed at $(date)"
