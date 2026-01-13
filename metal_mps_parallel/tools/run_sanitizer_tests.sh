#!/bin/bash
# R2: Sanitizer Test Runner
#
# Runs all TSan tests and reports status.
# Exit 0 = all tests passed
# Exit 1 = test failure
#
# Usage:
#   ./tools/run_sanitizer_tests.sh              # Default: 16 threads, 100 iterations
#   ./tools/run_sanitizer_tests.sh --extended   # Extended: 31 threads, 500 iterations
#   ./tools/run_sanitizer_tests.sh --quick      # Quick: 8 threads, 50 iterations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$ROOT_DIR/tests"

# Default configuration
THREADS=16
ITERATIONS=100
MODE="standard"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --extended)
            THREADS=31
            ITERATIONS=500
            MODE="extended"
            shift
            ;;
        --quick)
            THREADS=8
            ITERATIONS=50
            MODE="quick"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--extended|--quick]"
            echo ""
            echo "Options:"
            echo "  --extended   Run with 31 threads, 500 iterations"
            echo "  --quick      Run with 8 threads, 50 iterations"
            echo "  (default)    Run with 16 threads, 100 iterations"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  TSan Sanitizer Test Suite"
echo "=============================================="
echo "Mode: $MODE"
echo "Threads: $THREADS"
echo "Iterations: $ITERATIONS"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="
echo ""

# Set TSan options
export TSAN_OPTIONS="suppressions=$TESTS_DIR/tsan_suppressions.txt:halt_on_error=1:second_deadlock_stack=1"

# Track results
PASSED=0
FAILED=0
RESULTS=()

# Function to run a test
run_test() {
    local name="$1"
    shift
    local cmd="$@"

    echo ">>> Running: $name"
    if eval "$cmd"; then
        echo "<<< PASSED: $name"
        ((PASSED++))
        RESULTS+=("PASS: $name")
    else
        echo "<<< FAILED: $name"
        ((FAILED++))
        RESULTS+=("FAIL: $name")
    fi
    echo ""
}

# Check binaries exist
if [ ! -f "$TESTS_DIR/tsan_mps_test" ]; then
    echo "ERROR: $TESTS_DIR/tsan_mps_test not found"
    echo "Build it with: ./tests/build_tsan_test.sh"
    exit 1
fi

if [ ! -f "$TESTS_DIR/record_stream_test_tsan" ]; then
    echo "WARNING: $TESTS_DIR/record_stream_test_tsan not found"
    echo "Build it with: ./tests/build_record_stream_test.sh"
fi

# Run TSan tests
echo "=== TSan Tests ==="
echo ""

# Core MPS stream pool test
run_test "tsan_mps_test ($THREADS threads × $ITERATIONS iterations)" \
    "$TESTS_DIR/tsan_mps_test $THREADS $ITERATIONS"

# Record stream test (if available)
if [ -f "$TESTS_DIR/record_stream_test_tsan" ]; then
    run_test "record_stream_test_tsan" \
        "$TESTS_DIR/record_stream_test_tsan"
fi

# Summary
echo "=============================================="
echo "  Test Summary"
echo "=============================================="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo "=============================================="

# Exit code
if [ $FAILED -eq 0 ]; then
    echo ""
    echo "=== ALL SANITIZER TESTS PASSED ==="
    exit 0
else
    echo ""
    echo "=== SANITIZER TEST FAILURES DETECTED ==="
    exit 1
fi
