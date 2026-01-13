#!/bin/bash
# run_test_with_crash_check.sh - Run any test and FAIL if crashes occur
#
# Usage:
#   ./scripts/run_test_with_crash_check.sh python3 tests/test_stress.py
#   ./scripts/run_test_with_crash_check.sh python3 tests/complete_story_test_suite.py
#
# This script:
# 1. Counts crash logs BEFORE running the test
# 2. Runs the test with the AGX fix dylib
# 3. Counts crash logs AFTER
# 4. FAILS if any new crashes occurred (even if Python exited 0)
#
# CRITICAL: A test that causes crashes is NOT passing, regardless of Python exit code.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CRASH_DIR="$REPO_ROOT/crash_logs"
# Prefer the newest fix if it's been built locally, but keep a safe fallback.
# v2.9 fixes formal verification gaps (commit race, timeout escape, parallel encoder)
DEFAULT_AGX_FIX_DYLIB="$REPO_ROOT/agx_fix/build/libagx_fix_v2_9.dylib"
if [ ! -f "$DEFAULT_AGX_FIX_DYLIB" ]; then
    DEFAULT_AGX_FIX_DYLIB="$REPO_ROOT/agx_fix/build/libagx_fix_v2_8.dylib"
fi
if [ ! -f "$DEFAULT_AGX_FIX_DYLIB" ]; then
    DEFAULT_AGX_FIX_DYLIB="$REPO_ROOT/agx_fix/build/libagx_fix_v2_7.dylib"
fi
AGX_FIX_DYLIB="${AGX_FIX_DYLIB:-$DEFAULT_AGX_FIX_DYLIB}"
# Allow time for crash_monitor.sh / DiagnosticReports to land in crash_logs/
CRASH_WAIT_SECS="${MPS_TEST_CRASH_WAIT_SECS:-6}"

# Colors for output (disable when not writing to a TTY to keep logs plain text)
RED=""
GREEN=""
YELLOW=""
NC=""
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command>"
    echo "Example: $0 python3 tests/test_stress_extended.py"
    exit 1
fi

# Ensure crash_logs directory exists
mkdir -p "$CRASH_DIR"

# Metal visibility preflight (common failure mode under sandboxed/headless runners).
# If Metal is not visible, skip running MPS tests and exit 0 to allow non-Metal work.
if [ "${MPS_TEST_METAL_PREFLIGHT:-1}" != "0" ]; then
    METAL_DIAG="$REPO_ROOT/tests/metal_diagnostics.sh"
    if [ -x "$METAL_DIAG" ]; then
        METAL_OUT=""
        set +e
        if [ "${MPS_TEST_METAL_PREFLIGHT_VERBOSE:-0}" != "0" ]; then
            "$METAL_DIAG" --check --full
            METAL_RC=$?
        else
            METAL_OUT="$("$METAL_DIAG" --check 2>&1)"
            METAL_RC=$?
        fi
        set -e

        if [ "$METAL_RC" -eq 1 ]; then
            if [ -n "$METAL_OUT" ]; then
                echo "$METAL_OUT"
                echo ""
            fi
            echo "════════════════════════════════════════════════════════════════"
            echo "SKIP: Metal devices are NOT visible to this process."
            echo "Do not run MPS tests in this environment (common under sandbox/headless runners)."
            echo "════════════════════════════════════════════════════════════════"
            exit 0
        fi

        if [ "$METAL_RC" -eq 2 ]; then
            echo "WARNING: Unable to determine Metal visibility (tests/metal_diagnostics.sh --check returned 2)."
            if [ -n "$METAL_OUT" ]; then
                echo "$METAL_OUT"
                echo ""
            fi
        fi
    fi
fi

count_crash_logs() {
    find "$CRASH_DIR" -maxdepth 1 -type f \( -name "*.ips" -o -name "*.crash" \) 2>/dev/null | wc -l | tr -d ' '
}

validate_dyld_insert_libraries() {
    local libs="$1"
    local lib
    local -a lib_array=()

    IFS=':' read -r -a lib_array <<< "${libs}"
    for lib in "${lib_array[@]+"${lib_array[@]}"}"; do
        if [ -z "${lib}" ]; then
            continue
        fi
        if [ ! -f "${lib}" ]; then
            echo "ERROR: DYLD_INSERT_LIBRARIES references missing file: ${lib}" >&2
            echo "Fix: build the AGX fix dylib (cd agx_fix && make) or set AGX_FIX_DYLIB to an existing path." >&2
            exit 2
        fi
    done
}

already_captured() {
    local crash_file="$1"
    local base
    base="$(basename "$crash_file")"

    # We store captures as "<timestamp>_<original_basename>" (crash_monitor.sh convention).
    find "$CRASH_DIR" -maxdepth 1 -name "*_${base}" -print -quit 2>/dev/null | grep -q .
}

capture_new_crash_reports_since() {
    local since_epoch="$1"
    local copied=0

    # Create a marker file with the "since" timestamp so we can use `find -newer`.
    # `touch -t` expects local time: [[CC]YY]MMDDhhmm[.SS]
    local marker
    # Use a repo-local temp file to support sandboxed runners that restrict writes
    # to system temp locations (e.g. /tmp).
    marker="$(mktemp "${CRASH_DIR}/mps_crash_marker.XXXXXX")"
    if ! touch -t "$(date -r "$since_epoch" +'%Y%m%d%H%M.%S' 2>/dev/null)" "$marker" 2>/dev/null; then
        touch "$marker"
    fi

    local user_dir="$HOME/Library/Logs/DiagnosticReports"
    local system_dir="/Library/Logs/DiagnosticReports"

    for dir in "$user_dir" "$system_dir"; do
        if [ ! -d "$dir" ]; then
            continue
        fi

        # Only consider crash reports produced by Python processes to avoid pulling in unrelated Metal crashes.
        while IFS= read -r crash_file; do
            if [ ! -f "$crash_file" ]; then
                continue
            fi

            if already_captured "$crash_file"; then
                continue
            fi

            # Filter for our target domain to reduce false positives.
            if ! grep -q -E "Python|AGX|Metal|mps|torch" "$crash_file" 2>/dev/null; then
                continue
            fi

            local timestamp
            timestamp="$(date +%Y%m%d_%H%M%S)"
            local base
            base="$(basename "$crash_file")"
            local dest="$CRASH_DIR/${timestamp}_${base}"

            cp "$crash_file" "$dest"
            copied=$((copied + 1))
        done < <(
            find "$dir" -maxdepth 1 -type f \
                \( -name "*Python*.ips" -o -name "*python*.ips" -o -name "*Python*.crash" -o -name "*python*.crash" \) \
                -newer "$marker" 2>/dev/null || true
        )
    done

    rm -f "$marker" 2>/dev/null || true

    echo "$copied"
}

# Count crashes BEFORE
BEFORE_COUNT="$(count_crash_logs)"
BEFORE_TIME=$(date +%s)

echo "════════════════════════════════════════════════════════════════"
echo "TEST WITH CRASH CHECK"
echo "════════════════════════════════════════════════════════════════"
echo "Command: $*"
echo "AGX Fix (requested): $AGX_FIX_DYLIB"
echo "Crash wait: ${CRASH_WAIT_SECS}s"
echo "Crashes before: $BEFORE_COUNT"
echo "════════════════════════════════════════════════════════════════"

# Run the test with AGX fix
DYLD_EFFECTIVE="$AGX_FIX_DYLIB"
if [ -n "${DYLD_INSERT_LIBRARIES:-}" ]; then
    if [[ "${DYLD_INSERT_LIBRARIES}" == *"libagx_fix"* ]]; then
        DYLD_EFFECTIVE="${DYLD_INSERT_LIBRARIES}"
    else
        DYLD_EFFECTIVE="${AGX_FIX_DYLIB}:${DYLD_INSERT_LIBRARIES}"
    fi
fi

MPS_FORCE_GRAPH_PATH_EFFECTIVE="${MPS_FORCE_GRAPH_PATH:-1}"

echo "DYLD_INSERT_LIBRARIES (effective): $DYLD_EFFECTIVE"
echo "MPS_FORCE_GRAPH_PATH (effective): $MPS_FORCE_GRAPH_PATH_EFFECTIVE"
echo "════════════════════════════════════════════════════════════════"

validate_dyld_insert_libraries "$DYLD_EFFECTIVE"

set +e
DYLD_INSERT_LIBRARIES="$DYLD_EFFECTIVE" MPS_FORCE_GRAPH_PATH="$MPS_FORCE_GRAPH_PATH_EFFECTIVE" "$@"
TEST_EXIT_CODE=$?
set -e

# Small delay to ensure crash logs are written and/or copied by crash_monitor.sh.
sleep "$CRASH_WAIT_SECS"

# Opportunistically copy new crash reports from DiagnosticReports into crash_logs/
# (covers the case where crash_monitor.sh isn't running).
COPIED_CRASH_REPORTS="$(capture_new_crash_reports_since "$BEFORE_TIME")"
if [ "${COPIED_CRASH_REPORTS:-0}" -gt 0 ]; then
    echo "Captured $COPIED_CRASH_REPORTS new crash report(s) from DiagnosticReports."
fi

# Count crashes AFTER
AFTER_COUNT="$(count_crash_logs)"
NEW_CRASHES=$((AFTER_COUNT - BEFORE_COUNT))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "RESULTS"
echo "════════════════════════════════════════════════════════════════"
echo "Python exit code: $TEST_EXIT_CODE"
echo "Crashes before: $BEFORE_COUNT"
echo "Crashes after: $AFTER_COUNT"
echo "NEW CRASHES: $NEW_CRASHES"
echo "════════════════════════════════════════════════════════════════"

# FAIL if any new crashes
if [ "$NEW_CRASHES" -gt 0 ]; then
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  FAIL: $NEW_CRASHES NEW CRASH(ES) DETECTED!                            ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Latest crash details:"
    python3 "$SCRIPT_DIR/check_crashes.py" --sync >/dev/null 2>&1 || true
    python3 "$SCRIPT_DIR/check_crashes.py" --latest 2>/dev/null | head -30
    echo ""
    echo -e "${RED}A test that crashes is NOT passing.${NC}"
    exit 1
fi

# Check Python exit code
if [ "$TEST_EXIT_CODE" -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  FAIL: Test exited with code $TEST_EXIT_CODE                           ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit "$TEST_EXIT_CODE"
fi

# Success - no crashes AND exit code 0
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  PASS: No crashes, exit code 0                                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
exit 0
