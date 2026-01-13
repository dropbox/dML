#!/bin/bash
# Wrapper script for running MPS tests safely
#
# NOTE (2025-12-23): Native PyTorch 2.9.1 has improved MPS threading but can still
# intermittently crash at 8+ threads due to the underlying AGX driver race.
# Using an AGX-fix dylib + this wrapper's retry logic typically improves effective
# stability for verification runs, but some workloads can still crash due to
# driver-level races.
# By default, this script runs without the dylib (for speed in CI).
#
# To inject the AGX fix dylib, set:
#   MPS_USE_AGX_FIX=1 ./scripts/run_mps_test.sh ...
#
# Dylib selection:
# - Default: agx_fix/build/libagx_fix_v2_9.dylib (closes formal verification gaps)
# - Override: AGX_FIX_DYLIB=/path/to/libagx_fix_v2_9.dylib MPS_USE_AGX_FIX=1 ./scripts/run_mps_test.sh ...
# - Fallbacks: v2_8, v2_7, v2_5, v2_4_nr, ...
#
# Crash reporting:
# - This wrapper checks for newly captured crash logs in `crash_logs/` after each attempt.
# - By default, ANY new crash log causes a non-zero exit, even if a later retry succeeds.
#   Override with `MPS_TEST_STRICT_CRASH_LOGS=0` to keep exit-code behavior (still prints logs).
#
# Multi-threaded MPS workloads remain intermittent in Apple's AGX stack; this
# wrapper retries SIGSEGV up to:
#   MPS_TEST_MAX_RETRIES (default: 3)
#
# Usage:
#   ./scripts/run_mps_test.sh tests/verify_layernorm_fix.py
#   ./scripts/run_mps_test.sh tests/complete_story_test_suite.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CRASH_DIR="$REPO_ROOT/crash_logs"

mkdir -p "$CRASH_DIR"

# Metal visibility preflight (common failure mode under sandboxed/headless runners).
# If Metal is not visible, skip running MPS tests and exit 0 to allow non-Metal work.
if [ "${MPS_TEST_METAL_PREFLIGHT:-1}" != "0" ]; then
    METAL_DIAG="$REPO_ROOT/tests/metal_diagnostics.sh"
    if [ -x "$METAL_DIAG" ]; then
        METAL_OUT="$("$METAL_DIAG" --check 2>&1)"
        METAL_RC=$?

        if [ "$METAL_RC" -eq 1 ]; then
            if [ -n "$METAL_OUT" ]; then
                printf '%s\n\n' "$METAL_OUT"
            fi
            echo "════════════════════════════════════════════════════════════════"
            echo "SKIP: Metal devices are NOT visible to this process."
            echo "Do not run MPS tests in this environment (common under sandbox/headless runners)."
            echo "════════════════════════════════════════════════════════════════"
            exit 0
        fi

        if [ "$METAL_RC" -eq 2 ]; then
            echo "WARNING: Unable to determine Metal visibility (tests/metal_diagnostics.sh --check returned 2)." >&2
            if [ -n "$METAL_OUT" ]; then
                printf '%s\n' "$METAL_OUT" >&2
            fi
        fi
    fi
fi

STRICT_CRASH_LOGS="${MPS_TEST_STRICT_CRASH_LOGS:-1}"
CRASH_WAIT_SECS="${MPS_TEST_CRASH_WAIT_SECS:-6}"
SAW_NEW_CRASH_LOGS=0

MARKER_FILE="$(mktemp "$CRASH_DIR/.mps_test_marker.XXXXXX")"
cleanup_marker() {
    rm -f "$MARKER_FILE"
}
trap cleanup_marker EXIT

touch "$MARKER_FILE"

already_captured() {
    local crash_file="$1"
    local base
    base="$(basename "$crash_file")"

    # We store captures as "<timestamp>_<original_basename>" (crash_monitor.sh convention).
    find "$CRASH_DIR" -maxdepth 1 -name "*_${base}" -print -quit 2>/dev/null | grep -q .
}

capture_new_crash_reports_since_marker() {
    local marker_file="$1"
    local copied=0

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
                -newer "$marker_file" 2>/dev/null || true
        )
    done

    echo "$copied"
}

report_new_crash_logs() {
    local label="$1"

    # Allow crash_monitor.sh time to copy new .ips into crash_logs/.
    sleep "$CRASH_WAIT_SECS"

    # If crash_monitor.sh isn't running, opportunistically copy new crash reports from DiagnosticReports.
    local copied_reports
    copied_reports="$(capture_new_crash_reports_since_marker "$MARKER_FILE")"
    if [ "${copied_reports:-0}" -gt 0 ]; then
        echo "Captured $copied_reports new crash report(s) from DiagnosticReports." >&2
    fi

    local new_count
    new_count=$(find "$CRASH_DIR" -maxdepth 1 \( -name "*.ips" -o -name "*.crash" \) -newer "$MARKER_FILE" -print 2>/dev/null | wc -l | tr -d ' ')
    if [ "$new_count" != "0" ]; then
        SAW_NEW_CRASH_LOGS=1
        echo "NEW CRASH LOGS ($label): $new_count" >&2
        find "$CRASH_DIR" -maxdepth 1 \( -name "*.ips" -o -name "*.crash" \) -newer "$MARKER_FILE" -print 2>/dev/null \
            | sed "s|^$CRASH_DIR/||" | sed 's/^/  - /' >&2

        local crash_script="$REPO_ROOT/scripts/check_crashes.py"
        if [ -f "$crash_script" ]; then
            echo "Latest crash summary (crash_logs/):" >&2
            python3 "$crash_script" --latest --json 2>/dev/null | head -200 >&2 || true
        fi
    fi

    # Reset marker so each attempt only reports newly-added logs once.
    touch "$MARKER_FILE"
}

# AGX fix dylib + retry logic improves stability; native PyTorch can still crash without it
if [ "${MPS_USE_AGX_FIX:-0}" = "1" ]; then
    AGX_FIX_LIB="${AGX_FIX_DYLIB:-}"
    if [ -z "${AGX_FIX_LIB}" ]; then
        AGX_FIX_CANDIDATES=(
            "$REPO_ROOT/agx_fix/build/libagx_fix_v2_9.dylib"
            "$REPO_ROOT/agx_fix/build/libagx_fix_v2_8.dylib"
            "$REPO_ROOT/agx_fix/build/libagx_fix_v2_7.dylib"
            "$REPO_ROOT/agx_fix/build/libagx_fix_v2_5.dylib"
            "$REPO_ROOT/agx_fix/build/libagx_fix_v2_4_nr.dylib"
            "$REPO_ROOT/agx_fix/build/libagx_fix_v2_3.dylib"
            "$REPO_ROOT/agx_fix/build/libagx_fix_v2.dylib"
            "$REPO_ROOT/agx_fix/build/libagx_fix.dylib"
        )
        for candidate in "${AGX_FIX_CANDIDATES[@]}"; do
            if [ -f "${candidate}" ]; then
                AGX_FIX_LIB="${candidate}"
                break
            fi
        done
    fi

    if [ -n "${AGX_FIX_LIB}" ] && [ -f "${AGX_FIX_LIB}" ]; then
        if [ -n "${DYLD_INSERT_LIBRARIES:-}" ] && [ "${DYLD_INSERT_LIBRARIES}" != "$AGX_FIX_LIB" ]; then
            echo "WARNING: Overriding DYLD_INSERT_LIBRARIES=${DYLD_INSERT_LIBRARIES}" >&2
        fi
        export DYLD_INSERT_LIBRARIES="$AGX_FIX_LIB"
        echo "AGX fix loaded: $AGX_FIX_LIB (MPS_USE_AGX_FIX=1)"
    else
        echo "WARNING: AGX fix requested but not found (build: cd agx_fix && make)" >&2
    fi
else
    echo "Native PyTorch MPS (no dylib - use MPS_USE_AGX_FIX=1 for additional protection)"
fi

echo "Running: python3 $@"
echo "---"

# Some multi-threaded MPS workloads are still intermittent (Apple driver/framework).
# Retry SIGSEGV a small number of times to avoid flaky automation.
MAX_RETRIES="${MPS_TEST_MAX_RETRIES:-3}"
attempt=1
PRINT_CRASH="${MPS_TEST_PRINT_CRASH:-1}"

print_latest_crash_report() {
    if [ "${PRINT_CRASH}" = "0" ]; then
        return
    fi
    local crash_script="$REPO_ROOT/scripts/collect_crash_reports.sh"
    if [ -x "$crash_script" ]; then
        echo "Latest crash report (macOS DiagnosticReports):" >&2
        "$crash_script" --last 1 >&2 || true
    fi
}

while true; do
    python3 "$@"
    exit_code=$?

    report_new_crash_logs "attempt ${attempt}/${MAX_RETRIES} (rc=${exit_code})"

    if [ $exit_code -eq 0 ]; then
        if [ "$STRICT_CRASH_LOGS" = "1" ] && [ "$SAW_NEW_CRASH_LOGS" = "1" ]; then
            echo "FAIL: Command returned 0 but new crash logs were captured." >&2
            echo "Set MPS_TEST_STRICT_CRASH_LOGS=0 to ignore crash_logs/ and return success." >&2
            exit 1
        fi
        exit 0
    fi

    # 139 = 128 + SIGSEGV
    if [ $exit_code -eq 139 ] && [ $attempt -lt $MAX_RETRIES ]; then
        attempt=$((attempt + 1))
        echo "RETRY: SIGSEGV (exit 139) - attempt ${attempt}/${MAX_RETRIES}" >&2
        sleep 2
        print_latest_crash_report
        continue
    fi

    if [ $exit_code -eq 139 ]; then
        echo "FAIL: SIGSEGV (exit 139) - attempts exhausted (${attempt}/${MAX_RETRIES})" >&2
        sleep 2
        print_latest_crash_report
    fi

    exit $exit_code
done
