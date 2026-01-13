#!/bin/bash
# Run a command with crash capture
#
# This script runs a command and captures crash information if it fails.
# Designed for AI workers to use when running tests.
#
# Usage:
#   ./run_with_crash_capture.sh python3 tests/some_test.py
#   ./run_with_crash_capture.sh --with-agx-fix python3 tests/some_test.py
#
# On crash:
#   - Captures exit code
#   - Waits for macOS crash log to be written
#   - Copies crash log to crash_logs/
#   - Updates crash_summary.json
#   - Prints crash info for AI to read

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CRASH_DIR="$REPO_ROOT/crash_logs"
CRASH_SUMMARY="$CRASH_DIR/crash_summary.json"
LATEST_CRASH="$CRASH_DIR/latest_crash.txt"

# macOS crash log locations
USER_CRASH_DIR="$HOME/Library/Logs/DiagnosticReports"

mkdir -p "$CRASH_DIR"

# Parse options
USE_AGX_FIX=0
if [ "$1" = "--with-agx-fix" ]; then
    USE_AGX_FIX=1
    shift
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--with-agx-fix] <command> [args...]"
    exit 1
fi

# Inject AGX fix if requested
if [ "$USE_AGX_FIX" -eq 1 ]; then
    AGX_FIX=""
    AGX_FIX_CANDIDATES=(
        "$REPO_ROOT/agx_fix/build/libagx_fix_v2_9.dylib"
        "$REPO_ROOT/agx_fix/build/libagx_fix_v2_8.dylib"
        "$REPO_ROOT/agx_fix/build/libagx_fix_v2_7.dylib"
        "$REPO_ROOT/agx_fix/build/libagx_fix_v2_5.dylib"
        "$REPO_ROOT/agx_fix/build/libagx_fix_v2_3.dylib"
        "$REPO_ROOT/agx_fix/build/libagx_fix_v2.dylib"
        "$REPO_ROOT/agx_fix/build/libagx_fix.dylib"
    )
    for candidate in "${AGX_FIX_CANDIDATES[@]}"; do
        if [ -f "$candidate" ]; then
            AGX_FIX="$candidate"
            break
        fi
    done

    if [ -n "$AGX_FIX" ]; then
        export DYLD_INSERT_LIBRARIES="$AGX_FIX"
        echo "AGX fix loaded: $AGX_FIX"
    else
        echo "WARNING: AGX fix not found (build: cd agx_fix && make)" >&2
    fi
fi

# Record start time
START_TIME=$(date +%s)
START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
touch "$CRASH_DIR/.run_start_$START_TIME"

echo "Running: $@"
echo "Started: $START_ISO"
echo "---"

# Run command, capture exit code
set +e
"$@"
EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "---"
echo "Exit code: $EXIT_CODE"
echo "Duration: ${DURATION}s"

# Check for crash signals
# 128+N means killed by signal N
# 139 = 128+11 = SIGSEGV
# 134 = 128+6 = SIGABRT
# 136 = 128+8 = SIGFPE
if [ $EXIT_CODE -ge 128 ]; then
    SIGNAL=$((EXIT_CODE - 128))
    case $SIGNAL in
        11) SIGNAL_NAME="SIGSEGV (Segmentation fault)" ;;
        6)  SIGNAL_NAME="SIGABRT (Abort)" ;;
        8)  SIGNAL_NAME="SIGFPE (Floating point exception)" ;;
        9)  SIGNAL_NAME="SIGKILL (Killed)" ;;
        *)  SIGNAL_NAME="Signal $SIGNAL" ;;
    esac

    echo ""
    echo "========================================"
    echo "CRASH DETECTED"
    echo "========================================"
    echo "Signal: $SIGNAL_NAME"
    echo ""

    # Wait for macOS to write crash log (usually takes 1-3 seconds)
    echo "Waiting for crash log..."
    sleep 3

    # Find the crash log
    CRASH_FILE=""
    for attempt in 1 2 3 4 5; do
        # Look for crash files created after we started
        CRASH_FILE=$(find "$USER_CRASH_DIR" \( -name "*.ips" -o -name "*.crash" \) -newer "$CRASH_DIR/.run_start_$START_TIME" 2>/dev/null | head -1)
        if [ -n "$CRASH_FILE" ] && [ -f "$CRASH_FILE" ]; then
            break
        fi
        sleep 1
    done

    if [ -n "$CRASH_FILE" ] && [ -f "$CRASH_FILE" ]; then
        echo "Found crash log: $CRASH_FILE"
        echo ""

        # Copy to our crash directory
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        DEST="$CRASH_DIR/${TIMESTAMP}_$(basename "$CRASH_FILE")"
        cp "$CRASH_FILE" "$DEST"
        echo "$DEST" > "$LATEST_CRASH"

        # Extract and display key information (supports JSON .ips and text .crash formats)
        echo "======== CRASH SUMMARY ========"
        python3 - "$CRASH_FILE" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    text = path.read_text(errors="replace")
except Exception as e:
    print(f"ERROR: Failed to read crash log: {e}")
    raise SystemExit(0)


def parse_json_sequence(content: str):
    dec = json.JSONDecoder()
    idx = 0
    objs = []
    n = len(content)
    while idx < n:
        while idx < n and content[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = dec.raw_decode(content, idx)
        except json.JSONDecodeError:
            break
        objs.append(obj)
        idx = end
    return objs


def extract_ips(report: dict):
    proc = report.get("procName") or report.get("app_name") or report.get("name") or "unknown"
    exc = report.get("exception") if isinstance(report.get("exception"), dict) else {}
    exc_type = exc.get("type") or "unknown"
    subtype = exc.get("subtype") or ""
    exc_str = f"{exc_type} - {subtype}" if subtype else str(exc_type)

    fault = ""
    for candidate in (subtype, exc.get("codes"), report.get("vmRegionInfo")):
        if isinstance(candidate, str):
            m = re.search(r"0x[0-9a-fA-F]+", candidate)
            if m:
                fault = m.group(0)
                break

    crashed_in = "unknown"
    ft = report.get("faultingThread")
    threads = report.get("threads")
    stack = []
    if isinstance(ft, int) and isinstance(threads, list) and 0 <= ft < len(threads):
        t = threads[ft]
        if isinstance(t, dict) and isinstance(t.get("frames"), list):
            frames = t["frames"]
            if frames and isinstance(frames[0], dict):
                f0 = frames[0]
                sym = f0.get("symbol") or "unknown"
                sym_loc = f0.get("symbolLocation")
                img = f0.get("imageIndex")
                parts = []
                if img is not None:
                    parts.append(f"[{img}]")
                parts.append(str(sym))
                if sym_loc is not None:
                    parts.append(f"+ {sym_loc}")
                crashed_in = " ".join(parts)
            for fr in frames[:10]:
                if not isinstance(fr, dict):
                    continue
                sym = fr.get("symbol") or "???"
                loc = fr.get("symbolLocation")
                img = fr.get("imageIndex")
                parts = []
                if img is not None:
                    parts.append(f"[{img}]")
                parts.append(str(sym))
                if loc is not None:
                    parts.append(f"+ {loc}")
                stack.append(" ".join(parts))

    return proc, exc_str, fault, crashed_in, ft, stack


report = None
for obj in reversed(parse_json_sequence(text)):
    if isinstance(obj, dict) and ("exception" in obj or "threads" in obj or "procName" in obj):
        report = obj
        break

if report:
    proc, exc_str, fault, crashed_in, ft, stack = extract_ips(report)
    print(f"Process: {proc}")
    print(f"Exception: {exc_str}")
    print(f"Fault Address: {fault or 'unknown'}")
    print(f"Faulting Thread: {ft if ft is not None else 'unknown'}")
    print(f"Crashed In: {crashed_in}")
    print("")
    print("Crash Stack:")
    if stack:
        for line in stack:
            print(f"  {line}")
    else:
        print("  (no backtrace found)")
else:
    # Fallback for plain-text .crash formats.
    for line in text.splitlines():
        if re.match(r"^(Process|Path|Exception Type|Exception Codes|Crashed Thread|Thread [0-9]+ Crashed):", line):
            print(line)
PY

        echo "==============================="
        echo ""
        echo "Full crash log: $DEST"

        # Update crash summary JSON
        export CRASH_CAPTURE_COMMAND="$*"
        python3 << EOF
import json
import os
import re

crash_file = "$CRASH_FILE"
dest_file = "$DEST"
summary_file = "$CRASH_SUMMARY"
command = os.environ.get("CRASH_CAPTURE_COMMAND", "").strip()

# Extract info from crash file
try:
    with open(crash_file, 'r') as f:
        content = f.read()
except:
    content = ""

def parse_json_sequence(text: str):
    dec = json.JSONDecoder()
    idx = 0
    objs = []
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = dec.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        objs.append(obj)
        idx = end
    return objs

def parse_ips_report(report: dict) -> dict:
    proc = report.get("procName") or report.get("app_name") or report.get("name") or "unknown"
    exc = report.get("exception") if isinstance(report.get("exception"), dict) else {}
    exc_type = exc.get("type") or "unknown"
    subtype = exc.get("subtype") or ""
    exc_str = f"{exc_type} - {subtype}" if subtype else str(exc_type)

    fault = "unknown"
    for candidate in (subtype, exc.get("codes"), report.get("vmRegionInfo")):
        if isinstance(candidate, str):
            m = re.search(r"0x[0-9a-fA-F]+", candidate)
            if m:
                fault = m.group(0)
                break

    crashed_in = "unknown"
    ft = report.get("faultingThread")
    threads = report.get("threads")
    if isinstance(ft, int) and isinstance(threads, list) and 0 <= ft < len(threads):
        t = threads[ft]
        if isinstance(t, dict):
            frames = t.get("frames")
            if isinstance(frames, list) and frames and isinstance(frames[0], dict):
                f0 = frames[0]
                sym = f0.get("symbol") or "unknown"
                sym_loc = f0.get("symbolLocation")
                img = f0.get("imageIndex")
                parts = []
                if img is not None:
                    parts.append(f"[{img}]")
                parts.append(str(sym))
                if sym_loc is not None:
                    parts.append(f"+ {sym_loc}")
                crashed_in = " ".join(parts)

    return {
        "process": str(proc) if proc else "unknown",
        "exception": str(exc_str) if exc_str else "unknown",
        "fault_address": fault,
        "crashed_in": crashed_in,
    }

info = None
report = None
for obj in reversed(parse_json_sequence(content)):
    if isinstance(obj, dict) and ("exception" in obj or "threads" in obj or "procName" in obj):
        report = obj
        break

if report is not None:
    info = parse_ips_report(report)
else:
    process = re.search(r'Process:\s+(\S+)', content)
    exception = re.search(r'Exception Type:\s+(.+)', content)
    fault = re.search(r'Exception Codes:.*?(0x[0-9a-fA-F]+)', content)
    crashed_thread = re.search(r'Thread \d+ Crashed[^\n]*\n\s*\d+\s+(\S+)\s+(.+)', content)
    info = {
        "process": process.group(1) if process else "unknown",
        "exception": exception.group(1).strip() if exception else "unknown",
        "fault_address": fault.group(1) if fault else "unknown",
        "crashed_in": f"{crashed_thread.group(1)} {crashed_thread.group(2)}" if crashed_thread else "unknown"
    }

# Load or create summary
try:
    with open(summary_file, 'r') as f:
        data = json.load(f)
except:
    data = {"crashes": [], "last_check": None}

# Add new crash
new_crash = {
    "timestamp": "$START_ISO",
    "file": dest_file,
    "command": command,
    "exit_code": $EXIT_CODE,
    "signal": "$SIGNAL_NAME",
    "info": info
}

data["crashes"].insert(0, new_crash)
data["crashes"] = data["crashes"][:50]  # Keep last 50
data["last_check"] = "$START_ISO"

with open(summary_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Crash logged to: {summary_file}")
EOF
        unset CRASH_CAPTURE_COMMAND

    else
        echo "WARNING: Could not find crash log (macOS may not have generated one)"
        echo "Check manually: $USER_CRASH_DIR"
    fi
fi

# Cleanup temp marker
rm -f "$CRASH_DIR/.run_start_$START_TIME"

exit $EXIT_CODE
