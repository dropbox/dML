#!/bin/bash
# Crash Monitor - Captures crash logs for AI workers
#
# This script monitors for crash reports and copies them to a known location
# where AI workers can find them.
#
# Usage:
#   ./crash_monitor.sh start   - Start monitoring in background
#   ./crash_monitor.sh stop    - Stop monitoring
#   ./crash_monitor.sh status  - Check if monitoring
#   ./crash_monitor.sh check   - Check for recent crashes (manual)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CRASH_DIR="$REPO_ROOT/crash_logs"
PID_FILE="$REPO_ROOT/.crash_monitor.pid"
LATEST_CRASH="$CRASH_DIR/latest_crash.txt"
CRASH_SUMMARY="$CRASH_DIR/crash_summary.json"

# macOS crash log locations
SYSTEM_CRASH_DIR="/Library/Logs/DiagnosticReports"
USER_CRASH_DIR="$HOME/Library/Logs/DiagnosticReports"

mkdir -p "$CRASH_DIR"

# Initialize crash summary if it doesn't exist
if [ ! -f "$CRASH_SUMMARY" ]; then
    echo '{"crashes": [], "last_check": null}' > "$CRASH_SUMMARY"
fi

check_for_crashes() {
    local since="$1"  # Unix timestamp
    local found=0

    # Find crash reports newer than $since
    for dir in "$USER_CRASH_DIR" "$SYSTEM_CRASH_DIR"; do
        if [ -d "$dir" ]; then
            # Look for Python, AGX, and Metal crashes
            while IFS= read -r crash_file; do
                if [ -f "$crash_file" ]; then
                    file_time=$(stat -f %m "$crash_file" 2>/dev/null || echo 0)
                    if [ "$file_time" -gt "$since" ]; then
                        # Check if it's relevant (Python, AGX, Metal)
                        if grep -q -E "Python|AGX|Metal|mps|torch" "$crash_file" 2>/dev/null; then
                            echo "$crash_file"
                            found=1
                        fi
                    fi
                fi
            done < <(find "$dir" \( -name "*.ips" -o -name "*.crash" \) 2>/dev/null)
        fi
    done

    [ "$found" -eq 1 ]
}

copy_crash_log() {
    local crash_file="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local basename=$(basename "$crash_file")
    local dest="$CRASH_DIR/${timestamp}_${basename}"

    cp "$crash_file" "$dest"
    echo "$dest"
}

already_captured() {
    local crash_file="$1"
    local base
    base="$(basename "$crash_file")"

    # We store captures as "<timestamp>_<original_basename>".
    # Avoid duplicating the same crash report across monitor + manual scans.
    find "$CRASH_DIR" -maxdepth 1 -name "*_${base}" -print -quit 2>/dev/null | grep -q .
}

extract_crash_info() {
    local crash_file="$1"

    python3 - "$crash_file" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    text = path.read_text(errors="replace")
except Exception:
    print('{"process":"unknown","exception":"unknown","fault_address":"unknown","crashed_in":"unknown"}')
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


def extract_from_ips(report: dict):
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
        "fault_address": fault or "unknown",
        "crashed_in": crashed_in or "unknown",
    }


report = None
for obj in reversed(parse_json_sequence(text)):
    if isinstance(obj, dict) and ("exception" in obj or "threads" in obj or "procName" in obj):
        report = obj
        break

if report:
    info = extract_from_ips(report)
else:
    proc = re.search(r"^Process:\s+(\S+)", text, re.MULTILINE)
    exc = re.search(r"^Exception Type:\s+(.+)$", text, re.MULTILINE)
    codes = re.search(r"^Exception Codes:.*?(0x[0-9a-fA-F]+)", text, re.MULTILINE)
    crashed = re.search(r"^Thread \d+ Crashed[^\n]*\n\s*\d+\s+(\S+)\s+(.+)$", text, re.MULTILINE)
    info = {
        "process": proc.group(1) if proc else "unknown",
        "exception": exc.group(1).strip() if exc else "unknown",
        "fault_address": codes.group(1) if codes else "unknown",
        "crashed_in": f"{crashed.group(1)} {crashed.group(2)}".strip() if crashed else "unknown",
    }

print(json.dumps(info))
PY
}

update_latest() {
    local crash_file="$1"
    local dest="$2"

    # Update latest crash pointer
    echo "$dest" > "$LATEST_CRASH"

    # Update crash summary JSON
    local info=$(extract_crash_info "$crash_file")
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Use Python to update JSON (more reliable than jq for complex updates)
    python3 << EOF
import json
import os

summary_file = "$CRASH_SUMMARY"
try:
    with open(summary_file, 'r') as f:
        data = json.load(f)
except:
    data = {"crashes": [], "last_check": None}

new_crash = {
    "timestamp": "$timestamp",
    "file": "$dest",
    "info": $info
}

data["crashes"].insert(0, new_crash)
data["crashes"] = data["crashes"][:20]  # Keep last 20
data["last_check"] = "$timestamp"

with open(summary_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Crash logged: {new_crash['info']}")
EOF
}

monitor_loop() {
    echo "Crash monitor started at $(date)"
    echo "Monitoring: $USER_CRASH_DIR, $SYSTEM_CRASH_DIR"
    echo "Output: $CRASH_DIR"

    local last_check=$(date +%s)

    while true; do
        sleep 5  # Check every 5 seconds

        # Find new crashes since last check
        for dir in "$USER_CRASH_DIR" "$SYSTEM_CRASH_DIR"; do
            if [ -d "$dir" ]; then
                find "$dir" \( -name "*.ips" -o -name "*.crash" \) -newer "$CRASH_DIR/.last_check" 2>/dev/null | while read -r crash_file; do
                    if [ -f "$crash_file" ]; then
                        # Check if relevant
                        if grep -q -E "Python|AGX|Metal|mps|torch" "$crash_file" 2>/dev/null; then
                            if already_captured "$crash_file"; then
                                continue
                            fi
                            echo "$(date): Found crash: $crash_file"
                            dest=$(copy_crash_log "$crash_file")
                            update_latest "$crash_file" "$dest"
                        fi
                    fi
                done
            fi
        done

        # Update last check timestamp
        touch "$CRASH_DIR/.last_check"
    done
}

case "${1:-status}" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Crash monitor already running (PID $(cat "$PID_FILE"))"
            exit 0
        fi

        # Initialize last check marker
        touch "$CRASH_DIR/.last_check"

        # Start in background
        nohup "$0" _monitor > "$CRASH_DIR/monitor.log" 2>&1 &
        echo $! > "$PID_FILE"
        echo "Crash monitor started (PID $!)"
        echo "Logs: $CRASH_DIR/monitor.log"
        ;;

    _monitor)
        # Internal: run the monitor loop
        monitor_loop
        ;;

    stop)
        if [ -f "$PID_FILE" ]; then
            pid=$(cat "$PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                echo "Crash monitor stopped (was PID $pid)"
            else
                echo "Crash monitor not running (stale PID file)"
            fi
            rm -f "$PID_FILE"
        else
            echo "Crash monitor not running"
        fi
        ;;

    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Crash monitor: RUNNING (PID $(cat "$PID_FILE"))"
        else
            echo "Crash monitor: STOPPED"
        fi

        if [ -f "$LATEST_CRASH" ]; then
            echo "Latest crash: $(cat "$LATEST_CRASH")"
        else
            echo "Latest crash: none"
        fi

        if [ -f "$CRASH_SUMMARY" ]; then
            count=$(python3 -c "import json; print(len(json.load(open('$CRASH_SUMMARY'))['crashes']))" 2>/dev/null || echo 0)
            echo "Total crashes logged: $count"
        fi
        ;;

    check)
        # Manual check for crashes in last hour
        echo "Checking for crashes in last hour..."
        found=0
        for dir in "$USER_CRASH_DIR" "$SYSTEM_CRASH_DIR"; do
            if [ -d "$dir" ]; then
                while IFS= read -r crash_file; do
                    if grep -q -E "Python|AGX|Metal|mps|torch" "$crash_file" 2>/dev/null; then
                        echo "Found: $crash_file"
                        if ! already_captured "$crash_file"; then
                            dest=$(copy_crash_log "$crash_file")
                            update_latest "$crash_file" "$dest"
                        fi
                        found=1
                    fi
                done < <(find "$dir" \( -name "*.ips" -o -name "*.crash" \) -mmin -60 2>/dev/null)
            fi
        done

        if [ "$found" -eq 0 ]; then
            echo "No relevant crashes found in last hour"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|status|check}"
        exit 1
        ;;
esac
