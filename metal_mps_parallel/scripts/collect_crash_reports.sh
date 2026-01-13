#!/bin/bash
# Collect and parse macOS crash reports for Python/PyTorch processes
#
# Usage: ./scripts/collect_crash_reports.sh [--watch] [--last N]
#
# Options:
#   --watch    Watch for new crash reports in real-time
#   --last N   Show last N crash reports (default: 5)

CRASH_DIR="$HOME/Library/Logs/DiagnosticReports"
OUTPUT_DIR="reports/crash_reports"

mkdir -p "$OUTPUT_DIR"

show_last() {
    local count=${1:-5}
    echo "=== Last $count Python/PyTorch crash reports ==="

    # Find recent crash reports for Python
    ls -t "$CRASH_DIR"/Python*.ips "$CRASH_DIR"/python*.ips 2>/dev/null | head -n "$count" | while read -r file; do
        if [ -f "$file" ]; then
            echo ""
            echo "=========================================="
            echo "File: $(basename "$file")"
            echo "Time: $(stat -f '%Sm' "$file")"
            echo "=========================================="

	            # Extract key info using plutil or jq
	            if command -v jq &>/dev/null; then
	                # Parse JSON crash report.
	                #
	                # NOTE: .ips reports may contain multiple JSON objects concatenated; `jq -s`
	                # reads them all and we parse the last (full) object.
	                jq -s -r '
	                    .[-1] as $r |
	                    ($r.faultingThread // -1) as $ft |
	                    "Exception: \($r.exception.type // "unknown") - \($r.exception.subtype // "")",
	                    "Crashed Thread: \(if $ft >= 0 then $ft else "unknown" end)",
	                    "",
	                    "Crash Stack:",
	                    (if ($ft >= 0) and ($ft < ($r.threads | length)) then
	                        ($r.threads[$ft].frames[:10] | .[] | "  \(.imageIndex). \(.symbol // "???") + \(.symbolLocation // 0)")
	                     else
	                        "  (no faulting thread backtrace found)"
	                     end)
	                ' "$file" 2>/dev/null || cat "$file" | head -100
	            else
	                # Fallback: extract key lines
	                grep -E "Exception Type:|Crashed Thread:|^[0-9]+ +[a-zA-Z]" "$file" | head -30
	            fi
        fi
    done
}

watch_crashes() {
    echo "Watching for new crash reports in $CRASH_DIR..."
    echo "Press Ctrl+C to stop"

    fswatch -0 "$CRASH_DIR" 2>/dev/null | while read -d "" event; do
        if [[ "$event" == *Python*.ips ]] || [[ "$event" == *python*.ips ]]; then
            echo ""
            echo "=== NEW CRASH DETECTED ==="
            echo "File: $event"
            sleep 1  # Wait for file to be written
            show_last 1
        fi
    done

    # Fallback if fswatch not available
    if ! command -v fswatch &>/dev/null; then
        echo "fswatch not installed. Install with: brew install fswatch"
        echo "Falling back to polling..."
        local last_count=$(ls "$CRASH_DIR"/Python*.ips 2>/dev/null | wc -l)
        while true; do
            sleep 5
            local new_count=$(ls "$CRASH_DIR"/Python*.ips 2>/dev/null | wc -l)
            if [ "$new_count" -gt "$last_count" ]; then
                echo "=== NEW CRASH DETECTED ==="
                show_last 1
                last_count=$new_count
            fi
        done
    fi
}

copy_latest() {
    local latest=$(ls -t "$CRASH_DIR"/Python*.ips 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local dest="$OUTPUT_DIR/crash_${timestamp}.json"
        cp "$latest" "$dest"
        echo "Copied to: $dest"
    fi
}

parse_crash_summary() {
    local file="$1"
	    echo "=== Crash Summary ==="

	    if command -v jq &>/dev/null && [ -f "$file" ]; then
	        jq -s -r '
	            .[-1] as $r |
	            ($r.faultingThread // -1) as $ft |
	            "Process: \($r.procName // "unknown")",
	            "Date: \($r.captureTime // "unknown")",
	            "Exception: \($r.exception.type // "unknown") - \($r.exception.subtype // "")",
	            "Faulting Thread: \(if $ft >= 0 then $ft else "unknown" end)",
	            "",
	            "Crashed in:",
	            (if ($ft >= 0) and ($ft < ($r.threads | length)) then
	                ($r.threads[$ft].frames[:5] | .[] |
	                    "  [\(.imageIndex)] \(.symbol // "unknown") + \(.symbolLocation)")
	             else
	                "  (no faulting thread backtrace found)"
	             end)
	        ' "$file"
	    fi
	}

# Main
case "${1:-}" in
    --watch)
        watch_crashes
        ;;
    --last)
        show_last "${2:-5}"
        ;;
    --copy)
        copy_latest
        ;;
    --parse)
        parse_crash_summary "${2:-}"
        ;;
    *)
        echo "Usage: $0 [--watch] [--last N] [--copy] [--parse FILE]"
        echo ""
        show_last 3
        ;;
esac
