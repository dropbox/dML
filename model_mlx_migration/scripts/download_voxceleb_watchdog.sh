#!/bin/bash
# VoxCeleb Download Watchdog
# Monitors the download and restarts if it stalls
#
# Usage: ./scripts/download_voxceleb_watchdog.sh
#        ./scripts/download_voxceleb_watchdog.sh --background
#
# The watchdog will:
# 1. Monitor download progress every 60 seconds
# 2. If no progress for 5 minutes, kill and restart the download
# 3. Log all activity to logs/voxceleb_watchdog.log

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
DOWNLOAD_LOG="$LOG_DIR/voxceleb_download.log"
WATCHDOG_LOG="$LOG_DIR/voxceleb_watchdog.log"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_voxceleb_hf.py"
STATE_FILE="$LOG_DIR/.voxceleb_watchdog_state"

# Configuration
CHECK_INTERVAL=60        # Check every 60 seconds
STALL_THRESHOLD=300      # 5 minutes without progress = stall
MAX_RESTARTS=10          # Maximum number of restarts

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WATCHDOG_LOG"
}

get_download_size() {
    du -s "$PROJECT_DIR/data/voxceleb_hf" 2>/dev/null | cut -f1 || echo "0"
}

get_download_pid() {
    pgrep -f "download_voxceleb_hf.py" 2>/dev/null | head -1 || echo ""
}

is_download_running() {
    local pid=$(get_download_pid)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_download() {
    log "Starting VoxCeleb download..."
    nohup python3 "$DOWNLOAD_SCRIPT" --extract >> "$DOWNLOAD_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$STATE_FILE.pid"
    log "Download started with PID: $pid"
    sleep 5  # Give it time to start
}

kill_download() {
    local pid=$(get_download_pid)
    if [ -n "$pid" ]; then
        log "Killing stalled download (PID: $pid)..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 2
    fi
}

run_watchdog() {
    local last_size=0
    local last_progress_time=$(date +%s)
    local restart_count=0

    log "=== VoxCeleb Download Watchdog Started ==="
    log "Check interval: ${CHECK_INTERVAL}s"
    log "Stall threshold: ${STALL_THRESHOLD}s"
    log "Max restarts: $MAX_RESTARTS"

    # Start download if not running
    if ! is_download_running; then
        start_download
    fi

    while true; do
        sleep "$CHECK_INTERVAL"

        local current_size=$(get_download_size)
        local current_time=$(date +%s)

        # Check if download is running
        if ! is_download_running; then
            log "Download process not running!"

            # Check if download completed
            if [ "$current_size" -gt 100000000 ]; then  # > 100GB
                log "Download appears complete (${current_size}KB)"

                # Verify we have audio files
                local wav_count=$(find "$PROJECT_DIR/data/voxceleb_hf" -name "*.wav" 2>/dev/null | wc -l)
                local m4a_count=$(find "$PROJECT_DIR/data/voxceleb_hf" -name "*.m4a" 2>/dev/null | wc -l)

                if [ "$wav_count" -gt 0 ] || [ "$m4a_count" -gt 0 ]; then
                    log "Found $wav_count WAV files, $m4a_count M4A files"
                    log "=== Download Complete ==="
                    exit 0
                fi
            fi

            # Restart download
            restart_count=$((restart_count + 1))
            if [ "$restart_count" -gt "$MAX_RESTARTS" ]; then
                log "ERROR: Exceeded max restarts ($MAX_RESTARTS)"
                exit 1
            fi

            log "Restart #$restart_count"
            start_download
            last_progress_time=$current_time
            continue
        fi

        # Check for progress
        if [ "$current_size" -gt "$last_size" ]; then
            local delta=$((current_size - last_size))
            local delta_mb=$((delta / 1024))
            log "Progress: ${current_size}KB (+${delta_mb}MB)"
            last_size=$current_size
            last_progress_time=$current_time
        else
            local stall_time=$((current_time - last_progress_time))
            if [ "$stall_time" -gt "$STALL_THRESHOLD" ]; then
                log "WARNING: Download stalled for ${stall_time}s"

                restart_count=$((restart_count + 1))
                if [ "$restart_count" -gt "$MAX_RESTARTS" ]; then
                    log "ERROR: Exceeded max restarts ($MAX_RESTARTS)"
                    exit 1
                fi

                kill_download
                log "Restart #$restart_count"
                start_download
                last_progress_time=$current_time
            else
                log "No progress for ${stall_time}s (threshold: ${STALL_THRESHOLD}s)"
            fi
        fi

        # Log current status
        local size_gb=$(echo "scale=2; $current_size / 1048576" | bc)
        log "Status: ${size_gb}GB downloaded, restarts: $restart_count"
    done
}

# Handle arguments
if [ "$1" = "--background" ]; then
    log "Starting watchdog in background..."
    nohup "$0" >> "$WATCHDOG_LOG" 2>&1 &
    echo "Watchdog started with PID: $!"
    echo "Monitor with: tail -f $WATCHDOG_LOG"
    exit 0
fi

# Run in foreground
run_watchdog
