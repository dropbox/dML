#!/bin/bash
# start_voice_daemon.sh - Start voice TTS daemon and wait for readiness
# Worker #486: Created for daemon-pipe integration
# Copyright 2025 Andrew Yates. All rights reserved.
#
# Usage:
#   ./scripts/start_voice_daemon.sh                    # Start with default config
#   ./scripts/start_voice_daemon.sh config/custom.yaml # Start with custom config
#   ./scripts/start_voice_daemon.sh --stop             # Stop running daemon
#   ./scripts/start_voice_daemon.sh --status           # Check daemon status

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TTS_BINARY="$SCRIPT_DIR/../stream-tts-cpp/build/stream-tts-cpp"
TTS_CONFIG="${1:-$SCRIPT_DIR/../stream-tts-cpp/config/default.yaml}"
DAEMON_SOCKET="/tmp/stream-tts.sock"
DAEMON_PID_FILE="/tmp/stream-tts.pid"

# Handle --stop command
if [ "$1" = "--stop" ]; then
    if [ -S "$DAEMON_SOCKET" ]; then
        echo "Stopping daemon..."
        "$TTS_BINARY" --stop
        sleep 1
        if [ -S "$DAEMON_SOCKET" ]; then
            echo "Warning: Socket still exists, force cleanup"
            rm -f "$DAEMON_SOCKET"
        fi
        rm -f "$DAEMON_PID_FILE"
        echo "Daemon stopped"
    else
        echo "No daemon running"
    fi
    exit 0
fi

# Handle --status command
if [ "$1" = "--status" ]; then
    if [ -S "$DAEMON_SOCKET" ]; then
        echo "Daemon running:"
        "$TTS_BINARY" --status
    else
        echo "No daemon running (socket not found: $DAEMON_SOCKET)"
        exit 1
    fi
    exit 0
fi

# Check if binary exists
if [ ! -f "$TTS_BINARY" ]; then
    echo "ERROR: TTS binary not found at: $TTS_BINARY"
    echo "Build it first:"
    echo "  cd stream-tts-cpp && ./build.sh"
    exit 1
fi

# Check if daemon already running
if [ -S "$DAEMON_SOCKET" ]; then
    echo "Daemon already running at $DAEMON_SOCKET"
    "$TTS_BINARY" --status | head -5
    exit 0
fi

# Start daemon
echo "Starting voice daemon..."
echo "  Binary: $TTS_BINARY"
echo "  Config: $TTS_CONFIG"

# Note: PYTORCH_ENABLE_MPS_FALLBACK no longer needed with PyTorch 2.9.1+
# All MPS operations (including torch.angle) now run natively on Metal GPU

if [ -f "$TTS_CONFIG" ]; then
    "$TTS_BINARY" --daemon "$TTS_CONFIG" &
else
    echo "  (Config not found, using defaults)"
    "$TTS_BINARY" --daemon &
fi
DAEMON_PID=$!
echo $DAEMON_PID > "$DAEMON_PID_FILE"

# Wait for socket to appear (max 60s - models take time to load)
echo -n "Waiting for daemon to be ready"
for i in {1..60}; do
    if [ -S "$DAEMON_SOCKET" ]; then
        echo ""
        echo "Daemon ready (PID $DAEMON_PID)"
        echo ""
        "$TTS_BINARY" --status | head -10
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "ERROR: Daemon failed to start within 60 seconds"
echo "Check logs for errors. Killing process $DAEMON_PID..."
kill $DAEMON_PID 2>/dev/null
rm -f "$DAEMON_PID_FILE"
exit 1
