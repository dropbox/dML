#!/bin/bash
# Run the Voice TTS daemon with auto-restart
# Usage: ./scripts/run_daemon.sh [config] [--background]
#
# Examples:
#   ./scripts/run_daemon.sh                                    # Default English
#   ./scripts/run_daemon.sh kokoro-mps-ja.yaml                # Japanese
#   ./scripts/run_daemon.sh kokoro-mps-en2ja.yaml --background # Background with translation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Defaults
CONFIG="kokoro-mps-en.yaml"
BACKGROUND=false
LOG_FILE="/tmp/voice-daemon.log"
PID_FILE="/tmp/voice-daemon.pid"
MAX_RESTARTS=10
RESTART_DELAY=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --background|-b)
            BACKGROUND=true
            shift
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [config.yaml] [options]"
            echo ""
            echo "Run the Voice TTS daemon with auto-restart on failure."
            echo ""
            echo "Options:"
            echo "  --background, -b  Run in background"
            echo "  --log <file>      Log file (default: /tmp/voice-daemon.log)"
            echo "  -h, --help        Show this help"
            echo ""
            echo "Available configs:"
            ls -1 "$PROJECT_ROOT/stream-tts-cpp/config/"*.yaml | xargs -n1 basename | sed 's/^/  /'
            echo ""
            echo "Examples:"
            echo "  $0                           # English TTS"
            echo "  $0 kokoro-mps-ja.yaml        # Japanese TTS"
            echo "  $0 kokoro-mps-en2ja.yaml -b  # EN->JA translation, background"
            exit 0
            ;;
        *.yaml)
            CONFIG="$1"
            shift
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Resolve config path
if [[ ! "$CONFIG" = /* ]]; then
    CONFIG_PATH="$PROJECT_ROOT/stream-tts-cpp/config/$CONFIG"
else
    CONFIG_PATH="$CONFIG"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    error "Config not found: $CONFIG_PATH"
fi

BINARY="$PROJECT_ROOT/stream-tts-cpp/build/stream-tts-cpp"
if [[ ! -x "$BINARY" ]]; then
    error "Binary not found: $BINARY (run setup_laptop.sh first)"
fi

# Check if daemon already running
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        warn "Daemon already running (PID $OLD_PID)"
        echo "Stop it with: $BINARY --stop"
        echo "Or kill it with: kill $OLD_PID"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Function to run daemon with auto-restart
run_daemon() {
    local restart_count=0

    while [[ $restart_count -lt $MAX_RESTARTS ]]; do
        info "Starting daemon (attempt $((restart_count + 1))/$MAX_RESTARTS)..."
        info "Config: $CONFIG"
        info "Log: $LOG_FILE"
        echo ""

        # Run daemon
        "$BINARY" --daemon "$CONFIG_PATH" --log-file "$LOG_FILE" &
        DAEMON_PID=$!
        echo "$DAEMON_PID" > "$PID_FILE"

        info "Daemon started (PID $DAEMON_PID)"

        # Wait for daemon to exit
        wait "$DAEMON_PID"
        EXIT_CODE=$?

        rm -f "$PID_FILE"

        if [[ $EXIT_CODE -eq 0 ]]; then
            info "Daemon exited normally"
            break
        else
            restart_count=$((restart_count + 1))
            warn "Daemon crashed (exit code $EXIT_CODE)"

            if [[ $restart_count -lt $MAX_RESTARTS ]]; then
                warn "Restarting in ${RESTART_DELAY}s..."
                sleep "$RESTART_DELAY"
            else
                error "Max restarts ($MAX_RESTARTS) reached. Giving up."
            fi
        fi
    done
}

# Cleanup on exit
cleanup() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            info "Stopping daemon (PID $PID)..."
            kill "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
}
trap cleanup EXIT INT TERM

# Run
if [[ "$BACKGROUND" == "true" ]]; then
    info "Starting daemon in background..."
    nohup "$0" "$CONFIG" > /dev/null 2>&1 &
    sleep 2

    if [[ -f "$PID_FILE" ]]; then
        info "Daemon running in background (PID $(cat "$PID_FILE"))"
        echo ""
        echo "Commands:"
        echo "  $BINARY --speak \"Hello world\""
        echo "  $BINARY --status"
        echo "  $BINARY --stop"
        echo ""
        echo "Logs: tail -f $LOG_FILE"
    else
        error "Failed to start daemon in background"
    fi
else
    run_daemon
fi
