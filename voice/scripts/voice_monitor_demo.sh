#!/bin/bash
# voice_monitor_demo.sh - Demo script for VoiceMonitor
# Watches Claude Code worker logs and speaks events in real-time
# Copyright 2025 Andrew Yates. All rights reserved.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TTS_BIN="$PROJECT_ROOT/stream-tts-cpp/build/stream-tts-cpp"
WORKER_LOGS="$PROJECT_ROOT/worker_logs"
CONFIG_FILE="$PROJECT_ROOT/stream-tts-cpp/config/kokoro-mps-en.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  VoiceMonitor Demo - Real-time Voice for Claude Code       ${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_usage() {
    echo -e "${GREEN}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${GREEN}Options:${NC}"
    echo "  -v, --verbosity N   Set verbosity level (0-3, default: 2)"
    echo "                        0 = errors only"
    echo "                        1 = errors + completions"
    echo "                        2 = errors + completions + progress"
    echo "                        3 = everything including tool uses"
    echo "  -d, --dir DIR       Watch directory (default: worker_logs)"
    echo "  -c, --config FILE   Config file (default: kokoro-mps-en.yaml)"
    echo "  -l, --lang LANG     Language: en, ja, zh, es, fr (default: en)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  # Watch with default settings (verbosity 2, English)"
    echo "  $0"
    echo ""
    echo "  # Watch with higher verbosity (includes tool uses)"
    echo "  $0 --verbosity 3"
    echo ""
    echo "  # Watch with errors only"
    echo "  $0 --verbosity 0"
    echo ""
    echo "  # Watch a different directory"
    echo "  $0 --dir /path/to/logs"
    echo ""
    echo "  # Use Japanese voice"
    echo "  $0 --lang ja --config config/kokoro-mps-ja.yaml"
    echo ""
}

# Default values
VERBOSITY=2
LANG="en"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbosity)
            VERBOSITY="$2"
            shift 2
            ;;
        -d|--dir)
            WORKER_LOGS="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -l|--lang)
            LANG="$2"
            # Auto-select config based on language
            case $LANG in
                ja) CONFIG_FILE="$PROJECT_ROOT/stream-tts-cpp/config/kokoro-mps-ja.yaml" ;;
                zh) CONFIG_FILE="$PROJECT_ROOT/stream-tts-cpp/config/kokoro-mps-zh.yaml" ;;
                es) CONFIG_FILE="$PROJECT_ROOT/stream-tts-cpp/config/kokoro-mps-es.yaml" ;;
                fr) CONFIG_FILE="$PROJECT_ROOT/stream-tts-cpp/config/kokoro-mps-fr.yaml" ;;
                *) CONFIG_FILE="$PROJECT_ROOT/stream-tts-cpp/config/kokoro-mps-en.yaml" ;;
            esac
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

print_header

# Check if binary exists
if [[ ! -x "$TTS_BIN" ]]; then
    echo -e "${RED}Error: TTS binary not found at $TTS_BIN${NC}"
    echo "Please build first: cd stream-tts-cpp/build && cmake --build . -j8"
    exit 1
fi

# Check if worker logs directory exists
if [[ ! -d "$WORKER_LOGS" ]]; then
    echo -e "${RED}Error: Worker logs directory not found at $WORKER_LOGS${NC}"
    echo "Make sure worker_logs/ exists with Claude Code log files."
    exit 1
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${YELLOW}Warning: Config file not found at $CONFIG_FILE${NC}"
    echo "Using default config."
    CONFIG_FILE=""
fi

# Count log files
LOG_COUNT=$(ls -1 "$WORKER_LOGS"/worker_iter_*.jsonl 2>/dev/null | wc -l | tr -d ' ')

echo -e "${GREEN}Configuration:${NC}"
echo "  Binary:      $TTS_BIN"
echo "  Watch Dir:   $WORKER_LOGS"
echo "  Config:      ${CONFIG_FILE:-"(default)"}"
echo "  Language:    $LANG"
echo "  Verbosity:   $VERBOSITY"
echo "  Log Files:   $LOG_COUNT"
echo ""

echo -e "${GREEN}Verbosity Levels:${NC}"
case $VERBOSITY in
    0) echo "  0: Errors only (critical issues)" ;;
    1) echo "  1: Errors + Completions (task status)" ;;
    2) echo "  2: Errors + Completions + Progress (recommended)" ;;
    3) echo "  3: Everything including tool uses (verbose)" ;;
esac
echo ""

echo -e "${YELLOW}Starting VoiceMonitor...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Build command
CMD="$TTS_BIN --watch $WORKER_LOGS --verbosity $VERBOSITY"
if [[ -n "$CONFIG_FILE" ]]; then
    CMD="$CMD $CONFIG_FILE"
fi

# Run the voice monitor
exec $CMD
