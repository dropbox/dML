#!/bin/bash
#
# sync_submodules.sh - Synchronize git submodules to recorded SHAs
#
# Usage:
#   ./scripts/sync_submodules.sh           # Check status and sync
#   ./scripts/sync_submodules.sh --check   # Check only, don't modify
#   ./scripts/sync_submodules.sh --update  # Update to latest upstream
#
# This script ensures submodules match the parent repo's recorded SHAs.
# It detects divergence and provides clear guidance on how to fix issues.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Submodules to track
SUBMODULES=(
    "stream-tts-cpp"
    "external/whisper.cpp"
)

print_header() {
    echo ""
    echo "========================================"
    echo "  Submodule Sync Script"
    echo "========================================"
    echo ""
}

check_submodule_status() {
    local submodule="$1"
    local status_output

    # Get submodule status
    # Format: [+- ]SHA path (branch)
    # + = checked out at different commit than recorded
    # - = not initialized (but may have content as nested repo)
    # U = merge conflict
    status_output=$(git submodule status "$submodule" 2>/dev/null || echo "ERROR")

    if [[ "$status_output" == "ERROR" ]]; then
        echo "error"
        return
    fi

    local prefix="${status_output:0:1}"
    case "$prefix" in
        "-")
            # Check if it's actually a nested repo with matching SHA
            local recorded=$(get_recorded_sha "$submodule")
            local current=$(get_current_sha "$submodule")
            if [[ "$recorded" == "$current" ]] && [[ "$current" != "none" ]]; then
                echo "ok_nested"
            else
                echo "not_initialized"
            fi
            ;;
        "+")
            echo "diverged"
            ;;
        "U")
            echo "conflict"
            ;;
        " ")
            echo "ok"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

get_recorded_sha() {
    local submodule="$1"
    git ls-tree HEAD "$submodule" | awk '{print $3}'
}

get_current_sha() {
    local submodule="$1"
    if [[ -d "$submodule/.git" ]] || [[ -f "$submodule/.git" ]]; then
        (cd "$submodule" && git rev-parse HEAD 2>/dev/null) || echo "none"
    else
        echo "none"
    fi
}

print_status() {
    local all_ok=true

    echo "Submodule Status:"
    echo "-----------------"
    printf "%-30s %-15s %-12s %s\n" "SUBMODULE" "STATUS" "RECORDED" "CURRENT"
    echo ""

    for submodule in "${SUBMODULES[@]}"; do
        local status=$(check_submodule_status "$submodule")
        local recorded=$(get_recorded_sha "$submodule")
        local current=$(get_current_sha "$submodule")

        # Truncate SHAs for display
        local recorded_short="${recorded:0:8}"
        local current_short="${current:0:8}"

        case "$status" in
            "ok")
                printf "%-30s ${GREEN}%-15s${NC} %-12s %s\n" "$submodule" "OK" "$recorded_short" "$current_short"
                ;;
            "ok_nested")
                printf "%-30s ${GREEN}%-15s${NC} %-12s %s\n" "$submodule" "OK (nested)" "$recorded_short" "$current_short"
                ;;
            "diverged")
                printf "%-30s ${YELLOW}%-15s${NC} %-12s %s\n" "$submodule" "DIVERGED" "$recorded_short" "$current_short"
                all_ok=false
                ;;
            "not_initialized")
                printf "%-30s ${RED}%-15s${NC} %-12s %s\n" "$submodule" "NOT INIT" "$recorded_short" "-"
                all_ok=false
                ;;
            "conflict")
                printf "%-30s ${RED}%-15s${NC} %-12s %s\n" "$submodule" "CONFLICT" "$recorded_short" "$current_short"
                all_ok=false
                ;;
            *)
                printf "%-30s ${RED}%-15s${NC} %-12s %s\n" "$submodule" "ERROR" "$recorded_short" "$current_short"
                all_ok=false
                ;;
        esac
    done

    echo ""

    if $all_ok; then
        echo -e "${GREEN}All submodules are in sync.${NC}"
        return 0
    else
        echo -e "${YELLOW}Some submodules need attention.${NC}"
        return 1
    fi
}

sync_submodules() {
    echo "Syncing submodules to recorded SHAs..."
    echo ""

    # Initialize any uninitialized submodules
    git submodule init

    # Update all submodules to recorded SHA
    git submodule update --recursive

    echo ""
    echo -e "${GREEN}Submodules synchronized.${NC}"
}

update_submodules() {
    echo "Updating submodules to latest upstream..."
    echo ""

    for submodule in "${SUBMODULES[@]}"; do
        echo "Updating $submodule..."
        (cd "$submodule" && git fetch origin && git checkout origin/main 2>/dev/null || git checkout origin/master) || true
    done

    echo ""
    echo -e "${YELLOW}Submodules updated to latest. Run 'git add' and 'git commit' to record new SHAs.${NC}"
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --check     Check status only, don't modify"
    echo "  --update    Update submodules to latest upstream (requires manual commit)"
    echo "  --help      Show this help message"
    echo ""
    echo "Without options, checks status and syncs to recorded SHAs if needed."
}

# Main
print_header

case "${1:-}" in
    --check)
        print_status
        ;;
    --update)
        print_status || true
        echo ""
        update_submodules
        echo ""
        print_status || true
        ;;
    --help|-h)
        show_help
        ;;
    "")
        if ! print_status; then
            echo ""
            sync_submodules
            echo ""
            print_status
        fi
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
