#!/bin/bash
# run_analysis.sh - Run static analysis on MPS source files
#
# This script runs:
# 1. Clang Thread Safety Analysis (-Wthread-safety)
# 2. (Optional) Facebook Infer racerd/starvation checkers
#
# Usage:
#   ./run_analysis.sh [--clang-only|--infer-only|--all]
#
# Requirements:
# - clang (usually from Xcode Command Line Tools)
# - infer (optional, install via: brew install infer)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MPS_SRC="$REPO_ROOT/pytorch-mps-fork/aten/src/ATen/mps"
OUTPUT_DIR="$SCRIPT_DIR/results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if MPS source exists
check_mps_source() {
    if [ ! -d "$MPS_SRC" ]; then
        log_error "MPS source not found at: $MPS_SRC"
        log_info "Run: git clone https://github.com/pytorch/pytorch.git pytorch-mps-fork"
        exit 1
    fi
}

# Run Clang Thread Safety Analysis
run_clang_tsa() {
    log_info "Running Clang Thread Safety Analysis..."

    local output_file="$OUTPUT_DIR/clang_tsa_$(date +%Y%m%d_%H%M%S).log"
    local warning_count=0

    # Find all header files to check
    local headers=$(find "$MPS_SRC" -name "*.h" -o -name "*.hpp" 2>/dev/null)

    if [ -z "$headers" ]; then
        log_warn "No header files found in $MPS_SRC"
        return
    fi

    # Clang flags for thread safety analysis
    local clang_flags=(
        -Wthread-safety
        -Wthread-safety-beta
        -Wthread-safety-negative
        -Wthread-safety-verbose
        -fsyntax-only
        -std=c++17
        -I"$MPS_SRC"
        -I"$REPO_ROOT/pytorch-mps-fork"
        -I"$REPO_ROOT/pytorch-mps-fork/aten/src"
        -I"$REPO_ROOT/pytorch-mps-fork/c10"
        -I"$SCRIPT_DIR"  # For clang_annotations.h
    )

    log_info "Output: $output_file"

    # Run clang on each header
    for header in $headers; do
        echo "=== Analyzing: $(basename $header) ===" >> "$output_file"

        # Note: This is a syntax check only. Full analysis requires
        # compile_commands.json from a complete PyTorch build
        clang++ "${clang_flags[@]}" "$header" 2>> "$output_file" || true

        # Count warnings
        local file_warnings=$(grep -c "warning:" "$output_file" 2>/dev/null || echo "0")
        warning_count=$((warning_count + file_warnings))
    done

    log_info "Thread Safety Analysis complete."
    log_info "Warnings found: $warning_count"
    log_info "Full output: $output_file"

    # Return warning count for summary
    echo "$warning_count"
}

# Run Facebook Infer (if available)
run_infer() {
    log_info "Running Facebook Infer..."

    # Check if infer is installed
    if ! command -v infer &> /dev/null; then
        log_warn "Facebook Infer not installed. Install with: brew install infer"
        return
    fi

    local output_dir="$OUTPUT_DIR/infer_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$output_dir"

    # Check for compile_commands.json
    local compile_db="$REPO_ROOT/pytorch-mps-fork/build/compile_commands.json"

    if [ ! -f "$compile_db" ]; then
        log_warn "compile_commands.json not found at: $compile_db"
        log_info "To generate it, build PyTorch with: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..."
        log_info "Skipping Infer analysis."
        return
    fi

    log_info "Using compile database: $compile_db"
    log_info "Output directory: $output_dir"

    # Run Infer's concurrency checkers
    cd "$REPO_ROOT/pytorch-mps-fork"

    infer run \
        --compilation-database "$compile_db" \
        --results-dir "$output_dir" \
        --racerd \
        --starvation \
        --keep-going \
        -- || true

    # Generate report
    infer report \
        --results-dir "$output_dir" \
        --issues-txt "$output_dir/issues.txt" \
        --issues-json "$output_dir/issues.json" \
        2>/dev/null || true

    if [ -f "$output_dir/issues.txt" ]; then
        local issue_count=$(wc -l < "$output_dir/issues.txt")
        log_info "Infer found $issue_count potential issues"
        log_info "Details: $output_dir/issues.txt"
    fi
}

# Generate summary report
generate_summary() {
    local summary_file="$OUTPUT_DIR/analysis_summary_$(date +%Y%m%d_%H%M%S).md"

    cat > "$summary_file" << EOF
# Static Analysis Summary

**Date:** $(date +"%Y-%m-%d %H:%M:%S")
**Repository:** $REPO_ROOT

## Tools Run

### Clang Thread Safety Analysis
- Status: Completed
- Warnings: $1

### Facebook Infer
- Status: $(command -v infer &> /dev/null && echo "Completed" || echo "Not installed")

## Next Steps

1. Review warnings in the output files
2. Add TSA annotations to critical code paths
3. Fix any real concurrency bugs found
4. Re-run analysis to verify fixes

## Files Analyzed

$(find "$MPS_SRC" -name "*.h" -o -name "*.hpp" 2>/dev/null | wc -l) header files in:
- $MPS_SRC

EOF

    log_info "Summary written to: $summary_file"
}

# Main
main() {
    local mode="${1:---all}"

    log_info "Static Analysis for MPS Parallel Inference"
    log_info "=========================================="

    check_mps_source

    local clang_warnings=0

    case "$mode" in
        --clang-only)
            clang_warnings=$(run_clang_tsa)
            ;;
        --infer-only)
            run_infer
            ;;
        --all|*)
            clang_warnings=$(run_clang_tsa)
            run_infer
            ;;
    esac

    generate_summary "$clang_warnings"

    log_info "Analysis complete!"
}

main "$@"
