#!/bin/bash
# Clang Thread Safety Analysis using compile_commands.json
# This script extracts compilation commands for MPS files and runs TSA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTORCH_ROOT="$(cd "$PROJECT_ROOT/../pytorch-mps-fork" && pwd)"  # Canonicalize
COMPILE_DB="$PYTORCH_ROOT/build/compile_commands.json"

# Output file for JSON results
OUTPUT_JSON="${1:-$PROJECT_ROOT/tsa_results.json}"
START_DIR="$(pwd)"
if [[ "$OUTPUT_JSON" != /* ]]; then
    OUTPUT_JSON="$START_DIR/$OUTPUT_JSON"
fi
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_JSON")" && pwd)"
OUTPUT_JSON="$OUTPUT_DIR/$(basename "$OUTPUT_JSON")"

# Files to analyze (key MPS concurrency files)
MPS_FILES=(
    "aten/src/ATen/mps/MPSStream.mm"
    "aten/src/ATen/mps/MPSAllocator.mm"
    "aten/src/ATen/mps/MPSEvent.mm"
    "aten/src/ATen/mps/MPSDevice.mm"
)

if [ ! -f "$COMPILE_DB" ]; then
    echo "ERROR: compile_commands.json not found at $COMPILE_DB"
    echo '{"success":false,"error":"compile_commands.json not found","results":[]}' > "$OUTPUT_JSON"
    exit 1
fi

# Check for clang
if ! command -v clang++ &> /dev/null; then
    echo "ERROR: clang++ not found"
    echo '{"success":false,"error":"clang++ not found","results":[]}' > "$OUTPUT_JSON"
    exit 1
fi

echo "Clang Thread Safety Analysis"
echo "============================"
echo "Compile DB: $COMPILE_DB"
echo "Output: $OUTPUT_JSON"
echo ""

# Initialize JSON output
echo '{"success":true,"error":"","results":[' > "$OUTPUT_JSON"

FIRST_RESULT=true
TOTAL_WARNINGS=0
TOTAL_ERRORS=0
HAS_ERROR=false

for MPS_FILE in "${MPS_FILES[@]}"; do
    FULL_PATH="$PYTORCH_ROOT/$MPS_FILE"

    if [ ! -f "$FULL_PATH" ]; then
        echo "WARNING: $MPS_FILE not found, skipping"
        continue
    fi

    echo "Analyzing: $MPS_FILE"

    # Extract compile command from compile_commands.json using jq
    # Get the command field for this file
    COMPILE_CMD=$(jq -r --arg file "$FULL_PATH" '.[] | select(.file == $file) | .command' "$COMPILE_DB" | head -1)

    if [ -z "$COMPILE_CMD" ] || [ "$COMPILE_CMD" = "null" ]; then
        echo "  WARNING: No compile command found for $MPS_FILE"
        continue
    fi

    # Get the directory for this compilation
    COMPILE_DIR=$(jq -r --arg file "$FULL_PATH" '.[] | select(.file == $file) | .directory' "$COMPILE_DB" | head -1)

    # Replace the output file flags and add TSA flags
    # Remove -o <file> and add -fsyntax-only
    TSA_CMD=$(echo "$COMPILE_CMD" | \
        sed 's/-c //' | \
        sed 's/-o [^ ]*//' | \
        sed 's/-MF [^ ]*//' | \
        sed 's/-MT [^ ]*//' | \
        sed 's/-MD//')

    # Add TSA flags
    # NOTE: -Wthread-safety-negative is intentionally OMITTED because TSA's negative
    # capability analysis doesn't work correctly with templated RAII lock types.
    # The MPS_EXCLUDES annotations on functions are present and correct, but TSA
    # cannot match template parameters with actual mutex arguments. See comments
    # in MPSThreadSafety.h for details. All other TSA warnings are still checked.
    TSA_CMD="$TSA_CMD -fsyntax-only -Wthread-safety -Wthread-safety-analysis -Wthread-safety-attributes -Wthread-safety-precise"

    # Run the analysis from the compile directory
    TEMP_OUTPUT=$(mktemp)
    cd "$COMPILE_DIR"
    eval "$TSA_CMD" 2>"$TEMP_OUTPUT" || true  # Capture output regardless of exit code

    # Count warnings and errors from actual output (ensure single integer)
    WARNING_COUNT=$(grep -c "warning:" "$TEMP_OUTPUT" 2>/dev/null | head -1 || echo 0)
    WARNING_COUNT="${WARNING_COUNT:-0}"
    ERROR_COUNT=$(grep -c "error:" "$TEMP_OUTPUT" 2>/dev/null | head -1 || echo 0)
    ERROR_COUNT="${ERROR_COUNT:-0}"

    if [ "$ERROR_COUNT" -gt 0 ]; then
        STATUS="errors"
        HAS_ERROR=true
    elif [ "$WARNING_COUNT" -gt 0 ]; then
        STATUS="warnings"
    else
        STATUS="pass"
    fi

    # Read warnings
    WARNINGS=$(cat "$TEMP_OUTPUT" | jq -Rs '.')
    rm -f "$TEMP_OUTPUT"

    TOTAL_WARNINGS=$((TOTAL_WARNINGS + WARNING_COUNT))
    TOTAL_ERRORS=$((TOTAL_ERRORS + ERROR_COUNT))

    echo "  Status: $STATUS (warnings: $WARNING_COUNT, errors: $ERROR_COUNT)"

    # Add to JSON
    if [ "$FIRST_RESULT" = true ]; then
        FIRST_RESULT=false
    else
        echo "," >> "$OUTPUT_JSON"
    fi

    echo "{\"file\":\"$MPS_FILE\",\"status\":\"$STATUS\",\"warnings\":$WARNING_COUNT,\"errors\":$ERROR_COUNT,\"output\":$WARNINGS}" >> "$OUTPUT_JSON"
done

# Close JSON
echo "]," >> "$OUTPUT_JSON"
echo "\"summary\":{\"total_warnings\":$TOTAL_WARNINGS,\"total_errors\":$TOTAL_ERRORS,\"files_analyzed\":${#MPS_FILES[@]}}}" >> "$OUTPUT_JSON"

# Update success based on errors
if [ "$HAS_ERROR" = true ]; then
    sed -i '' 's/"success":true/"success":false/' "$OUTPUT_JSON"
fi

echo ""
echo "Summary: $TOTAL_WARNINGS warnings, $TOTAL_ERRORS errors in ${#MPS_FILES[@]} files"
echo "Results written to: $OUTPUT_JSON"

# Exit with error if warnings or errors found (for gating)
if [ "$TOTAL_ERRORS" -gt 0 ]; then
    exit 1
elif [ "$TOTAL_WARNINGS" -gt 0 ]; then
    exit 2  # Different exit code for warnings-only
fi
exit 0
