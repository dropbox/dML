#!/bin/bash
# Metal API Call Auditor for MPS Thread Safety
#
# This script finds all Metal API calls that could potentially race
# and need MPSEncodingLock protection.
#
# Usage: ./tools/audit_metal_calls.sh
# Output: tools/metal_api_audit.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
FORK_DIR="${REPO_ROOT}/pytorch-mps-fork"
OUTPUT_FILE="${SCRIPT_DIR}/metal_api_audit.txt"

if [ ! -d "$FORK_DIR" ]; then
    echo "ERROR: pytorch-mps-fork not found at $FORK_DIR"
    exit 1
fi

cd "$FORK_DIR"

echo "=== Metal API Call Audit ===" > "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Category 1: Command dispatch calls (most critical - these are where crashes occur)
echo "=== DISPATCH CALLS (P0 - crash site) ===" >> "$OUTPUT_FILE"
grep -rn "dispatchThread" aten/src/ATen/native/mps/ aten/src/ATen/mps/ 2>/dev/null >> "$OUTPUT_FILE" || true
echo "" >> "$OUTPUT_FILE"

# Category 2: Command encoder creation
echo "=== ENCODER CREATION ===" >> "$OUTPUT_FILE"
grep -rn "computeCommandEncoder\|blitCommandEncoder" aten/src/ATen/native/mps/ aten/src/ATen/mps/ 2>/dev/null >> "$OUTPUT_FILE" || true
echo "" >> "$OUTPUT_FILE"

# Category 3: Library/shader compilation
echo "=== LIBRARY/SHADER COMPILATION ===" >> "$OUTPUT_FILE"
grep -rn "newLibraryWith\|newFunctionWithName\|newComputePipelineState" aten/src/ATen/native/mps/ aten/src/ATen/mps/ 2>/dev/null >> "$OUTPUT_FILE" || true
echo "" >> "$OUTPUT_FILE"

# Category 4: Graph encoding
echo "=== GRAPH ENCODING ===" >> "$OUTPUT_FILE"
grep -rn "encodeToCommandBuffer" aten/src/ATen/native/mps/ aten/src/ATen/mps/ 2>/dev/null >> "$OUTPUT_FILE" || true
echo "" >> "$OUTPUT_FILE"

# Category 5: Event encoding
echo "=== EVENT ENCODING ===" >> "$OUTPUT_FILE"
grep -rn "encodeSignalEvent\|encodeWaitForEvent" aten/src/ATen/native/mps/ aten/src/ATen/mps/ 2>/dev/null >> "$OUTPUT_FILE" || true
echo "" >> "$OUTPUT_FILE"

# Count totals
DISPATCH_COUNT=$(grep -c "dispatchThread" "$OUTPUT_FILE" 2>/dev/null || echo 0)
ENCODER_COUNT=$(grep -c "CommandEncoder" "$OUTPUT_FILE" 2>/dev/null || echo 0)
LIBRARY_COUNT=$(grep -c "newLibrary\|newFunction\|newComputePipeline" "$OUTPUT_FILE" 2>/dev/null || echo 0)

echo "=== SUMMARY ===" >> "$OUTPUT_FILE"
echo "Dispatch calls: $DISPATCH_COUNT" >> "$OUTPUT_FILE"
echo "Encoder creations: $ENCODER_COUNT" >> "$OUTPUT_FILE"
echo "Library/shader compilations: $LIBRARY_COUNT" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "NEXT STEP: For each call above, verify it is within:" >> "$OUTPUT_FILE"
echo "  1. dispatch_sync_with_rethrow block, OR" >> "$OUTPUT_FILE"
echo "  2. Direct MPSEncodingLock scope, OR" >> "$OUTPUT_FILE"
echo "  3. std::call_once with lock inside" >> "$OUTPUT_FILE"

echo ""
echo "Audit complete. Results in: $OUTPUT_FILE"
echo ""
echo "Summary:"
echo "  Dispatch calls: $DISPATCH_COUNT"
echo "  Encoder creations: $ENCODER_COUNT"
echo "  Library/shader compilations: $LIBRARY_COUNT"
echo ""
echo "Review each call and verify MPSEncodingLock protection."
