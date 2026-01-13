#!/bin/bash
# GPU Utilization Measurement Script for M4 Max (MPS)
# Worker #240 - 2025-12-06
#
# Usage:
#   sudo ./scripts/measure_gpu.sh [duration_seconds]
#
# This script uses macOS powermetrics to capture GPU power and utilization.
# Must be run with sudo for powermetrics access.

set -e

DURATION=${1:-10}
INTERVAL_MS=500
SAMPLES=$((DURATION * 1000 / INTERVAL_MS))
OUTPUT_DIR="/tmp/gpu_metrics"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/gpu_metrics_${TIMESTAMP}.txt"
PLIST_FILE="${OUTPUT_DIR}/gpu_metrics_${TIMESTAMP}.plist"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "GPU Utilization Measurement"
echo "==========================="
echo "Duration: ${DURATION}s"
echo "Interval: ${INTERVAL_MS}ms"
echo "Samples: ${SAMPLES}"
echo "Output: ${OUTPUT_FILE}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script requires sudo to run powermetrics"
    echo "Usage: sudo $0 [duration_seconds]"
    exit 1
fi

# Run powermetrics with GPU sampler
echo "Starting GPU measurement at $(date)..."
echo ""

# Capture both text and plist output
powermetrics \
    --samplers gpu_power,cpu_power \
    -i "${INTERVAL_MS}" \
    -n "${SAMPLES}" \
    --format text \
    2>&1 | tee "$OUTPUT_FILE"

echo ""
echo "Measurement complete."
echo "Raw output saved to: ${OUTPUT_FILE}"
echo ""

# Parse and summarize GPU metrics
echo "=== GPU Summary ==="
if grep -q "GPU" "$OUTPUT_FILE"; then
    echo "GPU metrics found in output"
    grep -E "(GPU|gpu)" "$OUTPUT_FILE" | head -20
else
    echo "No GPU-specific metrics found."
    echo "This may indicate the GPU sampler is not supported on this hardware."
fi

# Also try plist format for machine-readable data
PLIST_SAMPLES=$((SAMPLES / 2))  # Fewer samples for plist
if [ "$PLIST_SAMPLES" -lt 1 ]; then
    PLIST_SAMPLES=1
fi

echo ""
echo "Capturing plist format for machine parsing..."
powermetrics \
    --samplers gpu_power \
    -i "${INTERVAL_MS}" \
    -n "${PLIST_SAMPLES}" \
    --format plist \
    > "$PLIST_FILE" 2>&1

echo "Plist output saved to: ${PLIST_FILE}"

# Try to extract key metrics from plist
if command -v plutil &> /dev/null; then
    echo ""
    echo "=== Plist Contents (first 50 lines) ==="
    plutil -p "$PLIST_FILE" 2>/dev/null | head -50 || cat "$PLIST_FILE" | head -50
fi

echo ""
echo "=== Files Created ==="
ls -la "${OUTPUT_DIR}/gpu_metrics_${TIMESTAMP}"*
