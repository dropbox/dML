#!/bin/bash
# Copyright 2024-2025 Andrew Yates
# Average checkpoints for all CTC language models
#
# Run this after training completes for each language to create
# averaged checkpoints with better generalization.

set -e

CHECKPOINT_DIR="checkpoints"
PYTHON="python"

echo "========================================"
echo "CTC Checkpoint Averaging - All Languages"
echo "========================================"
echo ""

# Function to average last N checkpoints for a language
average_language() {
    local lang=$1
    local dir=$2
    local n_checkpoints=${3:-5}

    if [ ! -d "$dir" ]; then
        echo "[$lang] SKIP - directory not found: $dir"
        return
    fi

    # Find latest checkpoints
    local checkpoints=$(ls -t "$dir"/step_*.npz 2>/dev/null | head -$n_checkpoints)
    local count=$(echo "$checkpoints" | wc -l | tr -d ' ')

    if [ "$count" -lt 2 ]; then
        echo "[$lang] SKIP - need at least 2 checkpoints, found $count"
        return
    fi

    # Get the latest step number for naming
    local latest=$(ls -t "$dir"/step_*.npz | head -1 | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+')
    local output="$dir/averaged_${latest}_${count}pt.npz"

    if [ -f "$output" ]; then
        echo "[$lang] SKIP - already exists: $output"
        return
    fi

    echo "[$lang] Averaging $count checkpoints..."
    echo "  Latest step: $latest"
    echo "  Output: $output"

    # Build checkpoint args
    local ckpt_args=""
    for ckpt in $checkpoints; do
        ckpt_args="$ckpt_args $ckpt"
    done

    $PYTHON scripts/average_ctc_checkpoints.py \
        --checkpoints $ckpt_args \
        --output "$output"

    echo "[$lang] Done!"
    echo ""
}

# Process all CTC languages
echo "Processing CTC languages..."
echo ""

# English (full training)
average_language "English" "$CHECKPOINT_DIR/ctc_english_full" 5

# v3 models (latest versions)
average_language "Chinese" "$CHECKPOINT_DIR/ctc_chinese_v3" 5
average_language "French" "$CHECKPOINT_DIR/ctc_french_v3" 5
average_language "German" "$CHECKPOINT_DIR/ctc_german_v3" 5
average_language "Hindi" "$CHECKPOINT_DIR/ctc_hindi_v3" 5
average_language "Japanese" "$CHECKPOINT_DIR/ctc_japanese_v3" 5
average_language "Korean" "$CHECKPOINT_DIR/ctc_korean_v3" 5
average_language "Spanish" "$CHECKPOINT_DIR/ctc_spanish_v3" 5

# Older versions (if no v3 exists)
average_language "Chinese-legacy" "$CHECKPOINT_DIR/ctc_chinese" 5
average_language "French-legacy" "$CHECKPOINT_DIR/ctc_french" 5
average_language "German-legacy" "$CHECKPOINT_DIR/ctc_german" 5
average_language "Hindi-legacy" "$CHECKPOINT_DIR/ctc_hindi" 5
average_language "Japanese-legacy" "$CHECKPOINT_DIR/ctc_japanese" 5
average_language "Korean-legacy" "$CHECKPOINT_DIR/ctc_korean" 5
average_language "Spanish-legacy" "$CHECKPOINT_DIR/ctc_spanish" 5

# Low-resource languages
average_language "Kashmiri" "$CHECKPOINT_DIR/ctc_kashmiri" 3

echo "========================================"
echo "Checkpoint Averaging Complete"
echo "========================================"
echo ""
echo "To evaluate averaged checkpoints, run:"
echo "  PYTHONPATH=. python scripts/test_ctc_streaming_eval.py --checkpoint <path>"
echo ""
