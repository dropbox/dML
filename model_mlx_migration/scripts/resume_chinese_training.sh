#!/bin/bash
# Script to resume Chinese CTC training with expanded dataset
# Waits for downloads, extracts, and resumes from existing checkpoint

set -e
cd /Users/ayates/model_mlx_migration

CHINESE_DIR="data/multilingual/chinese"
AISHELL_TGZ="$CHINESE_DIR/data_aishell.tgz"
FREEST_TGZ="$CHINESE_DIR/ST-CMDS-20170001_1-OS.tar.gz"
CHECKPOINT="checkpoints/ctc_chinese/step_9000.npz"

echo "=== Chinese CTC Training Resume Script ==="
echo "Will resume from: $CHECKPOINT"
echo ""

# Wait for AISHELL download
echo "Waiting for AISHELL-1 download (14.5GB)..."
while true; do
    if [ -f "$AISHELL_TGZ" ]; then
        size=$(stat -f%z "$AISHELL_TGZ" 2>/dev/null || stat -c%s "$AISHELL_TGZ" 2>/dev/null)
        # AISHELL is ~14.5GB = 15569256448 bytes, check if > 14GB
        if [ "$size" -gt 15000000000 ]; then
            echo "AISHELL-1 download complete: $(du -h $AISHELL_TGZ | cut -f1)"
            break
        fi
        echo "  AISHELL-1: $(du -h $AISHELL_TGZ | cut -f1) / 14.5GB"
    fi
    sleep 60
done

# Wait for Free ST download
echo "Waiting for Free ST Chinese download (8GB)..."
while true; do
    if [ -f "$FREEST_TGZ" ]; then
        size=$(stat -f%z "$FREEST_TGZ" 2>/dev/null || stat -c%s "$FREEST_TGZ" 2>/dev/null)
        # Free ST is ~8GB = 8589934592 bytes, check if > 7.5GB
        if [ "$size" -gt 8000000000 ]; then
            echo "Free ST Chinese download complete: $(du -h $FREEST_TGZ | cut -f1)"
            break
        fi
        echo "  Free ST: $(du -h $FREEST_TGZ | cut -f1) / 8GB"
    fi
    sleep 60
done

echo ""
echo "=== Extracting datasets ==="

# Extract AISHELL
if [ ! -d "$CHINESE_DIR/data_aishell" ]; then
    echo "Extracting AISHELL-1..."
    cd "$CHINESE_DIR"
    tar xzf data_aishell.tgz
    cd /Users/ayates/model_mlx_migration
    echo "AISHELL-1 extracted"
else
    echo "AISHELL-1 already extracted"
fi

# Extract Free ST
if [ ! -d "$CHINESE_DIR/ST-CMDS-20170001_1-OS" ]; then
    echo "Extracting Free ST Chinese..."
    cd "$CHINESE_DIR"
    tar xzf ST-CMDS-20170001_1-OS.tar.gz
    cd /Users/ayates/model_mlx_migration
    echo "Free ST Chinese extracted"
else
    echo "Free ST Chinese already extracted"
fi

echo ""
echo "=== Starting Combined Chinese CTC Training ==="
echo "Resuming from: $CHECKPOINT"
echo "Datasets: THCHS-30 + AISHELL-1 + Free ST Chinese"
echo "Expected samples: ~230,000"
echo ""

# Start training with combined dataset, resuming from checkpoint
nohup python3 -m tools.whisper_mlx.train_ctc \
    --combined-chinese \
    --chinese-data-dirs \
        data/multilingual/chinese/data_thchs30 \
        data/multilingual/chinese/data_aishell \
        data/multilingual/chinese/ST-CMDS-20170001_1-OS \
    --chinese-data-types thchs30 aishell freest \
    --output-dir checkpoints/ctc_chinese_combined \
    --epochs 3 --batch-size 4 --spec-augment --label-smoothing 0.1 \
    --resume "$CHECKPOINT" \
    > checkpoints/ctc_chinese_combined_training.log 2>&1 &

echo "Training started with PID: $!"
echo "Log: checkpoints/ctc_chinese_combined_training.log"
echo ""
echo "Monitor with: tail -f checkpoints/ctc_chinese_combined/training.log"
