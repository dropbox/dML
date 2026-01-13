#!/bin/bash
# Extract audio from MELD mp4 files to 16kHz mono WAV for Whisper

MELD_DIR="/Users/ayates/model_mlx_migration/data/emotion_punctuation/MELD.Raw"

extract_split() {
    local input_dir=$1
    local output_dir=$2
    local split_name=$3

    echo "[$split_name] Starting extraction from $input_dir to $output_dir"
    count=0
    total=$(ls "$input_dir"/*.mp4 2>/dev/null | wc -l)

    for mp4 in "$input_dir"/*.mp4; do
        if [ -f "$mp4" ]; then
            basename=$(basename "$mp4" .mp4)
            wav="$output_dir/${basename}.wav"
            if [ ! -f "$wav" ]; then
                ffmpeg -i "$mp4" -ac 1 -ar 16000 -vn "$wav" -y -loglevel error
            fi
            count=$((count + 1))
            if [ $((count % 500)) -eq 0 ]; then
                echo "[$split_name] Processed $count / $total"
            fi
        fi
    done
    echo "[$split_name] Done: $count files extracted"
}

# Extract all splits
extract_split "$MELD_DIR/train_splits" "$MELD_DIR/audio_train" "train"
extract_split "$MELD_DIR/dev_splits_complete" "$MELD_DIR/audio_dev" "dev"
extract_split "$MELD_DIR/output_repeated_splits_test" "$MELD_DIR/audio_test" "test"

echo "All MELD audio extraction complete!"
