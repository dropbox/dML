#!/bin/bash
# Quick test of the prototype pipeline
# This version keeps audio files for playback

set -e

echo "=== Testing Prototype Pipeline ==="
echo ""

source venv/bin/activate

# Test texts
TEST_TEXTS=(
    "Hello, how are you today?"
    "I am working on a voice project."
    "This system translates English to Japanese."
)

# Create output directory
OUT_DIR="test_output"
mkdir -p "$OUT_DIR"

echo "Output directory: $OUT_DIR"
echo ""

for i in "${!TEST_TEXTS[@]}"; do
    TEXT="${TEST_TEXTS[$i]}"
    echo "[$((i+1))/${#TEST_TEXTS[@]}] Processing: $TEXT"

    # Translate
    TRANSLATED=$(echo "$TEXT" | python3 stream-tts-rust/python/translation_worker.py 2>&1 | grep -v '^\[Translation\]' | tail -1)
    echo "  → $TRANSLATED"

    # Generate audio (save to output dir)
    AUDIO_FILE="$OUT_DIR/audio_$i.aiff"
    say -v Kyoko -o "$AUDIO_FILE" "$TRANSLATED"
    echo "  → $AUDIO_FILE"

    # Play it
    afplay "$AUDIO_FILE"

    echo ""
done

echo "=== Test Complete ==="
echo "Audio files saved in: $OUT_DIR/"
