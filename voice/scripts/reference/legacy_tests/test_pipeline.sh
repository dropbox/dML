#!/bin/bash
# Test the full translation + TTS pipeline

set -e

echo "=== Testing Full Pipeline ==="
echo ""

# Activate virtualenv
source venv/bin/activate

# Test input
TEST_TEXT="Hello, how are you today?"

echo "Input: $TEST_TEXT"
echo ""

# Step 1: Translation
echo "[1/3] Translation..."
TRANSLATED=$(echo "$TEST_TEXT" | python3 stream-tts-rust/python/translation_worker.py 2>/dev/null)
echo "  Translated: $TRANSLATED"
echo ""

# Step 2: TTS (using background process to keep files alive)
echo "[2/3] Text-to-Speech..."
# Start TTS worker in background
python3 stream-tts-rust/python/tts_worker.py 2>/dev/null &
TTS_PID=$!

# Send text and read audio file path
AUDIO_FILE=$(echo "$TRANSLATED" | python3 stream-tts-rust/python/tts_worker.py 2>&1 | grep -v '^\[TTS\]' | head -1)
echo "  Audio file: $AUDIO_FILE"
echo ""

# Step 3: Play audio (do it quickly before cleanup)
echo "[3/3] Playing audio..."
if [ -f "$AUDIO_FILE" ]; then
    afplay "$AUDIO_FILE" &
    PLAY_PID=$!
    echo "  ✓ Playing..."
    wait $PLAY_PID
    echo "  ✓ Playback complete"
else
    echo "  ✗ Audio file not found"
    exit 1
fi

echo ""
echo "=== Pipeline Test Complete ==="
