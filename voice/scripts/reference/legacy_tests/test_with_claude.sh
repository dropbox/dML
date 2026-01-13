#!/bin/bash
# Test the integrated pipeline with real Claude output

cd /Users/ayates/voice

echo "=== Testing Integrated Pipeline with Claude ==="
echo ""
echo "Asking Claude a simple question..."
echo ""

# Use Claude CLI to ask a simple question and pipe through our TTS pipeline
claude --dangerously-skip-permissions \
  -p "Say hello in 3 different friendly ways. Keep each greeting short and simple." \
  --permission-mode acceptEdits \
  --output-format stream-json \
  --verbose \
  2>&1 | /Users/ayates/voice/stream-tts-rust/target/release/stream-tts-rust

echo ""
echo "=== Test complete ==="
