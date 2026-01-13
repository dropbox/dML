#!/bin/bash
# Test the stream-tts-rust pipeline with real Claude Code output

set -e

echo "=== Testing Stream TTS Rust Pipeline ==="
echo ""
echo "This will run Claude Code with TTS output."
echo ""

# Run Claude Code with stream-json output and pipe to TTS
claude code \
  --dangerously-skip-permissions \
  --permission-mode acceptEdits \
  -p "Say hello in three different ways" \
  --output-format stream-json \
  2>&1 | \
  ./stream-tts-rust/target/release/stream-tts-rust

echo ""
echo "=== Test Complete ==="
