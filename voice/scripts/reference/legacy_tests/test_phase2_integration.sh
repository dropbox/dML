#!/bin/bash
# Test Phase 2 Integration
# Tests the full pipeline with Phase 2 optimized workers

set -e

echo "==================================="
echo "Phase 2 Integration Test"
echo "==================================="
echo

# Test with a simple Claude-format message
echo "Testing with Claude stream-json format..."
echo

# Create a test message simulating Claude output
cat << 'EOF' | stream-tts-rust/target/release/stream-tts-rust
{"type":"message_start","message":{"role":"assistant","content":[]},"stats":{"cache_hit":false}}
{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello! "}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"This is a Phase 2 test. "}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I am testing the optimized translation and TTS workers."}}
{"type":"content_block_stop","index":0}
{"type":"message_stop","stats":{"input_tokens":100,"output_tokens":50}}
EOF

echo
echo "==================================="
echo "Test complete!"
echo "==================================="
