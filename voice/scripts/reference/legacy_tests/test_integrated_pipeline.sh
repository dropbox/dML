#!/bin/bash
# Test the integrated Rust + Python pipeline with sample Claude JSON

set -e

echo "=== Testing Integrated Pipeline ==="
echo ""
echo "This test will:"
echo "1. Send sample Claude JSON to the Rust coordinator"
echo "2. Rust will extract text and spawn Python workers"
echo "3. Python will translate English → Japanese"
echo "4. Python will synthesize speech"
echo "5. Audio will play automatically"
echo ""

# Sample Claude JSON messages
cat << 'INNEREOF' | /Users/ayates/voice/stream-tts-rust/target/release/stream-tts-rust
{"type":"message_start","message":{"role":"assistant","content":[]}}
{"type":"content_block_start","content_block":{"type":"text","text":""}}
{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello, I will help you with your request."}}
{"type":"content_block_delta","delta":{"type":"text_delta","text":" Let me start by analyzing the problem."}}
{"type":"content_block_delta","delta":{"type":"text_delta","text":" This is a test of the streaming translation system."}}
{"type":"content_block_stop"}
{"type":"message_stop"}
INNEREOF

echo ""
echo "=== Pipeline Test Complete ==="
