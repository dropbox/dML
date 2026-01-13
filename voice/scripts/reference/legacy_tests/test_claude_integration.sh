#!/bin/bash
# Test with simulated Claude output
echo "Testing with simulated Claude Code stream-json output..."

cat << 'JSON' | ./prototype_pipeline.py
{"type":"message_start","message":{"id":"test","type":"message","role":"assistant","content":[]}}
{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello! I can help you with your code."}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Let me analyze the files."}}
{"type":"content_block_stop","index":0}
{"type":"message_delta","delta":{"stop_reason":"end_turn"}}
{"type":"message_stop"}
JSON

echo "Integration test complete!"
