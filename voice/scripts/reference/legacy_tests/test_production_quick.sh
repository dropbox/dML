#!/bin/bash

# Production TTS Quick Test

echo "==========================================="
echo "Production TTS Quick Test"
echo "==========================================="
echo ""
echo "Configuration:"
echo "  Translation: NLLB-200-600M (Metal GPU, bfloat16)"
echo "  TTS: Google TTS (cloud)"
echo ""

# Create test JSON in Claude stream-json format
cat << 'TESTEOF' | ./stream-tts-rust/target/release/stream-tts-rust 2>&1 | grep -E "(Processing|Translated|Audio|Played|Complete)"
{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant"}}
{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, this is a test of the voice TTS system."}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" It should translate this English text to Japanese and speak it."}}
{"type":"content_block_stop","index":0}
{"type":"message_delta","delta":{"stop_reason":"end_turn"}}
{"type":"message_stop"}
TESTEOF

echo ""
echo "==========================================="
echo "Test completed!"
echo "==========================================="
echo ""
echo "Expected performance:"
echo "  Translation: ~154ms per sentence"
echo "  TTS: ~185ms per sentence"
echo "  Total: ~340ms per sentence"
