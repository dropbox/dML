#!/bin/bash

# Test multiple sentences with the TTS system

echo "==========================================="
echo "Multi-Sentence TTS Test"
echo "==========================================="
echo ""

cat << 'TESTEOF' | ./stream-tts-rust/target/release/stream-tts-rust 2>&1 | grep -E "(Processing|Translated|Complete)"
{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant"}}
{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The Voice TTS system is now working perfectly."}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" It translates English to Japanese in real-time."}}
{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" The performance is excellent with only 340 milliseconds of latency."}}
{"type":"content_block_stop","index":0}
{"type":"message_delta","delta":{"stop_reason":"end_turn"}}
{"type":"message_stop"}
TESTEOF

echo ""
echo "==========================================="
echo "Test completed! Audio should have played."
echo "==========================================="
