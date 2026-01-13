#!/bin/bash
# Quick test of C++ TTS system

echo "Testing C++ TTS System"
echo "======================"
echo ""

echo '{"content":[{"type":"text","text":"Hello from the new C++ system"}]}' | \
  ./stream-tts-cpp/build/stream-tts

echo ""
echo "Test complete!"
