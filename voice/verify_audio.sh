#!/bin/bash
# AUDIO VERIFICATION SCRIPT
# Human must run this and confirm they HEAR audio
#
# Usage: ./verify_audio.sh
#
# This script tests:
# 1. macOS system audio (say command)
# 2. WAV playback (afplay)
# 3. miniaudio direct playback
# 4. TTS pipeline playback
#
# All tests should produce audible sound. If you hear nothing:
# - Check system volume (not muted)
# - Check output device (System Preferences > Sound)
# - Check for headphones/external devices

set -e

echo "======================================"
echo "  AUDIO VERIFICATION SCRIPT"
echo "======================================"
echo ""

# Check we're in the right place
if [ ! -d "stream-tts-cpp" ]; then
    echo "ERROR: Run this from the voice/ directory"
    exit 1
fi

cd stream-tts-cpp

# Check binary exists
if [ ! -f "build/stream-tts-cpp" ]; then
    echo "ERROR: build/stream-tts-cpp not found"
    echo "Run: cd stream-tts-cpp && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

echo "System volume:"
osascript -e "get volume settings"
echo ""

echo "======================================"
echo "  TEST 1: macOS 'say' command"
echo "======================================"
echo "Playing 'Audio test one' via macOS TTS..."
say "Audio test one"
echo "Did you hear 'Audio test one'? [y/n]"
read -r T1
[ "$T1" = "y" ] && R1="PASS" || R1="FAIL"

echo ""
echo "======================================"
echo "  TEST 2: WAV file via afplay"
echo "======================================"
echo "Generating TTS audio and saving to WAV..."
echo '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Audio test two"}}' | \
    ./build/stream-tts-cpp config/default.yaml --save-audio /tmp/verify_audio.wav 2>/dev/null
echo "Playing WAV via afplay..."
afplay /tmp/verify_audio.wav
echo "Did you hear Japanese speech? [y/n]"
read -r T2
[ "$T2" = "y" ] && R2="PASS" || R2="FAIL"

echo ""
echo "======================================"
echo "  TEST 3: miniaudio direct playback"
echo "======================================"
if [ -f "build/test_miniaudio_diagnostic" ]; then
    echo "Playing 440Hz tone for 3 seconds via miniaudio..."
    ./build/test_miniaudio_diagnostic 2>&1 | grep -E "(Name:|Sample Rate:|PLAYING|COMPLETE)" || true
    echo ""
    echo "Did you hear a 440Hz tone? [y/n]"
    read -r T3
    [ "$T3" = "y" ] && R3="PASS" || R3="FAIL"
else
    echo "test_miniaudio_diagnostic not built, skipping"
    R3="SKIP"
fi

echo ""
echo "======================================"
echo "  TEST 4: TTS pipeline playback"
echo "======================================"
echo "Running TTS with miniaudio playback..."
echo '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Test four"}}' | \
    ./build/stream-tts-cpp config/default.yaml 2>&1 | grep -E "(Audio playback)" || true
echo ""
echo "Did you hear Japanese speech directly (not from WAV)? [y/n]"
read -r T4
[ "$T4" = "y" ] && R4="PASS" || R4="FAIL"

echo ""
echo "======================================"
echo "  RESULTS"
echo "======================================"
echo "Test 1 (macOS say):    $R1"
echo "Test 2 (WAV afplay):   $R2"
echo "Test 3 (miniaudio):    $R3"
echo "Test 4 (TTS pipeline): $R4"
echo ""

if [ "$R1" = "PASS" ] && [ "$R2" = "PASS" ] && [ "$R3" = "PASS" ] && [ "$R4" = "PASS" ]; then
    echo "SUCCESS: All audio tests passed!"
    echo "The TTS system is WORKING."
    exit 0
elif [ "$R1" = "PASS" ] && [ "$R2" = "PASS" ] && [ "$R3" = "FAIL" ]; then
    echo "PARTIAL: System audio works, but miniaudio has issues."
    echo "Workaround: Use --save-audio flag and play with afplay."
    exit 1
elif [ "$R1" = "FAIL" ]; then
    echo "FAIL: System audio is broken."
    echo "Check: System volume, mute state, output device."
    exit 2
else
    echo "PARTIAL: Some tests failed."
    echo "Review individual results above."
    exit 1
fi
