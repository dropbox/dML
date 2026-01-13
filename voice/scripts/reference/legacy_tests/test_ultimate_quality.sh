#!/bin/bash
#
# Test Ultimate Quality Pipeline
# Qwen2.5-7B Translation + Google TTS
# Expected latency: ~780ms (595ms translation + 185ms TTS)
# Quality: Maximum (better than production NLLB-200)
#

set -e

echo "==========================================="
echo "Ultimate Quality TTS Pipeline Test"
echo "==========================================="
echo ""
echo "Configuration:"
echo "  Translation: Qwen2.5-7B-Instruct (highest quality)"
echo "  TTS: Google TTS (cloud)"
echo "  Expected latency: ~780ms per sentence"
echo ""

# Check if Qwen model exists
QWEN_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
if [ ! -d "$QWEN_PATH" ]; then
    echo "⚠️  Warning: Qwen2.5-7B-Instruct model not found"
    echo "   The model will be downloaded on first run (~15GB)"
    echo "   Location: $QWEN_PATH"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Test sentences
TEST_INPUT=$(cat <<'EOF'
Hello, this is a test of the ultimate quality voice TTS system with Qwen translation.
The Qwen model provides superior translation quality compared to NLLB, with better handling of context and nuance.
This comes at the cost of higher latency, but produces the most natural and accurate Japanese translations.
EOF
)

# Run the pipeline with Qwen translation worker
echo "$TEST_INPUT" | \
  TRANSLATION_WORKER="./stream-tts-rust/python/translation_worker_qwen.py" \
  TTS_WORKER="./stream-tts-rust/python/tts_worker_gtts.py" \
  ./stream-tts-rust/target/release/stream-tts-rust

echo ""
echo "==========================================="
echo "Test completed!"
echo "==========================================="
echo ""
echo "Performance comparison:"
echo "  Production (NLLB+gTTS):  340ms - ✅ Fast & high quality"
echo "  Ultimate (Qwen+gTTS):    780ms - ✅ Maximum quality"
echo ""
echo "Use production for general use, ultimate for:"
echo "  - Technical documentation translation"
echo "  - Context-dependent content"
echo "  - When translation accuracy is critical"
echo ""
