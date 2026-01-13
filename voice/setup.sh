#!/bin/bash
# setup.sh - Set up the TTS and translation environment
# Copyright 2025 Andrew Yates. All rights reserved.

echo "🔧 Setting up Voice Processing Environment"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install Python pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install development/testing Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
REQ_FILE="requirements-dev.txt"
if [ ! -f "$REQ_FILE" ]; then
    echo "⚠️  $REQ_FILE not found, falling back to requirements.txt (runtime is C++-only)"
    REQ_FILE="requirements.txt"
fi
pip install -r "$REQ_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    deactivate
    exit 1
fi

deactivate

# Make scripts executable
echo ""
echo "🔐 Making scripts executable..."
for script in json_to_text.py run_worker.sh verify_audio.sh; do
    if [ -f "$script" ]; then
        chmod +x "$script"
    fi
done

echo "✅ Scripts are now executable"

# Check if Claude CLI is available
echo ""
if command -v claude &> /dev/null; then
    echo "✅ Claude CLI found: $(which claude)"
else
    echo "⚠️  Claude CLI not found in PATH"
    echo "   Please install Claude Code from: https://claude.com/claude-code"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Usage examples:"
echo ""
echo "  1. Build and run the C++ TTS pipeline:"
echo "     cmake -S stream-tts-cpp -B stream-tts-cpp/build -DCMAKE_BUILD_TYPE=Release"
echo "     cmake --build stream-tts-cpp/build --target stream-tts-cpp"
echo "     ./stream-tts-cpp/build/stream-tts-cpp --speak \"Hello world\" --lang en"
echo ""
echo "  2. Run autonomous worker (JSON->audio pipe):"
echo "     ./run_worker.sh"
echo ""
echo "  3. Verify audio output:"
echo "     ./verify_audio.sh"
echo ""
echo "Configuration:"
echo "  - Edit stream-tts-cpp/config/default.yaml to:"
echo "    • Change voice (tts.voice)"
echo "    • Change language (tts.language)"
echo "    • Enable/disable translation (translation.enabled)"
echo ""
