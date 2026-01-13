# Usage Guide: Voice Processing for Claude Code

Complete guide to using live TTS with Japanese translation for Claude Code.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Testing](#testing)
3. [Usage Modes](#usage-modes)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

---

## Initial Setup

### 1. Install Dependencies

```bash
cd voice
./setup.sh
```

This creates a virtual environment and installs all required Python packages.

### 2. Verify Installation

```bash
./test_tts.sh
```

You should hear Japanese speech if everything is working.

---

## Testing

### Quick Test

Test translation and TTS without Claude:

```bash
source venv/bin/activate
echo "Hello, how are you today?" | python3 tts_translate_stream.py
```

You should hear: "こんにちは、今日はお元気ですか？"

### Test JSON Formatter

```bash
echo '{"type":"text","content":[{"type":"text","text":"Testing the system"}]}' | ./json_to_text.py
```

---

## Usage Modes

### Mode 1: One-off Command with TTS

Run a single Claude Code command with TTS:

```bash
./claude_code_tts.sh "explain the authentication flow in this project"
```

Claude will:
1. Respond to your question
2. Translate output to Japanese
3. Speak the response

### Mode 2: Interactive Session with TTS

```bash
./claude_code_tts.sh
```

This starts an interactive Claude Code session with TTS enabled.

### Mode 3: Autonomous Worker with TTS

```bash
./run_worker_tts.sh
```

This runs Claude continuously in a loop, processing each response with TTS.

To provide hints to the worker:

```bash
echo "Focus on the database schema" > HINT.txt
```

The worker will read and consume the hint on the next iteration.

### Mode 4: Regular Worker (No TTS)

```bash
./run_worker.sh
```

Original autonomous worker mode without TTS or translation.

---

## Configuration

### Voice Settings

Edit `tts_translate_stream.py` line 31-43:

```python
# Enable/disable translation
TRANSLATE_TO_JAPANESE = True  # Set to False for English-only

# Japanese voice selection
JAPANESE_VOICE = "ja-JP-NanamiNeural"  # Female, natural

# Alternative voices:
# "ja-JP-KeitaNeural"  - Male voice
# "ja-JP-AoiNeural"    - Female, child-like
# "ja-JP-DaichiNeural" - Male, deeper

# Speech speed
SPEAK_SPEED = "+0%"  # Range: "-50%" to "+100%"
```

### List Available Voices

```bash
source venv/bin/activate
edge-tts --list-voices | grep ja-JP
```

### Translation Settings

The system uses Google Translate (via deep-translator). To change target language, edit `tts_translate_stream.py` line 61:

```python
self.translator = GoogleTranslator(source='en', target='ja')
# Change 'ja' to any supported language code:
# 'es' = Spanish, 'fr' = French, 'de' = German, etc.
```

---

## Troubleshooting

### No Audio Output

**Issue**: Script runs but no sound

**Solutions**:
1. Check system volume and audio output device
2. Verify ffmpeg or afplay is available:
   ```bash
   which ffplay  # Linux/macOS with ffmpeg
   which afplay  # macOS default
   ```
3. Install ffmpeg if needed:
   ```bash
   brew install ffmpeg  # macOS
   sudo apt install ffmpeg  # Linux
   ```

### Translation Errors

**Issue**: "Translation error" messages

**Solutions**:
1. Check internet connection (Google Translate requires connectivity)
2. Rate limiting: Add delays between requests
3. The system automatically falls back to English if translation fails

### PyAudio Errors

**Issue**: Warnings about PyAudio

**Action**: PyAudio is optional. The system uses pydub with system audio players (afplay/ffplay). You can safely ignore PyAudio warnings.

### Import Errors

**Issue**: "ModuleNotFoundError"

**Solution**: Activate virtual environment first:
```bash
source venv/bin/activate
```

Or ensure scripts that need venv activate it automatically (claude_code_tts.sh, run_worker_tts.sh).

### Slow Performance

**Issue**: Long delays before speech

**Solutions**:
1. Use a faster voice model (already using the fastest)
2. Reduce SPEAK_SPEED
3. Translation adds ~1-2 seconds; disable if not needed:
   ```python
   TRANSLATE_TO_JAPANESE = False
   ```

---

## Advanced Usage

### Custom Audio Format

Edit `tts_translate_stream.py` around line 87:

```python
# Change MP3 to WAV for better quality (larger file)
temp_file = os.path.join(self.temp_dir, f"speech_{datetime.now().timestamp()}.wav")
communicate = edge_tts.Communicate(text, voice, rate=SPEAK_SPEED)
await communicate.save(temp_file)
audio = AudioSegment.from_wav(temp_file)  # Change from_mp3 to from_wav
```

### Filter What Gets Spoken

Edit `should_speak()` function in `tts_translate_stream.py` around line 134:

```python
def should_speak(text, tool_name=None):
    """Determine if text should be spoken"""
    # Don't speak very short messages
    if len(text.strip()) < 20:  # Increase threshold
        return False

    # Don't speak tool outputs
    if tool_name:
        return False

    # Only speak if contains certain keywords
    keywords = ["error", "complete", "success"]
    if any(kw in text.lower() for kw in keywords):
        return True

    return True  # Speak everything else
```

### Batch Processing

Process multiple Claude outputs:

```bash
for file in logs/*.jsonl; do
    cat "$file" | ./json_to_text.py | ./tts_translate_stream.py
done
```

### Save Audio Files

Modify `tts_translate_stream.py` to save instead of play:

```python
# Around line 90, comment out play and save file:
# play(audio)
output_path = f"audio_output_{datetime.now().timestamp()}.mp3"
audio.export(output_path, format="mp3")
print(f"Saved: {output_path}")
```

### Integration with Other Tools

Pipe any text through the TTS system:

```bash
# Git commit messages
git log --oneline -5 | ./tts_translate_stream.py

# Code reviews
cat review.txt | ./tts_translate_stream.py

# Documentation
cat README.md | ./tts_translate_stream.py
```

---

## Performance Tips

1. **Reduce Latency**: Disable translation if you don't need it
2. **Batch Requests**: Queue multiple text segments for better efficiency
3. **Local TTS**: For offline use, consider piper-tts (requires additional setup)
4. **Caching**: Edge TTS caches voice data automatically

---

## File Structure

```
voice/
├── README.md              # Overview and quick start
├── USAGE.md               # This file - detailed usage guide
├── requirements.txt       # Runtime deps (none; see requirements-dev.txt)
├── requirements-dev.txt   # Dev/test Python dependencies
├── setup.sh               # Setup script (.venv)
├── .venv/                 # Virtual environment (created by setup.sh)
├── json_to_text.py        # JSON formatter
├── run_worker.sh          # Autonomous worker (stdin -> daemon pipe)
├── verify_audio.sh        # Audio verification helper
├── stream-tts-cpp/        # C++ implementation
└── worker_logs/           # Worker session logs
```

---

## Getting Help

### Debug Mode

Add debug output to `tts_translate_stream.py`:

```python
# Around line 150, add:
print(f"DEBUG: Processing text: {text[:100]}")
print(f"DEBUG: Should speak: {should_speak(text)}")
```

### Check Dependencies

```bash
source venv/bin/activate
pip list | grep -E "edge-tts|deep-translator|pydub"
```

### View Logs

Worker logs are saved in `worker_logs/`:

```bash
ls -lh worker_logs/
cat worker_logs/worker_iter_1_*.jsonl | head -50
```

---

## Examples

### Example 1: Code Review with TTS

```bash
./claude_code_tts.sh "review the code in src/auth.py for security issues"
```

### Example 2: Continuous Development Session

```bash
# Start worker
./run_worker_tts.sh

# In another terminal, provide hints:
echo "Implement the user registration feature" > HINT.txt
# Wait for iteration to complete
echo "Now add tests for the registration" > HINT.txt
# Continue...
```

### Example 3: English-Only Mode

```bash
# Edit tts_translate_stream.py
sed -i '' 's/TRANSLATE_TO_JAPANESE = True/TRANSLATE_TO_JAPANESE = False/' tts_translate_stream.py

# Run
./claude_code_tts.sh "explain async/await in Python"
```

---

## Next Steps

- Experiment with different Japanese voices
- Adjust speech speed for your preference
- Try different translation target languages
- Integrate with your CI/CD pipeline for spoken build notifications

---

## License

Copyright 2025 Andrew Yates. All rights reserved.
