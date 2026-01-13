# Voice Configuration Guide

## Quick Start

The TTS system is now fully configurable via `config.yaml`. All settings take effect immediately when you restart the workers.

### Current Configuration

**File**: `config.yaml`

```yaml
tts:
  engine: "macos_say"
  voice: "Kyoko"      # Change this to any voice from the list below
  rate: 300           # Speech speed in words per minute
```

## Available Japanese Voices

### Premium Voices (BEST QUALITY)
- **Flo** - Female, modern, expressive
- **Eddy** - Male, clear, professional
- **Shelley** - Female, warm, natural
- **Reed** - Male, deep, authoritative
- **Rocko** - Male, energetic
- **Sandy** - Female, friendly

### Classic Voice (GOOD QUALITY)
- **Kyoko** - Female, traditional, reliable

### Character Voices
- **Grandma** - Elderly female
- **Grandpa** - Elderly male

## Speech Rate Guide

- **Normal**: 175 WPM (default for most voices)
- **Fast**: 250-280 WPM (good for power users)
- **Very Fast**: 300-350 WPM ⭐ **Current setting**
- **Maximum**: 400 WPM (may sacrifice quality)

## Testing Voices

Run the voice comparison script:

```bash
./test_voices.sh
```

This will play a test phrase with 5 different premium voices so you can compare quality.

## Changing Configuration

1. Edit `config.yaml`
2. Change the `voice` and/or `rate` fields
3. Restart the TTS pipeline

Example - Switch to male voice at very fast speed:

```yaml
tts:
  voice: "Eddy"
  rate: 320
```

## Testing Your Configuration

Test the full pipeline:

```bash
echo '{"content":[{"type":"text","text":"Testing my new voice configuration"}]}' | ./target/release/stream-tts-rust
```

## XTTS v2 Status

**Status**: Not available due to Python 3.14 compatibility

Coqui XTTS v2 (AI-powered TTS) requires Python < 3.12. Our environment uses Python 3.14, which is too new.

**Current solution**: macOS built-in voices provide excellent quality and ultra-low latency (< 100ms). Premium voices like Flo and Eddy rival commercial TTS systems.

**Future**: If needed, create a separate Python 3.11 environment for XTTS v2. However, macOS voices are currently optimal for this use case.

## Performance Metrics

With current configuration (macOS say, rate 300):

- **Synthesis**: ~50-100ms per sentence
- **Translation**: ~50-150ms (after warmup)
- **Total latency**: ~100-250ms end-to-end

This is already meeting the target of < 250ms for real-time conversational TTS.

## Tips

1. **Best Quality**: Use Flo, Shelley, or Eddy voices
2. **Fastest Speech**: Set rate to 300-350 WPM
3. **Most Natural**: Use Kyoko at 250-280 WPM
4. **Testing**: Always run `./test_voices.sh` after changes

---

Copyright 2025 Andrew Yates. All rights reserved.
