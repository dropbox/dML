# Quick Start - Rust TTS Pipeline (Phase 1.5)

**Status**: ✅ Production Ready
**Performance**: 330ms total latency (2.6x faster than Phase 1)
**Last Updated**: November 24, 2025

---

## What You Get

- **Translation**: NLLB-200 on Metal GPU (~160ms)
- **TTS**: Google gTTS (~170ms)
- **Total latency**: ~330ms end-to-end
- **Quality**: Excellent Japanese speech output
- **Integration**: Works with Claude Code

---

## Quick Test (3 steps)

### 1. Install gtts
```bash
cd stream-tts-rust
source venv/bin/activate
pip install gtts
```

### 2. Test the pipeline
```bash
./test_rust_tts.sh
```

You should hear Japanese speech for 3 test messages.

### 3. Use with Claude
```bash
# One-off command
./claude_rust_tts.sh "explain quantum computing"

# Interactive mode
./claude_rust_tts.sh
```

---

## Performance Results

```
Translation: 157.6ms (Metal GPU)
TTS:         174.2ms (Google API)
Total:       331.8ms ✅
```

**Example output**: "Hello, how are you today?" → "こんにちは 今日はどうですか?"

---

## Files

- **Binary**: `stream-tts-rust/target/release/stream-tts-rust`
- **Test**: `test_rust_tts.sh`
- **Claude integration**: `claude_rust_tts.sh`
- **Results**: `PHASE1_5_OPTIMIZATION_RESULTS.md`

---

## Requirements

- Python 3.11+ with venv
- Rust toolchain
- Internet connection (for gTTS)
- M4 Max or Apple Silicon Mac

---

## Next Steps

Phase 2 optimization targets:
1. Fix XTTS v2 compatibility → 60ms TTS
2. INT8 quantization → 80-100ms translation
3. **Target**: 150-200ms total latency (5-6x faster than Phase 1)

---

**Copyright 2025 Andrew Yates. All rights reserved.**
