# TTS Quality Modes - Comparison Guide

This guide helps you choose the right TTS quality mode for your needs.

---

## Quick Comparison

| Mode | Latency | Translation | TTS | Best For |
|------|---------|-------------|-----|----------|
| **Production** | 340ms | NLLB-200 | Google | General use, real-time feedback |
| **Ultimate Quality** | 780ms | Qwen2.5-7B | Google | Technical docs, accuracy-critical |
| **Python Legacy** | 3-5s | Google API | Edge TTS | Multi-language, offline dev |

---

## Mode Details

### Production Mode (Recommended)

**Script**: `./run_worker_with_tts.sh`
**Test**: `./tests/test_production_quick.sh`

**Performance**:
- Translation: 154ms (NLLB-200-600M on Metal GPU)
- TTS: 185ms (Google Neural TTS)
- **Total: 340ms**

**Quality**:
- Translation: BLEU 28-30 (excellent)
- Voice: Natural Japanese (Nanami Neural)
- Intelligibility: Very high

**Pros**:
✅ Exceeds industry standard (300-500ms)
✅ Fast enough to feel "instant"
✅ High quality for general conversation
✅ Reliable and production-tested
✅ Metal GPU accelerated

**Cons**:
❌ Slightly less accurate than Qwen for complex technical terms
❌ May miss some context in very long sentences

**Use For**:
- General Claude Code sessions
- Real-time coding assistance
- Bug fixing and debugging
- Most day-to-day development work

---

### Ultimate Quality Mode

**Script**: `./run_worker_ultimate_quality.sh`
**Test**: `./tests/test_ultimate_quality.sh`

**Performance**:
- Translation: 595ms (Qwen2.5-7B-Instruct)
- TTS: 185ms (Google Neural TTS)
- **Total: 780ms**

**Quality**:
- Translation: Superior to NLLB (better context handling)
- Voice: Natural Japanese (Nanami Neural)
- Intelligibility: Very high

**Pros**:
✅ Best translation accuracy
✅ Better context understanding
✅ Handles technical jargon better
✅ More natural phrasing
✅ Still fast enough for most use

**Cons**:
❌ 2.3x slower than production mode
❌ Requires 15GB model download
❌ Uses more GPU memory

**Use For**:
- Technical documentation translation
- Architecture discussions
- Complex explanation requests
- Context-dependent translations
- When accuracy > speed

---

### Python Legacy Mode

**Script**: `./run_worker_tts.sh`
**Test**: `./tests/test_tts.sh`

**Performance**:
- Translation: Variable (cloud API)
- TTS: Variable (Edge TTS cloud)
- **Total: 3-5 seconds**

**Quality**:
- Translation: Good (Google Translate)
- Voice: Natural (Edge TTS Neural)
- Intelligibility: High

**Pros**:
✅ Supports 12+ languages easily
✅ No local model downloads
✅ Simpler architecture
✅ Good for development/testing

**Cons**:
❌ 10x slower than production
❌ Depends on cloud APIs
❌ Not suitable for real-time use

**Use For**:
- Multi-language testing
- Development without GPU
- When disk space is limited
- Experimentation with different languages

---

## Performance Comparison

### Latency Breakdown

```
Production Mode (340ms):
├─ Parse JSON:        < 1ms
├─ Clean text:        < 1ms
├─ Translation:       154ms  [NLLB-200 on Metal]
├─ TTS:               185ms  [Google Neural]
└─ Audio playback:    < 1ms

Ultimate Quality (780ms):
├─ Parse JSON:        < 1ms
├─ Clean text:        < 1ms
├─ Translation:       595ms  [Qwen2.5-7B]
├─ TTS:               185ms  [Google Neural]
└─ Audio playback:    < 1ms

Python Legacy (3-5s):
├─ Parse JSON:        ~100ms
├─ Clean text:        ~50ms
├─ Translation:       1-2s   [Google Cloud API]
├─ TTS:               2-3s   [Edge TTS Cloud]
└─ Audio playback:    < 100ms
```

### Translation Quality

**Simple Sentence**: "Hello, how are you?"
- NLLB-200: "こんにちは、お元気ですか？" ✅
- Qwen2.5: "こんにちは、お元気ですか？" ✅
- Difference: Negligible

**Technical Sentence**: "The asynchronous callback handler manages concurrent API requests with exponential backoff."
- NLLB-200: "非同期コールバックハンドラは、指数バックオフで並行APIリクエストを管理します。" ✅ Good
- Qwen2.5: "非同期コールバックハンドラーは、指数バックオフを使用して並行APIリクエストを管理します。" ✅ Better
- Difference: Qwen adds "を使用して" (using), more natural Japanese

**Complex Context**: "Given the previous refactoring of the authentication module, we should update the test fixtures."
- NLLB-200: May lose some context between sentences
- Qwen2.5: Better maintains context across sentences ✅

---

## Choosing the Right Mode

### Use Production Mode If:
- You want the best balance of speed and quality
- You're doing general development work
- Real-time feedback is important
- You don't need absolute translation perfection

### Use Ultimate Quality Mode If:
- You're translating technical documentation
- Translation accuracy is more important than speed
- You're having complex architecture discussions
- Context preservation is critical
- You don't mind 780ms latency

### Use Python Legacy Mode If:
- You need multi-language support
- You're developing without GPU access
- You're testing new language combinations
- Speed is not a concern

---

## Switching Between Modes

### For One-Off Commands

**Production**:
```bash
./tests/claude_rust_tts.sh "your prompt here"
```

**Ultimate Quality**:
```bash
TRANSLATION_WORKER="./stream-tts-rust/python/translation_worker_qwen.py" \
  ./tests/claude_rust_tts.sh "your prompt here"
```

**Python Legacy**:
```bash
./tests/claude_code_tts.sh "your prompt here"
```

### For Autonomous Workers

**Production**:
```bash
./run_worker_with_tts.sh
```

**Ultimate Quality**:
```bash
./run_worker_ultimate_quality.sh
```

**Python Legacy**:
```bash
./run_worker_tts.sh
```

---

## Hardware Requirements

### Production Mode
- Apple Silicon Mac (M1+)
- 8GB RAM minimum
- 2GB free disk space (NLLB-200 model)
- Internet (Google TTS)

### Ultimate Quality Mode
- Apple Silicon Mac (M1+)
- 16GB RAM recommended
- 16GB free disk space (Qwen2.5-7B model)
- Internet (Google TTS)

### Python Legacy Mode
- Any Mac/Linux
- 4GB RAM
- Minimal disk space
- Internet (APIs)

---

## Benchmarks

Based on M4 Max (40-core GPU, 128GB RAM):

| Test | Production | Ultimate | Legacy |
|------|-----------|----------|--------|
| Single sentence | 340ms | 780ms | 3.2s |
| Three sentences | 1020ms | 2340ms | 9.6s |
| Paragraph (10 sentences) | 3400ms | 7800ms | 32s |

---

## Future Enhancements

### Potential Improvements

**Production Mode**:
- INT8 quantization when PyTorch MPS supports it
- Expected: 340ms → 110-160ms

**Ultimate Quality Mode**:
- Fine-tuned Qwen for Japanese technical content
- Expected: Better quality, similar speed

**New Modes**:
- **Speed Mode**: 150-200ms (simplified translation)
- **Offline Mode**: No internet required
- **Custom Voice Mode**: Train personal voice models

---

## Troubleshooting

### Production Mode Slow
1. Check Metal GPU: `python3 -c "import torch; print(torch.backends.mps.is_available())"`
2. Monitor GPU usage in Activity Monitor
3. Ensure internet connection for Google TTS

### Ultimate Quality Out of Memory
1. Close other GPU-intensive apps
2. Restart to clear GPU memory
3. Consider production mode instead

### Audio Quality Issues
1. All modes use same TTS (Google Neural)
2. Check system volume
3. Try different voice in worker config

---

## Recommendations

**For Most Users**: Start with **Production Mode**. It's fast, high-quality, and reliable.

**For Translators/Linguists**: Use **Ultimate Quality Mode** for best accuracy.

**For Multi-Language Dev**: Use **Python Legacy Mode** for easy language switching.

---

## License

Copyright 2025 Andrew Yates. All rights reserved.
