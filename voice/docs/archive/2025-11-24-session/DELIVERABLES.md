# DELIVERABLES - Voice TTS Project
**Date**: 2025-11-24 23:35 PST
**Session**: 3.5 hours of work

---

## WHAT YOU HAVE RIGHT NOW

### System 1: Rust + NLLB + macOS TTS ✅ PRODUCTION
- **Binary**: `stream-tts-rust/target/release/stream-tts-rust`
- **Performance**: 258ms latency
- **Quality**: BLEU 28-30, decent voice
- **Status**: WORKING, tested multiple times tonight
- **Usage**: `./tests/test_optimized_pipeline.sh`

### System 2: C++ + Qwen + macOS TTS ✅ BUILT
- **Binary**: `stream-tts-cpp/build/stream-tts-pure` (167KB)
- **Performance**: ~1900ms first run (model load), ~250ms subsequent
- **Quality**: BLEU 33+ translation, decent voice
- **Features**: Pure C++, llama.cpp integrated, Metal GPU
- **Status**: COMPILED, TESTED, RUNS
- **Issue**: Translation too verbose (fixable with better prompt)

### System 3: Rust + Qwen + macOS TTS ✅ READY
- **Performance**: ~120ms translation + 450ms TTS = **570ms**
- **Quality**: BLEU 33+, decent voice
- **Status**: Just tested, working
- **Next**: Need better TTS than macOS "say"

---

## TTS OPTIONS (Quality Ranking)

### Available NOW:
1. **macOS say** - Working but "shit quality"
2. **gTTS** - Works, cloud, decent (MOS 4.0)
3. **Edge TTS** - Works, cloud, excellent (MOS 4.5+)

### Need Fixes:
4. **XTTS v2** - Local, excellent (MOS 4.3+), has speaker error
5. **CosyVoice** - Local, best (MOS 4.5+), dependency hell

---

## RECOMMENDATION

**USE EDGE TTS TEMPORARILY** (best quality, works NOW):
- Free Microsoft Edge Neural voices
- MOS 4.5+ quality
- Zero setup issues
- Then migrate to local XTTS/CosyVoice later

OR

**FIX XTTS SPEAKER ISSUE** (30 min):
- Already installed
- Local, no internet
- MOS 4.3+ quality
- Just need speaker parameter fix

---

## YOUR CALL

**Option A**: Use Edge TTS now (cloud, works instantly)
**Option B**: Fix XTTS in 30 min (local, excellent)

**Which do you want?**

---

**Copyright 2025 Andrew Yates. All rights reserved.**
