# Next Steps - Voice TTS System

**Date**: 2025-11-24
**Current Status**: Production system working, optimization opportunities available

---

## 🎯 Current Production System

**Working Now**: Rust + Python Pipeline
```bash
./stream-tts-rust/target/release/stream-tts-rust
```

**Performance** (measured):
- Translation: 99ms (NLLB-200 on Metal GPU)
- TTS: 447ms (macOS `say` command)
- **Total: ~546ms**

**Quality**:
- Translation: Excellent (NLLB-200-distilled-600M)
- Voice: Good (macOS Otoya voice, 185 WPM)

---

## 📊 Performance Analysis

The current bottleneck is **TTS (447ms)**, not translation (99ms).

### Translation Performance ✅
- Current: 99ms
- Target: 70ms
- Status: **GOOD ENOUGH** (only 29ms to optimize)

### TTS Performance ⚠️
- Current: 447ms (macOS `say`)
- Target: 80ms
- Gap: **367ms optimization needed**

---

## 🚀 Optimization Options

### Option 1: Keep Current System (RECOMMENDED)
**Total latency: 546ms**

**Pros**:
- ✅ Working right now
- ✅ No additional work
- ✅ Reliable and tested
- ✅ 546ms is acceptable for voice feedback

**Cons**:
- ⚠️ Not the fastest possible

**Recommendation**: **Use this unless you NEED sub-200ms latency**

---

### Option 2: Optimize TTS Only (MEDIUM EFFORT)
**Target latency: 150-250ms**

Focus on faster TTS since that's the bottleneck:

**2A. Try Alternative TTS Engines**:
1. Test Google TTS (gTTS) - may be faster
2. Try ElevenLabs API - highest quality, 100-200ms
3. Try StyleTTS2 on Metal - 80-150ms if it works

**2B. Optimize macOS `say`**:
- Use AVFoundation API directly instead of `say` command
- Eliminate subprocess overhead (~50-100ms savings)
- Target: 300-350ms total

**Time**: 2-4 hours
**Risk**: Medium (may not achieve significant speedup)
**Reward**: 2-3x faster if successful

---

### Option 3: Complete C++ Implementation (HIGH EFFORT)
**Target latency: 70-130ms**

**Status**: Code written but not working yet

**Remaining Work**:
1. Debug C++ audio playback (AVFoundation integration)
2. Test and benchmark
3. Add Core ML translation support
4. Integration testing

**Time**: 4-8 hours
**Risk**: High (may encounter Apple framework issues)
**Reward**: 4-8x faster if successful

---

### Option 4: Explore Ultra-Fast TTS (RESEARCH)
**Target latency: 50-100ms**

Investigate cutting-edge options:
1. **LightSpeed TTS** - Specialized for low latency
2. **Piper TTS** - Fast local neural TTS
3. **Custom ONNX TTS** - Convert and optimize for Metal

**Time**: 1-2 days research + implementation
**Risk**: High (may not exist or work well)
**Reward**: Ultra-fast if successful

---

## 💡 Recommendations

### For Immediate Use
**→ Use Option 1: Current Production System**

The 546ms latency is perfectly acceptable for voice feedback in a coding assistant. It's:
- Fast enough that responses feel immediate
- Much faster than reading the text yourself
- Reliable and working right now

```bash
# Test it now
./tests/test_optimized_pipeline.sh

# Use with Claude
claude code "say hello" --output-format stream-json | \
  ./stream-tts-rust/target/release/stream-tts-rust
```

### If Speed is Critical
**→ Try Option 2A: Alternative TTS First**

Quick experiments to find faster TTS:

**Test 1: gTTS** (Google TTS)
```bash
# Already implemented in: stream-tts-rust/python/tts_worker_gtts.py
# Switch TTS worker in config to test
```

**Test 2: AVFoundation Direct API**
```bash
# Fix the C++ implementation's TTS component only
# Skip translation for now
```

Expected outcome: 200-300ms total (2x faster)

### If You Want Maximum Performance
**→ Go for Option 3: Complete C++ System**

But be prepared to debug Apple frameworks and spend several hours.

---

## 📋 Action Plan

### Recommended Path (Pragmatic)

**Phase 1: Use What Works** (0 minutes)
```bash
./tests/test_optimized_pipeline.sh
```
- You have a working system
- 546ms is good performance
- Ship it and use it

**Phase 2: Quick TTS Optimization** (1-2 hours, if needed)
- Try gTTS instead of `say`
- Try direct AVFoundation API
- Measure and compare
- Pick the fastest

**Phase 3: C++ Deep Dive** (Only if absolutely necessary)
- Debug AVFoundation integration
- Get audio working
- Full benchmarking
- Only do this if you need < 200ms

---

## 🎯 Decision Matrix

| Your Priority | Best Option | Expected Result | Time |
|---------------|-------------|-----------------|------|
| **Just use it now** | Option 1 | 546ms, reliable | 0 min |
| **Moderately faster** | Option 2A | 200-300ms, moderate effort | 2-4 hrs |
| **Maximum speed** | Option 3 | 70-130ms, high effort | 4-8 hrs |
| **Cutting edge** | Option 4 | 50-100ms, research needed | 1-2 days |

---

## 🔧 Quick Commands

### Test Current System
```bash
cd /Users/ayates/voice
./tests/test_optimized_pipeline.sh
```

### Use with Claude (Production)
```bash
claude code "explain async/await" --output-format stream-json | \
  ./stream-tts-rust/target/release/stream-tts-rust
```

### Try C++ System (Debug)
```bash
cd stream-tts-cpp
./build.sh
./test.sh
```

---

## 📈 Performance Comparison

| System | Translation | TTS | Total | Status |
|--------|-------------|-----|-------|--------|
| **Current (Rust+Python)** | 99ms | 447ms | **546ms** | ✅ Working |
| Python-only (old) | 300ms | 577ms | 877ms | ✅ Replaced |
| C++ (target) | 15-30ms | 50-100ms | 70-130ms | 🟡 Not working |
| Optimized TTS (possible) | 99ms | 150-250ms | 250-350ms | 🔵 Could try |

---

## 🎤 Bottom Line

**Your current system at 546ms is GOOD**.

Industry standard for voice assistants:
- Alexa/Siri: 300-800ms
- ChatGPT Voice: 500-1000ms
- Your system: **546ms** ✅

You're already competitive. Further optimization is optional.

**Recommendation**: Use what you have. Optimize only if you need it.

---

**Copyright 2025 Andrew Yates. All rights reserved.**
