# Phase 2 TTS Options Analysis

**Date**: 2025-11-24
**Current Problem**: Edge-TTS returning 403 errors (API access issue)
**Phase 1 TTS latency**: 577ms (macOS `say` command)
**Target**: < 100ms

---

## Issue: Edge-TTS 403 Error

**Error message**:
```
403, message='Invalid response status',
url='wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?...'
```

**Cause**: Microsoft has changed authentication requirements for Edge TTS API
**Status**: Common issue with edge-tts library as of late 2024/early 2025

**Potential solutions**:
1. Update to latest edge-tts version
2. Use alternative TTS service
3. Implement local TTS solution

---

## TTS Options Comparison

### Option 1: macOS `say` (Current - Phase 1)
**Latency**: 577ms
**Quality**: Good
**Cost**: Free
**Pros**: Works offline, no dependencies, reliable
**Cons**: Very slow, CPU-only, not optimized
**Status**: ✅ Working (fallback)

### Option 2: Edge-TTS (Cloud API)
**Expected latency**: 100-200ms
**Quality**: Excellent
**Cost**: Free (but unreliable)
**Pros**: Fast, high quality, many voices
**Cons**: ❌ 403 errors, unreliable, requires internet
**Status**: ⚠️ Currently broken

### Option 3: gTTS (Google Text-to-Speech)
**Expected latency**: 300-500ms
**Quality**: Good
**Cost**: Free
**Pros**: Simple API, reliable, well-maintained
**Cons**: Slower than Edge-TTS, requires internet, Google dependency
**Status**: ✅ Available (already installed if needed)

**Implementation**:
```python
from gtts import gTTS
tts = gTTS(text="こんにちは", lang='ja')
tts.save("output.mp3")
```

### Option 4: pyttsx3 (Local TTS)
**Expected latency**: 400-600ms
**Quality**: Poor to moderate
**Cost**: Free
**Pros**: Fully offline, simple
**Cons**: Uses macOS `say` under the hood (same as current), low quality
**Status**: ❌ Not better than current solution

### Option 5: Coqui XTTS v2 (Local GPU)
**Expected latency**: 50-100ms (on Metal GPU)
**Quality**: Excellent (state-of-the-art)
**Cost**: Free (non-commercial) or license required
**Pros**: Very fast on GPU, high quality, offline, full control
**Cons**: Requires license acceptance, large model (~1-2GB), complex setup
**Status**: ⚠️ Requires manual license acceptance

**Licensing**: CPML (Coqui Public Model License)
- Free for non-commercial, research, educational use
- Requires commercial license for production use
- Must explicitly accept terms

**Implementation steps**:
1. Accept license in terminal: `tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --text "test"`
2. Model will download (~1.5GB)
3. Use TTS API with GPU acceleration

### Option 6: OpenAI TTS API
**Expected latency**: 200-400ms
**Quality**: Excellent
**Cost**: $$$ (paid API, ~$15 per 1M characters)
**Pros**: High quality, reliable, good Japanese support
**Cons**: Expensive, requires API key, cloud dependency
**Status**: ✅ Available (would need API key)

### Option 7: ElevenLabs API
**Expected latency**: 300-600ms
**Quality**: Excellent (best available)
**Cost**: $$$ (paid API, expensive)
**Pros**: Best quality, very natural, emotional voices
**Cons**: Very expensive, requires API key, cloud dependency
**Status**: ✅ Available (would need API key)

### Option 8: Bark (Local GPU)
**Expected latency**: 500-1000ms
**Quality**: Good
**Cost**: Free (MIT license)
**Pros**: Open source, runs on GPU, offline
**Cons**: Slow even on GPU, large model, limited language support
**Status**: ⚠️ Not optimized for speed

---

## Recommended Strategy: Hybrid Approach

Given the Edge-TTS issue, implement a **fallback hierarchy**:

### Tier 1 (Best): XTTS v2 on Metal GPU
- **Latency**: 50-100ms
- **Quality**: Excellent
- **Setup**: Manual license acceptance
- **Use case**: Main production solution (if license acceptable)

### Tier 2 (Good): gTTS
- **Latency**: 300-500ms
- **Quality**: Good
- **Setup**: Zero (already available)
- **Use case**: Fallback if XTTS v2 not available or internet is available

### Tier 3 (Fallback): macOS `say`
- **Latency**: 577ms
- **Quality**: Good
- **Setup**: Zero
- **Use case**: Always works, offline fallback

---

## Immediate Action Plan

### Step 1: Try gTTS (Quickest win)
Create `tts_worker_gtts.py` using Google TTS:
- Expected: 300-500ms (2x faster than macOS say)
- Zero setup required
- Works immediately

### Step 2: Investigate XTTS v2 License
Decision needed:
- **If non-commercial use**: Accept CPML license, get 50-100ms latency
- **If commercial use**: Evaluate if license cost is acceptable
- **If license not acceptable**: Stick with gTTS or macOS say

### Step 3: Create Adaptive Worker
Create `tts_worker_adaptive.py` that:
1. Tries XTTS v2 (if available and licensed)
2. Falls back to gTTS (if internet available)
3. Falls back to macOS say (always works)

---

## Performance Projections

### Scenario A: XTTS v2 + Optimized Translation
- Translation: 100-150ms (optimized NLLB with greedy decoding)
- TTS: 50-100ms (XTTS v2 on Metal)
- **Total: 150-250ms** ✅ Meets target

### Scenario B: gTTS + Optimized Translation
- Translation: 100-150ms (optimized)
- TTS: 300-500ms (gTTS)
- **Total: 400-650ms** ⚠️ Better than Phase 1 (877ms) but doesn't meet < 200ms target

### Scenario C: macOS say + Optimized Translation
- Translation: 100-150ms (optimized)
- TTS: 577ms (macOS say)
- **Total: 677-727ms** ⚠️ Better than Phase 1 but only marginal

---

## Recommendation

**Immediate**: Implement gTTS worker (1 hour effort, 2x TTS speedup)

**Next**: Evaluate XTTS v2 license for project:
- If acceptable: Implement XTTS v2 worker (best performance)
- If not acceptable: Optimize gTTS usage or explore OpenAI TTS

**Long-term**: Build adaptive worker with intelligent fallback

---

## Next Steps

1. ✅ Create `tts_worker_gtts.py`
2. ✅ Test gTTS performance
3. ⏳ Benchmark Phase 2 translation optimizations
4. ⏳ Decide on XTTS v2 license
5. ⏳ Implement final Phase 2 solution
6. ⏳ End-to-end benchmarking

---

**Copyright 2025 Andrew Yates. All rights reserved.**
