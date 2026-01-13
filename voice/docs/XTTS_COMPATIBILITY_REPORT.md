# XTTS v2 Compatibility Report

**Date**: 2025-11-24
**Status**: ⚠️ **BLOCKED** - PyTorch 2.9 incompatibility

---

## Executive Summary

XTTS v2 (Coqui TTS) **cannot currently be used** with PyTorch 2.9 due to security changes in `torch.load`. The model loads successfully on Metal GPU, but fails during checkpoint loading with strict `weights_only=True` enforcement.

**Recommendation**: Use alternative TTS solutions (Google TTS, Edge TTS) until either:
1. Coqui TTS updates their checkpoint format
2. PyTorch provides better compatibility mode
3. We downgrade to PyTorch < 2.6

---

## Compatibility Issues Fixed

### 1. ✅ Transformers Version (RESOLVED)

**Issue**: `BeamSearchScorer` removed from transformers 4.50+

**Error**:
```
cannot import name 'BeamSearchScorer' from 'transformers'
```

**Solution**: Downgrade transformers to 4.38.0
```bash
pip install 'transformers==4.38.0'
```

**Status**: ✅ **FIXED** - XTTS now imports successfully

---

## Compatibility Issues Remaining

### 2. ⚠️ PyTorch 2.9 weights_only (BLOCKING)

**Issue**: PyTorch 2.6+ changed default `torch.load(weights_only=False → True)` for security

**Error**:
```
Weights only load failed...
WeightsUnpickler error: Unsupported global: GLOBAL TTS.tts.configs.xtts_config.XttsConfig
```

**Root Cause**:
- XTTS checkpoints contain custom Python objects (XttsConfig)
- PyTorch 2.6+ blocks loading custom objects by default
- Monkeypatching `torch.load` doesn't work for all TTS internal code paths

**Attempted Solutions**:
1. ❌ Monkeypatch torch.load in worker script - incomplete coverage
2. ❌ torch.serialization.add_safe_globals - requires changes to TTS library
3. ⏳ Downgrade PyTorch (not tested - may break Metal GPU support)

**Status**: ⚠️ **BLOCKED** - No simple workaround found

---

### 3. ⚠️ XTTS Speaker Requirement (MINOR)

**Issue**: XTTS v2 is a multi-speaker model requiring speaker embeddings

**Error**:
```
Model is multi-speaker but no `speaker` is provided.
```

**Solution**: Provide speaker_wav or speaker embedding:
```python
tts.tts(
    text="こんにちは",
    language="ja",
    speaker_wav="/path/to/japanese_voice.wav"  # 6-10 second reference
)
```

**Status**: ⏳ **SOLVABLE** - Once PyTorch issue resolved

---

## Test Results

### Model Loading
- ✅ Model downloads successfully (1.8GB)
- ✅ Imports work with transformers 4.38.0
- ✅ Metal GPU (MPS) detected and available
- ✅ Model initialization starts (~17s load time)
- ❌ Checkpoint loading fails (PyTorch 2.9 incompatibility)

### Performance (Projected)
Based on XTTS specifications and M4 Max hardware:
- **Model load**: 15-20s (one-time)
- **Inference**: 50-80ms per sentence (estimated)
- **Quality**: Excellent (near-human)
- **Language**: Multilingual including Japanese

**Current**: Cannot test due to loading failure

---

## Dependency Versions

### Current Environment
```
Python: 3.11
PyTorch: 2.9.1 (with Metal GPU support)
Transformers: 4.38.0 (downgraded from 4.57.2)
TTS (Coqui): 0.22.0
```

### Known Working Combinations
According to community reports:
- PyTorch 2.1.x + Transformers 4.35.x + TTS 0.22.0 ✅
- PyTorch 2.2.x + Transformers 4.38.x + TTS 0.22.0 ✅
- PyTorch 2.6+ + Any Transformers + TTS 0.22.0 ❌

---

## Workarounds Considered

### Option A: Downgrade PyTorch (⚠️ RISKY)
```bash
pip install 'torch==2.2.0' 'torchaudio==2.2.0' 'torchvision==0.17.0'
```

**Pros**:
- Should resolve weights_only issue
- Known working configuration

**Cons**:
- May break Metal GPU (MPS) support
- Older CUDA/MPS backends
- Not tested on M4 Max

**Status**: Not attempted (risk too high)

---

### Option B: Patch TTS Library (🔧 COMPLEX)
Modify TTS source code to use `weights_only=False`:

```python
# In venv/lib/python3.11/site-packages/TTS/utils/io.py
def load_fsspec(path, map_location=None, **kwargs):
    with fs.open(path, "rb") as f:
        return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
```

**Pros**:
- Keeps PyTorch 2.9
- Maintains Metal GPU support

**Cons**:
- Modifies installed package
- May break on TTS updates
- Security risk acknowledged

**Status**: Not attempted (maintenance burden)

---

### Option C: Use Alternative TTS (✅ RECOMMENDED)

**Google TTS (gTTS)** - Current solution:
- ✅ Works with PyTorch 2.9
- ✅ Excellent Japanese quality
- ✅ Fast (192ms per sentence)
- ⚠️ Requires internet
- ⚠️ Cloud-based (not local GPU)

**Microsoft Edge TTS**:
- ✅ Works with any PyTorch
- ✅ Excellent voices
- ✅ Fast (200-300ms)
- ⚠️ Requires internet
- ⚠️ Cloud-based

**Apple CoreSpeech (say command)**:
- ✅ Local, no dependencies
- ✅ Works offline
- ⚠️ Slower (577ms)
- ⚠️ Lower quality

**Status**: ✅ Using gTTS successfully (Phase 1.5)

---

## Recommendations

### Immediate (This Week)
1. ✅ **Continue using Google TTS** - 192ms latency, excellent quality
2. ⏳ Monitor Coqui TTS project for PyTorch 2.6+ support
3. ⏳ Test Edge TTS as alternative (similar quality, no PyTorch dependency)

### Short-term (Next Month)
1. Re-evaluate XTTS when:
   - Coqui releases TTS 0.23+ with PyTorch 2.6+ support
   - PyTorch provides better compatibility mode
   - Community finds reliable workaround
2. Consider PyTorch downgrade if Metal GPU confirmed compatible

### Long-term (Quarter)
1. Evaluate other local TTS options:
   - Piper TTS (fast, lightweight)
   - StyleTTS 2 (high quality)
   - VITS models (good quality/speed tradeoff)
2. Consider C++/Metal direct implementation for maximum performance

---

## Current Pipeline Performance

Using **Google TTS** (Phase 1.5 Optimized):
```
Total latency: 346ms
├─ Translation (NLLB-200): 154ms
├─ TTS (Google): 192ms
└─ Audio playback: realtime
```

**Target with XTTS** (if fixed):
```
Total latency: ~220ms (estimated)
├─ Translation: 154ms
├─ TTS (XTTS local GPU): 50-80ms
└─ Audio playback: realtime
```

**Improvement potential**: ~126ms saved (36% faster)

---

## Conclusion

XTTS v2 is **currently unusable** due to PyTorch 2.9 incompatibility. The transformers issue was successfully resolved, but the PyTorch `weights_only` security change is a harder blocker.

**Recommended path forward**:
1. ✅ Continue Phase 1.5 optimizations with Google TTS
2. ✅ Achieve 150-200ms target through other optimizations:
   - INT8 quantization for translation
   - Pipeline parallelization
   - Batch processing
3. ⏳ Revisit XTTS in 1-2 months when compatibility improves

**Current Status**: **Phase 1.5 COMPLETE** (346ms latency, 2.5x faster than baseline)

**Next milestone**: INT8 quantization + parallelization → 150-200ms target

---

**Copyright 2025 Andrew Yates. All rights reserved.**
