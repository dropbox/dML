# XTTS v2 Integration Blockers

**Date**: 2025-11-24
**Status**: ⚠️ BLOCKED - Multiple compatibility issues

---

## Summary

Attempted to integrate Coqui XTTS v2 for GPU-accelerated TTS to achieve < 100ms latency target. Encountered multiple technical blockers with PyTorch 2.9 and transformers compatibility.

---

## Blockers Encountered

### Blocker 1: Transformers Version Conflict
**Issue**: XTTS v2 requires `BeamSearchScorer` which was removed in transformers 4.46+

**Error**:
```
cannot import name 'BeamSearchScorer' from 'transformers'
```

**Attempted fix**: Downgraded to transformers 4.45.2
**Result**: Partial success - import error resolved

### Blocker 2: PyTorch 2.9 weights_only Default
**Issue**: PyTorch 2.9 changed default `torch.load(weights_only=True)` for security

**Error**:
```
Weights only load failed. This file can still be loaded, to do so you have two options...
WeightsUnpickler error: Unsupported global: GLOBAL TTS.tts.configs.xtts_config.XttsConfig
was not an allowed global by default.
```

**Root cause**: TTS library uses custom config classes that aren't in PyTorch's safe globals list
**Potential fixes**:
1. Downgrade PyTorch to < 2.6 (loses Metal optimizations)
2. Patch TTS library to use `torch.serialization.add_safe_globals()`
3. Fork TTS library with PyTorch 2.9 compatibility
4. Use alternative TTS solution

### Blocker 3: Python 3.14 Compatibility (Original)
**Issue**: Coqui TTS doesn't officially support Python 3.14
**Solution**: Created Python 3.11 venv ✅ Resolved
**Impact**: Required separate environment management

---

## Technical Analysis

### Why XTTS v2 is Challenging

1. **Stale dependency**: Coqui TTS was last updated in 2023
2. **PyTorch evolution**: PyTorch 2.6+ introduced breaking security changes
3. **Transformers evolution**: API changes broke compatibility
4. **No active maintenance**: Coqui AI company ceased operations

### Compatibility Matrix

| Component | Required | Available | Compatible |
|-----------|----------|-----------|------------|
| Python | < 3.12 | 3.11 ✅ | ✅ Yes |
| Transformers | < 4.46 | 4.45.2 ✅ | ✅ Yes |
| PyTorch | < 2.6 | 2.9.1 ❌ | ❌ No |
| TTS Library | 0.22.0 | 0.22.0 ✅ | ⚠️ Partial |

---

## Potential Solutions

### Solution A: Downgrade PyTorch (Not Recommended)
**Approach**: Use PyTorch 2.5 or earlier
**Pros**: Would likely work with XTTS v2
**Cons**:
- Loses Metal/MPS improvements in 2.9
- Loses performance optimizations
- Security regression (weights_only default exists for good reason)
- Not sustainable long-term

**Verdict**: ❌ Not worth the trade-offs

### Solution B: Patch TTS Library (Complex)
**Approach**: Fork Coqui TTS and add PyTorch 2.9 compatibility
**Steps**:
1. Fork https://github.com/coqui-ai/TTS
2. Add safe globals for all custom configs
3. Test thoroughly
4. Maintain fork

**Pros**: Could work with PyTorch 2.9
**Cons**:
- Significant maintenance burden
- Unknown other compatibility issues
- 2-3 days of work minimum

**Verdict**: ⚠️ Possible but high cost

### Solution C: Alternative TTS (Recommended)
**Approach**: Use different local TTS solution
**Options**:
1. **Piper TTS** - Fast, C++, onnx-based, actively maintained
2. **StyleTTS 2** - New, GPU-accelerated, high quality
3. **MMS-TTS** (Meta) - Multilingual, well-maintained
4. **Apple's new voices** - Native Metal, excellent quality

**Verdict**: ✅ Best path forward

### Solution D: Optimize Current Solution (Pragmatic)
**Approach**: Stick with gTTS (192ms) + optimize elsewhere
**Current performance**:
- Translation: 154ms (optimized)
- TTS: 192ms (gTTS)
- Total: 346ms (2.5x faster than Phase 1)

**Optimization opportunities**:
1. Parallelize translation + TTS for sentence batches
2. INT8 quantize translation model (154ms → ~80ms)
3. Use faster translation model (NLLB-600M → NLLB-200M)
4. Pre-cache common phrases

**Projected performance with optimizations**:
- Translation: 80ms (INT8 quantized)
- TTS: 192ms (gTTS, unchanged)
- Parallelization: -30% overlap
- **Target: ~190-220ms** 🎯 Close to 150ms target

**Verdict**: ✅ Most practical immediate path

---

## Recommendations

### Immediate (This Week)
1. **Accept current gTTS solution** (192ms is good)
2. **Focus on translation optimization**:
   - Implement INT8 quantization (154ms → 80ms target)
   - Try smaller NLLB-200M model
   - Profile Metal GPU utilization

3. **Implement pipeline parallelization**:
   - Overlap translation of next sentence while TTS plays current
   - Expected 20-30% latency reduction

### Short-term (Next 2 Weeks)
1. **Evaluate Piper TTS**:
   - C++ based, fast, ONNX runtime
   - < 50ms reported latency
   - Active maintenance
   - Easy Metal/CoreML integration

2. **Evaluate Apple's Neural TTS**:
   - Native Metal optimization
   - Excellent quality
   - May require macOS 14+ APIs
   - Zero external dependencies

### Long-term (Month 2-3)
1. **Consider StyleTTS 2** if quality/speed not met
2. **Explore commercial TTS APIs** if budget allows (OpenAI, ElevenLabs)
3. **Revisit XTTS v2** if Coqui project gets revived or fork is maintained

---

## Current Best Performance

**Phase 1.5 Optimized** (as of 2025-11-24 evening):
- Translation: 154ms (NLLB-200-600M, bfloat16, greedy)
- TTS: 192ms (Google TTS/gTTS)
- **Total: 346ms**
- **Improvement vs Phase 1**: 2.5x faster (877ms → 346ms)

**Gap to target**:
- Target: 150ms
- Current: 346ms
- Gap: 196ms (2.3x over)

**Realistic near-term target with optimizations**:
- INT8 quantization: -70ms
- Pipeline parallelization: -60ms
- **Projected: 216ms** (44% over target, but 4x faster than Phase 1)

---

## Lessons Learned

1. **Dependency freshness matters**: XTTS v2 (2023) can't keep up with PyTorch (2025)
2. **Security updates break things**: PyTorch 2.6+ weights_only change is legitimate but breaks old code
3. **Fork maintenance is expensive**: Would need ongoing work to maintain XTTS v2 fork
4. **"Good enough" is often better**: 346ms → 216ms is excellent progress, XTTS v2 might not be worth the complexity
5. **Pragmatic optimization wins**: Focus on what works (translation speed) vs fighting compatibility issues

---

## Next Steps

**Recommended path forward**:

1. ✅ **Accept gTTS solution** - 192ms is reasonable, reliable, zero-setup
2. ⏳ **Optimize translation** - INT8 quantization for 2x speedup
3. ⏳ **Add parallelization** - Overlap processing for 30% improvement
4. ⏳ **Benchmark end-to-end** - Validate 200-220ms total target
5. ⏳ **Document Phase 2 complete** - Write PHASE2_COMPLETE.md
6. ⏳ **Research Piper TTS** - Evaluate for Phase 3 if sub-150ms needed

**Decision**: Move forward with Solution D (optimize current system) rather than fighting XTTS v2 compatibility

---

**Copyright 2025 Andrew Yates. All rights reserved.**
