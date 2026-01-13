# Phase 2 Implementation Plan

**Date**: 2025-11-24
**Status**: Planning
**Goal**: Reduce latency from 877ms to < 200ms (target: 100-150ms)

---

## Current Status (Phase 1 Complete)

**Performance Baseline**:
- **Total latency**: 877ms average
- **Translation**: 300ms (34% of time) - NLLB-600M on Metal
- **TTS**: 577ms (66% of time) - macOS `say` command
- **Architecture**: Rust coordinator + Python workers
- **Quality**: Excellent translations and natural speech

**Bottleneck Analysis**:
1. **TTS (577ms)**: Biggest bottleneck
   - macOS `say` command is CPU-based, slow
   - Writes to disk before playback
   - Not GPU-accelerated
   - **Improvement potential**: 5-10x speedup

2. **Translation (300ms)**: Moderate bottleneck
   - NLLB-600M model on Metal via PyTorch MPS
   - Beam search (num_beams=3) adds overhead
   - PyTorch MPS has some overhead
   - **Improvement potential**: 2-3x speedup

---

## Phase 2 Strategy: Optimize Existing Architecture

**Philosophy**: Make incremental, high-impact optimizations rather than rewriting everything.

**Target**: 100-150ms total latency (5-8x speedup from Phase 1)

---

## Priority 1: Fast TTS Replacement (HIGHEST IMPACT)

**Current**: 577ms (66% of latency)
**Target**: 50-100ms
**Expected savings**: 477-527ms

### Option A: Edge-TTS (Cloud API) ⭐ RECOMMENDED
**Pros**:
- Already installed in venv
- Fast synthesis (typically 100-200ms)
- High quality voices
- No local GPU required (frees resources for translation)
- Free, no API key needed
- Many Japanese voices available

**Cons**:
- Requires internet connection
- Potential latency variability

**Implementation**:
```python
import edge_tts
import asyncio

async def text_to_speech(text: str, voice: str = "ja-JP-NanamiNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("output.mp3")
```

**Estimated latency**: 100-200ms (3-6x faster than macOS say)

### Option B: XTTS v2 on Metal GPU
**Pros**:
- Local, no internet required
- GPU-accelerated
- High quality
- Full control

**Cons**:
- Licensing requirement (non-commercial or purchase license)
- Model download ~1-2GB
- Requires manual license acceptance
- Initial load time ~5-10 seconds
- Complex integration

**Estimated latency**: 50-100ms (5-10x faster)

### Option C: Apple Neural Engine (Core ML)
**Pros**:
- Native to macOS
- Uses Neural Engine (dedicated ML hardware)
- No licensing issues
- Offline

**Cons**:
- Requires model conversion to Core ML format
- Limited documentation
- May not support Japanese well
- Complex implementation

**Estimated latency**: 100-150ms

### ✅ Recommendation: Start with Edge-TTS
**Why**:
1. **Fastest to implement**: Already installed, simple API
2. **Good performance**: 100-200ms is 3-6x faster than current
3. **Frees GPU**: Translation can use full Metal GPU
4. **Reliable**: Microsoft infrastructure
5. **No licensing**: Free and ready to use

**Implementation time**: 1-2 hours

---

## Priority 2: Translation Optimization (MODERATE IMPACT)

**Current**: 300ms (34% of latency)
**Target**: 100-150ms
**Expected savings**: 150-200ms

### Optimizations to Try

#### 1. Greedy Decoding (Quick Win)
**Current**: `num_beams=3` (beam search)
**Change**: `num_beams=1` (greedy search)
**Expected speedup**: 2-2.5x
**Trade-off**: Slightly lower quality (likely still good enough)

```python
outputs = self.model.generate(
    **inputs,
    forced_bos_token_id=self.tgt_lang_id,
    max_length=512,
    num_beams=1,  # Changed from 3
    early_stopping=True
)
```

**Estimated latency**: 120-150ms

#### 2. Shorter Max Length
**Current**: `max_length=512`
**Change**: Analyze typical sentence lengths, reduce to 256 or 128
**Expected speedup**: 1.2-1.5x

#### 3. Float16 Precision
**Current**: `dtype=torch.float32`
**Change**: `dtype=torch.float16` or `dtype=torch.bfloat16`
**Expected speedup**: 1.5-2x
**Trade-off**: Minimal quality impact

```python
self.model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16  # Changed from float32
).to(self.device)
```

#### 4. Static KV Cache
**Implementation**: Use `use_cache=True` and reuse KV cache for similar inputs
**Expected speedup**: 1.2-1.5x for repeated patterns

#### 5. Model Quantization (INT8)
**Tools**: `optimum` library + `torch.quantization`
**Expected speedup**: 2-3x
**Trade-off**: Some quality loss
**Implementation complexity**: Medium

**Combined estimated latency**: 100-150ms (2-3x faster)

---

## Priority 3: Pipeline Optimization (LOW IMPACT)

**Current overhead**: ~14ms
**Target**: < 10ms

### Quick Wins

1. **Pre-allocate buffers** in translation worker
2. **Async audio playback** in Rust coordinator
3. **Remove unnecessary string truncation** in logs
4. **Reduce stderr logging** (keep for errors only)

**Expected savings**: 5-10ms

---

## Phase 2 Implementation Roadmap

### Week 1: TTS Optimization (Days 1-3)

#### Day 1: Edge-TTS Integration
- [ ] Create `tts_worker_edgetts.py`
- [ ] Implement async synthesis with edge-tts
- [ ] Test with Japanese text
- [ ] Measure latency
- [ ] Compare quality with macOS say

**Expected result**: 100-200ms TTS latency

#### Day 2: Translation Optimization - Quick Wins
- [ ] Change to greedy decoding (`num_beams=1`)
- [ ] Test with bfloat16 precision
- [ ] Reduce max_length to 256
- [ ] Measure latency improvement
- [ ] Verify translation quality still acceptable

**Expected result**: 120-150ms translation latency

#### Day 3: Integration & Testing
- [ ] Update Rust coordinator to use new TTS worker
- [ ] End-to-end testing with Claude output
- [ ] Performance benchmarking
- [ ] Quality validation (listen to audio, check translations)

**Expected total latency**: 220-350ms (2.5-4x faster than Phase 1)

### Week 2: Advanced Optimization (Days 4-7)

#### Day 4-5: Model Quantization
- [ ] Install `optimum` library
- [ ] Quantize NLLB model to INT8
- [ ] Benchmark quantized model
- [ ] If quality good, integrate into worker

**Expected translation latency**: 50-80ms

#### Day 6: XTTS v2 Exploration (Optional)
- [ ] Accept XTTS v2 license (if non-commercial use OK)
- [ ] Download and test XTTS v2
- [ ] Measure latency vs Edge-TTS
- [ ] If better, create alternative worker

**Expected TTS latency**: 50-100ms

#### Day 7: Final Integration & Polish
- [ ] Choose best TTS approach (Edge-TTS or XTTS v2)
- [ ] Final benchmarking
- [ ] Documentation update
- [ ] Production readiness check

**Target total latency**: 100-180ms

---

## Success Criteria

### Must Have (Phase 2 Complete)
- ✅ Total latency < 200ms average
- ✅ Translation quality remains high (spot check 10 samples)
- ✅ Audio quality remains natural
- ✅ All integration tests pass
- ✅ Rust coordinator unchanged (only worker swap)
- ✅ Documentation updated

### Nice to Have (Stretch Goals)
- 🎯 Total latency < 150ms
- 🎯 Translation < 100ms
- 🎯 TTS < 80ms
- 🎯 GPU utilization > 80%
- 🎯 Support offline mode (XTTS v2)

---

## Risk Mitigation

### Risk 1: Edge-TTS Quality Lower Than Expected
**Mitigation**: Keep macOS say worker as fallback, test with multiple voices

### Risk 2: Translation Quality Degrades with Optimizations
**Mitigation**:
- Test each optimization separately
- Keep quality validation set (10-20 examples)
- Rollback if quality unacceptable

### Risk 3: Edge-TTS Latency Too High (Network)
**Mitigation**:
- Measure in real-world conditions
- Implement timeout (fallback to local TTS)
- Consider XTTS v2 if Edge-TTS insufficient

### Risk 4: MPS/Metal Instability
**Mitigation**:
- Test with various input sizes
- Add error recovery in worker
- CPU fallback if MPS fails

---

## Performance Targets Summary

| Component | Phase 1 | Phase 2 Target | Phase 2 Stretch |
|-----------|---------|----------------|-----------------|
| Translation | 300ms | 120-150ms | 80-100ms |
| TTS | 577ms | 100-200ms | 50-80ms |
| Overhead | 14ms | < 10ms | < 5ms |
| **Total** | **891ms** | **< 200ms** | **< 150ms** |
| **Speedup** | **1x** | **4.5-7x** | **6-11x** |

---

## Next Steps

1. **Immediate** (Today):
   - Implement Edge-TTS worker
   - Test with sample Japanese text
   - Measure latency

2. **This Week**:
   - Integrate Edge-TTS into pipeline
   - Optimize translation worker (greedy + bfloat16)
   - End-to-end benchmarking

3. **Next Week**:
   - Advanced optimizations (quantization)
   - XTTS v2 exploration
   - Final tuning and polish

---

**Let's start with Priority 1: Edge-TTS Integration** 🚀

This gives us the biggest immediate improvement (3-6x TTS speedup) with minimal risk and implementation time.

---

**Copyright 2025 Andrew Yates. All rights reserved.**
