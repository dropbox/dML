# Phase 2 Optimization Plan

**Date**: 2025-11-24
**Goal**: Reduce latency from 877ms to 100-150ms

## Current Status

### Baseline Performance (Phase 1)
- **Translation**: 282ms (NLLB-200-600M, Metal GPU)
- **TTS**: 581ms (macOS `say` command)
- **Total**: 863ms

### Bottleneck Analysis
1. **TTS (67% of latency)**: macOS `say` is not GPU-accelerated and writes to disk
2. **Translation (33% of latency)**: Beam search and FP32 precision add overhead

## Optimization Strategy

### Step 1: Translation Optimization (282ms → 70ms target)

**Techniques**:
1. **Greedy decoding**: num_beams=1 instead of 3 (3x faster)
2. **BFloat16 precision**: Native M4 Max support (2x faster)
3. **Reduced max_length**: 256 instead of 512 (1.5x faster)
4. **torch.compile**: JIT optimization for Metal (1.3x faster)

**Expected speedup**: 4x (282ms → 70ms)

**Status**: In progress
- Created `translation_worker_optimized.py` with all techniques
- Installing PyTorch in venv
- Will benchmark once installation complete

### Step 2: TTS Upgrade (581ms → 60ms target)

**Approach**: Replace macOS `say` with XTTS v2 on Metal GPU

**Implementation**:
1. Load Coqui XTTS v2 model
2. Run inference on Metal GPU (MPS)
3. Stream audio directly to playback (no disk I/O)
4. Optimize with bfloat16 + torch.compile

**Expected speedup**: 10x (581ms → 60ms)

**Status**: Pending
- Will create `tts_worker_xtts.py` after translation optimization complete

### Step 3: Parallelization (130ms → 90ms target)

**Approach**: Overlap translation and TTS processing

**Implementation**:
1. Pipeline sentences: translate sentence N while synthesizing N-1
2. Use async queues between stages
3. Prefetch audio buffers

**Expected speedup**: 30% reduction

**Status**: Pending
- Requires both optimized workers complete first

## Expected Timeline

### Today (2025-11-24)
- [x] Create optimized translation worker
- [ ] Install dependencies
- [ ] Benchmark translation optimization
- [ ] Measure actual speedup

### Tomorrow (2025-11-25)
- [ ] Implement XTTS v2 TTS worker
- [ ] Benchmark TTS optimization
- [ ] Test integrated pipeline

### Week of 2025-11-25
- [ ] Implement parallelization
- [ ] Final benchmarks
- [ ] Document results
- [ ] Update CURRENT_STATUS.md

## Target Performance

| Component | Current | Optimized | Target |
|-----------|---------|-----------|--------|
| Translation | 282ms | 70ms | < 70ms |
| TTS | 581ms | 60ms | < 60ms |
| Total | 863ms | 130ms | < 150ms |

**Success Criteria**: Total latency < 150ms (5.7x faster than baseline)

## Next Steps

1. Complete PyTorch installation
2. Run translation optimization benchmark
3. If target met (< 70ms), proceed to TTS upgrade
4. If target not met, try additional optimizations:
   - INT8 quantization
   - Smaller model (200M instead of 600M)
   - Direct Metal kernels (C++)

---

**Copyright 2025 Andrew Yates. All rights reserved.**
