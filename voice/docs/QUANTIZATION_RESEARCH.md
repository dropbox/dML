# INT8 Quantization Research for NLLB on Metal

**Date**: 2025-11-24
**Goal**: Reduce translation latency from 154ms to 80-100ms using INT8 quantization

## Research Findings

### PyTorch Native Quantization on MPS

**Status**: ❌ **NOT SUPPORTED**

**Test Results** (from `test_quantization_mps.py`):
- Quantized tensors cannot be moved to MPS device
- Dynamic quantization fails with "NoQEngine" error on MPS
- Static quantization only works on CPU backend

**Why**: MPS backend does not implement quantized operations (qint8, quint8)

### TorchAO (New PyTorch Quantization Library)

**Status**: ⚠️ **INCOMPATIBLE**

**Issues**:
- torchao 0.14.1 is incompatible with PyTorch 2.9.1
- Error: "Skipping import of cpp extensions due to incompatible torch version"
- See: https://github.com/pytorch/ao/issues/2919

**Workaround**: Would need to downgrade PyTorch, but this risks breaking Metal GPU support

## Alternative Approaches

### Option 1: Float16 instead of BFloat16 ⭐ EASIEST

**Rationale**:
- BFloat16: 16-bit (1 sign, 8 exponent, 7 mantissa) - better range
- Float16: 16-bit (1 sign, 5 exponent, 10 mantissa) - better precision
- Both are 16-bit, so no memory savings
- Float16 *might* be faster on some hardware

**Pros**:
- Simple change: `torch_dtype=torch.float16`
- Full MPS support
- No quantization complexity

**Cons**:
- Unlikely to give significant speedup (already using bfloat16)
- No memory savings vs current setup

**Expected Impact**: 0-5% improvement (minimal)

### Option 2: Smaller NLLB Model ⭐⭐ RECOMMENDED

**Current**: NLLB-200-distilled-600M (600M parameters)
**Alternative**: NLLB-200-distilled-200M (200M parameters)

**Rationale**:
- 3x fewer parameters = 3x less computation
- Still maintains excellent quality for EN→JA
- Full Metal GPU support

**Pros**:
- Significant speedup expected (2-3x)
- Less memory usage
- Same API, easy to test

**Cons**:
- Slightly lower translation quality
- Need to benchmark quality

**Expected Impact**: 154ms → 50-80ms (2-3x improvement)

### Option 3: Custom Metal Kernels for INT8

**Approach**: Write custom Metal shaders for INT8 GEMM (matrix multiplication)

**Pros**:
- True INT8 performance (4x less memory, 2-4x faster)
- Maximum control

**Cons**:
- Weeks of work
- Complex Metal programming
- Maintenance burden
- Need to handle quantization/dequantization

**Expected Impact**: 154ms → 40-60ms (3-4x improvement)
**Timeline**: 2-3 weeks of development

### Option 4: Core ML Conversion

**Approach**: Convert NLLB model to Core ML format with INT8 quantization

**Steps**:
1. Export NLLB to ONNX
2. Convert ONNX to Core ML (with coremltools)
3. Apply INT8 quantization during conversion
4. Use Core ML inference

**Pros**:
- Native Apple acceleration
- True INT8 quantization
- No PyTorch compatibility issues

**Cons**:
- Complex conversion process
- Different API (need wrapper)
- May lose some quality
- Limited control over generation

**Expected Impact**: 154ms → 50-80ms (2-3x improvement)
**Timeline**: 1-2 days of work

### Option 5: CPU INT8 Quantization

**Approach**: Run quantized model on CPU instead of GPU

**Pros**:
- PyTorch quantization fully works on CPU
- True INT8 performance

**Cons**:
- CPU is MUCH slower than Metal GPU
- Would likely INCREASE latency, not decrease
- Defeats purpose of GPU acceleration

**Expected Impact**: 154ms → 300-500ms (2-3x WORSE)
**Not recommended**

### Option 6: ONNX Runtime with CoreML EP

**Approach**:
1. Export NLLB to ONNX
2. Use ONNX Runtime with CoreML Execution Provider
3. Apply quantization at ONNX level

**Pros**:
- ONNX Runtime has good quantization support
- CoreML EP uses Metal GPU
- More mature than direct Core ML

**Cons**:
- Different runtime (ONNX Runtime, not PyTorch)
- Need to handle tokenization separately
- Conversion complexity

**Expected Impact**: 154ms → 60-90ms (1.7-2.5x improvement)
**Timeline**: 1-2 days of work

## Recommended Path Forward

### Immediate (Today): Test Smaller Model

**Action**: Test NLLB-200-distilled-200M (200M params)

**Implementation**:
```python
# Change one line in translation_worker_optimized.py
model_name = "facebook/nllb-200-distilled-200M"  # was 600M
```

**Expected Result**:
- 154ms → 50-80ms (2-3x faster)
- Translation quality: Good (BLEU ~22-24 vs ~25-27)

**Time Investment**: 30 minutes (download + benchmark)

**Risk**: Low (easy to revert)

### Short-term (This Week): Core ML Conversion

If smaller model doesn't hit target, proceed with Core ML:

1. Export NLLB to ONNX (1 hour)
2. Convert to Core ML with quantization (2 hours)
3. Write Core ML wrapper for inference (3 hours)
4. Benchmark and validate quality (2 hours)

**Total**: 1 day of work
**Expected**: 154ms → 50-80ms with INT8

### Long-term (Next Month): Custom Metal Kernels

Only if sub-50ms latency is required:
- Write custom INT8 GEMM kernels in Metal
- Optimize for M4 Max architecture
- Profile and iterate

**Total**: 2-3 weeks of work
**Expected**: 154ms → 30-50ms

## Recommendation

**Start with Option 2 (smaller model)** because:
1. Easiest to test (change one line)
2. High probability of success
3. No new dependencies
4. Reversible if quality suffers

If quality is acceptable, we get:
- **Current**: 346ms total (154ms translation + 192ms TTS)
- **With 200M model**: 230-270ms total (50-80ms translation + 192ms TTS)
- **Improvement**: 25-35% faster, close to target

## Decision

**Next step**: Test NLLB-200-distilled-200M model

**Success criteria**:
- Translation latency < 80ms
- Translation quality acceptable (BLEU > 22)
- Total pipeline latency < 280ms

If successful, this becomes Phase 1.6 completion.

---

**Copyright 2025 Andrew Yates. All rights reserved.**
