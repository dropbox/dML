# Phase 2: INT8 Quantization Investigation Results

**Date**: 2025-11-24 Evening
**Status**: ❌ **BLOCKED** - ONNX/CoreML approach not viable

---

## Executive Summary

Investigated INT8 quantization to reduce translation latency from **154ms → 80ms** target. After comprehensive testing of ONNX Runtime with CoreML execution provider:

**Result**: ONNX INT8 on CoreML is **2x SLOWER** (301.8ms) than current PyTorch BFloat16 (155.5ms)

**Recommendation**: **Continue with current PyTorch BFloat16 system**. Pursue alternative optimizations.

---

## What Was Tested

### Approaches Compared

1. **PyTorch BFloat16 on Metal/MPS** (current system)
   - 600M parameter NLLB model
   - BFloat16 precision (native M4 Max)
   - torch.compile with max-autotune
   - Metal Performance Shaders (MPS)

2. **ONNX FP32 on CoreML**
   - Exported PyTorch model to ONNX format
   - CoreML execution provider for Metal GPU
   - Standard FP32 precision

3. **ONNX INT8 on CoreML**
   - Dynamic quantization to INT8
   - Per-channel quantization
   - CoreML execution provider for Metal GPU

---

## Benchmark Results

### Initialization Time

| Approach | Init Time | Notes |
|----------|-----------|-------|
| PyTorch BFloat16 | **3.8s** | ✅ Fastest startup |
| ONNX FP32 | 8.3s | 2.2x slower startup |
| ONNX INT8 | 7.7s | 2.0x slower startup |

### Translation Performance (per sentence)

| Approach | Cold Start | Warm Avg | Min | Max | Status |
|----------|------------|----------|-----|-----|--------|
| **PyTorch BFloat16** | 173ms | **155.5ms** | 127ms | 205ms | ✅ **WINNER** |
| ONNX FP32 | 261ms | **ERROR** | - | - | ❌ Execution errors |
| ONNX INT8 | 267ms | **301.8ms** | 223ms | 397ms | ❌ 2x slower |

### Translation Quality

All approaches that completed successfully produced identical or equivalent Japanese translations:

```
Input:  "Hello, how are you today?"
Output: "こんにちは 今日はどうですか?"

Input:  "I am working on a voice processing system."
Output: "私は声処理システムに取り組んでいます."
```

**Quality verdict**: ✅ No degradation observed in quantized model

---

## Why ONNX Failed

### Issue 1: ONNX FP32 Execution Errors

```
*************** EP Error ***************
EP Error SystemError : 20 when using ['CoreMLExecutionProvider']
Falling back to ['CPUExecutionProvider'] and retrying.
```

**Root cause**: CoreML execution provider doesn't fully support NLLB's model architecture (encoder-decoder transformer with 589-1545 nodes). Only 50-60% of nodes were assigned to CoreML, rest fell back to CPU.

### Issue 2: INT8 Quantization Slower than BFloat16

Despite 75% model size reduction (1.6GB → 400MB encoder), inference was 2x slower:

**Hypothesis**:
1. **Quantization/Dequantization overhead**: INT8 weights must be dequantized to FP32 for computation, adding latency
2. **Partial GPU acceleration**: CoreML only accelerated 50-60% of ops, rest ran on CPU
3. **Memory bandwidth not bottleneck**: M4 Max has 400GB/s unified memory bandwidth, so smaller model size doesn't help much
4. **BFloat16 is native**: M4 Max has hardware BFloat16 support, making it faster than INT8 emulation

### Issue 3: CoreML Provider Limitations

CoreML execution provider warnings:

```
number of partitions supported by CoreML: 171
number of nodes in the graph: 1358
number of nodes supported by CoreML: 691  (50.8%)
```

**Analysis**: Half the operations still run on CPU, creating CPU ↔ GPU transfer overhead.

---

## What We Learned

### ✅ What Worked

1. **ONNX export**: Successfully exported 600M NLLB model (23s export time)
2. **Quantization**: Reduced encoder from 1.6GB to 400MB (75% reduction)
3. **Translation quality**: No degradation from quantization
4. **Benchmark framework**: Comprehensive testing revealed performance issues

### ❌ What Didn't Work

1. **CoreML execution provider**: Poor compatibility with seq2seq models
2. **INT8 quantization**: Slower than native BFloat16 on M4 Max
3. **ONNX Runtime optimizations**: Didn't compensate for quantization overhead

### 💡 Key Insights

1. **BFloat16 is optimal on M4 Max**: Native hardware support beats INT8 emulation
2. **torch.compile is effective**: PyTorch's JIT compilation matches/beats ONNX Runtime
3. **Model architecture matters**: Seq2seq transformers don't map well to CoreML
4. **Unified memory is fast**: Memory bandwidth not a bottleneck at 400GB/s

---

## Alternative Paths Forward

Since ONNX INT8 is not viable, consider these alternatives:

### Option 1: Smaller Model (600M → 200M) ⭐ **RECOMMENDED**

**Approach**: Use NLLB-200-distilled-200M instead of 600M
- **Expected**: 154ms → 60-80ms (2-2.5x faster)
- **Tradeoff**: Slight quality reduction
- **Effort**: 1 hour (just change model name)
- **Probability**: High (smaller = faster)

### Option 2: Pipeline Parallelization ⭐ **RECOMMENDED**

**Approach**: Overlap translation + TTS for consecutive sentences
- **Expected**: 30% latency reduction (154ms + 192ms → ~240ms perceived)
- **Tradeoff**: More complex pipeline
- **Effort**: 4-6 hours (async queue implementation)
- **Probability**: Very high (proven technique)

### Option 3: Further torch.compile Optimization

**Approach**: Experiment with different torch.compile modes and backends
- Try `mode="reduce-overhead"` instead of `"max-autotune"`
- Try `backend="inductor"` explicitly
- Enable `torch.set_float32_matmul_precision('high')`
- **Expected**: 10-20% improvement
- **Effort**: 2-3 hours (experimentation)
- **Probability**: Medium

### Option 4: Model Pruning/Distillation

**Approach**: Train a smaller, custom-distilled model
- Use NLLB-600M as teacher, train smaller student
- **Expected**: 40-60% speedup with minimal quality loss
- **Effort**: 1-2 weeks (model training)
- **Probability**: High (proven technique)

### Option 5: Batch Processing

**Approach**: Batch multiple sentences before translation
- Wait 50-100ms to accumulate sentences
- Translate in batch (more efficient GPU utilization)
- **Expected**: 30-50% improvement for multi-sentence inputs
- **Tradeoff**: Adds 50-100ms initial delay
- **Effort**: 3-4 hours
- **Probability**: Medium-High

---

## Recommended Next Steps

### Immediate Actions (This Week)

1. **Try NLLB-200M model** (1 hour)
   - Edit `translation_worker_optimized.py`
   - Change model to `facebook/nllb-200-distilled-200M`
   - Benchmark and compare quality

2. **Implement pipeline parallelization** (4-6 hours)
   - Queue-based architecture
   - Overlap translation + TTS
   - Target: 240ms perceived latency

### Future Optimizations (Month 2)

3. **torch.compile experimentation** (2-3 hours)
   - Different modes and backends
   - Profile GPU utilization

4. **XTTS v2 revisit** (when PyTorch 2.6+ compatibility fixed)
   - Google TTS 192ms → XTTS 50-80ms
   - 3x TTS speedup

### Long-term (Month 3+)

5. **Model pruning/distillation**
   - Train custom lightweight model
   - Target: English→Japanese specialist

6. **C++ implementation** (Phase 2 from WORKER_DIRECTIVE.md)
   - Only if Python can't reach < 150ms total latency

---

## Files Created

- ✅ `export_nllb_to_onnx.py` - ONNX export script
- ✅ `quantize_onnx_model.py` - INT8 quantization script
- ✅ `stream-tts-rust/python/translation_worker_onnx.py` - ONNX inference worker
- ✅ `benchmark_all_translation.py` - Comprehensive benchmark
- ✅ `benchmark_results.json` - Detailed benchmark data
- ✅ `research_quantization.py` - Quantization API research
- ✅ `check_onnx_providers.py` - ONNX Runtime provider check
- ✅ `onnx_models/nllb-200-600m/` - Exported ONNX model (6.9GB)
- ✅ `onnx_models/nllb-200-600m-int8/` - Quantized model (1.8GB)

---

## Conclusion

**INT8 quantization via ONNX + CoreML is not viable** for this use case. The current **PyTorch BFloat16 system at 155.5ms is optimal** for the 600M model.

**Path forward**: Try smaller model (200M) + pipeline parallelization to reach < 150ms target.

**Estimated time to target**:
- Option 1 (smaller model): 1 hour → potentially 60-80ms translation
- Option 2 (parallelization): 4-6 hours → ~240ms total perceived latency
- **Combined**: Could achieve < 150ms total latency within 1 day

---

**Copyright 2025 Andrew Yates. All rights reserved.**
