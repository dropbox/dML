# Manager Session 2 - December 29, 2025 (2am)

## Recent Work Completed

### Kokoro TTS Performance Optimization

**Goal**: Prove and achieve optimal performance for Kokoro TTS on MLX

**Key Results**:
| Text Length | Optimal Latency | RTF |
|-------------|-----------------|-----|
| "Hi" (2 chars) | **50ms** | 10x real-time |
| "Hello world." (12 chars) | **50ms** | 14x real-time |
| Medium (44 chars) | **91ms** | 17x real-time |

**Optimizations Applied**:

1. **Frame Bucketing** (`src/kokoro/kokoro.h`, `src/kokoro/model.cpp`)
   - Rounded output audio frames to fixed buckets (100, 200, 300, 400, 600, 800, 1200, 1600 frames)
   - Reduces MLX kernel recompilation for different output lengths
   - Audio is trimmed to actual length after synthesis

2. **Profiling Instrumentation** (`src/kokoro/model.cpp`)
   - Added `KOKORO_PROFILE=1` environment variable for detailed timing
   - Identified pipeline bottlenecks:
     - Voice embedding: <1ms
     - BERT forward: <1ms
     - Text encoder: <1ms
     - Predictor (BiLSTM): **10ms** (sequential, architectural limit)
     - Decoder: **40ms** (parallelized)

3. **Compiled Activations**
   - `snake1d()` and `leaky_relu()` use `mx::compile(shapeless=true)`

4. **Removed Unnecessary `mx::eval()`**
   - Eliminated premature GPU synchronization calls

**Files Created**:
- `src/mlx_inference_engine/profile_kokoro_detailed.cpp` - Detailed pipeline profiler
- `src/mlx_inference_engine/prove_optimal.cpp` - Optimal performance test
- `src/mlx_inference_engine/prove_warmed_optimal.cpp` - Per-phrase warmup test
- `PERFORMANCE_PROOF.md` - Documentation of findings
- Updated `README.md` with C++ Performance Proof section

**Key Finding**: MLX JIT compilation causes variance (50-800ms for same input). When properly warmed (same tensor shapes), achieves consistent **50ms for short text**.

---

## Current Worker Status

**Iteration**: 28 (running since 2025-12-29T02:34:04Z)
**PID**: 11079
**Log**: `worker_logs/worker_iter_28_claude_20251229_005922.jsonl`

### Worker Roadmap (from NOVEL_RESEARCH_ROADMAP.md)

**Completed by Workers #2076-#2085**:
- [x] Round-Trip Verification - IMPLEMENTED and VALIDATED
- [x] Prosody Beam Search - IMPLEMENTED and VALIDATED
- [x] MELD Training - Completed 5 epochs
- [x] Combined Novel Techniques Evaluation - Initial results

**Results on MELD test set (20 samples)**:
| Technique | RTF | ? F1 | ! F1 | Macro F1 |
|-----------|-----|------|------|----------|
| Baseline | 1.674 | 0.750 | 0.154 | 0.518 |
| Prosody Beam | 1.155 | **0.889** | **0.267** | 0.521 |

- Question mark F1: **+18.5%** improvement
- Exclamation F1: **+73%** improvement
- Speed: **31% faster**

**Deprecated**:
- Emotion-Aware Punctuation - Whisper native punctuation (0.53 F1) outperforms trained head (0.19 F1)

**Training Jobs Running**:
- CTC English: ~93.5% complete (step ~45235/48375), ~28h remaining

---

## Raw Metal Optimization Analysis

### Why Raw Metal?
MLX provides excellent abstraction but introduces overhead:
1. JIT compilation variance (50-800ms for same input)
2. Dynamic dispatch overhead
3. Kernel fusion limitations

### Potential Metal Targets

#### 1. Kokoro BiLSTM Predictor (HIGH IMPACT)
- **Current**: 10ms (sequential, architectural limit)
- **Opportunity**: Fused Metal kernel with shared memory for hidden states
- **Expected Gain**: 3-5x improvement → 2-3ms
- **Effort**: HIGH (LSTM is complex)

#### 2. Kokoro HiFiGAN Decoder (MEDIUM IMPACT)
- **Current**: 40ms
- **Opportunity**: Fused upsampling + ResBlock kernels
- **Expected Gain**: 2x improvement → 20ms
- **Effort**: MEDIUM

#### 3. Whisper Attention (HIGH IMPACT)
- **Current**: Variable (depends on sequence length)
- **Opportunity**: Flash Attention Metal implementation
- **Expected Gain**: 2-4x for long sequences
- **Effort**: HIGH (attention is complex)

#### 4. Snake1D Activation (LOW IMPACT)
- **Current**: <1ms (already fast)
- **Opportunity**: Fully fused with preceding operations
- **Expected Gain**: Minimal
- **Effort**: LOW

### Implementation Strategy

**Phase 1: Abstraction Layer**
```cpp
// Create backend abstraction
class ComputeBackend {
    virtual mx::array lstm_forward(...) = 0;
    virtual mx::array attention(...) = 0;
};

class MLXBackend : public ComputeBackend { /* current */ };
class MetalBackend : public ComputeBackend { /* new */ };
```

**Phase 2: Metal Kernels**
1. Start with Snake1D (simple, learn Metal API)
2. Implement fused HiFiGAN block
3. Implement optimized LSTM
4. Implement Flash Attention

**Phase 3: Benchmarking**
- A/B comparison: MLX vs Metal for each component
- Automated regression tests

---

## Remaining Work Priority List

### P0 - Critical Path

1. **Wait for CTC English Training** (~28h remaining)
   - Required for CTC speculative decoding evaluation

2. **CTC Speculative Decoding Implementation**
   - Use CTC draft tokens to accelerate Whisper decoder
   - Only remaining P2 task from original roadmap

### P1 - High Priority

3. **Full Novel Techniques Evaluation**
   - Run with `max_samples=0` for statistical significance
   - Currently only tested on 20 samples

4. **Token Bucketing for Kokoro**
   - Frame bucketing done; add token-level bucketing
   - Reduces JIT variance for predictor

5. **CosyVoice3 C++ Completion**
   - Stub exists at `src/mlx_inference_engine/cosyvoice3_model.cpp`
   - Needs full implementation

### P2 - Medium Priority

6. **Translation C++ Testing**
   - Code exists, marked "untested"
   - Need validation against Python baseline

7. **LLM C++ Verification**
   - Code exists, needs testing with various models

8. **Raw Metal Backend (Phase 1)**
   - Create abstraction layer
   - Implement Snake1D as proof of concept

### P3 - Low Priority / Future

9. **Multilingual CTC Training**
   - Currently using 6GB subset
   - Full 534GB dataset available

10. **Phoneme Head Production**
    - Multi-day effort
    - Depends on CTC completion

11. **Raw Metal Backend (Phase 2-3)**
    - Full LSTM/Attention Metal implementations

---

## Next Actions for Manager

1. **Monitor CTC Training**
   - Check `worker_status.json` periodically
   - Expected completion: ~28h from now

2. **Review Worker Logs**
   - `worker_logs/worker_iter_28_claude_20251229_005922.jsonl`

3. **Consider Machine B Allocation**
   - Could run Token Bucketing experiments
   - Could start CosyVoice3 completion

4. **Prepare CTC Speculative Decoding**
   - Review existing CUSTOM_METAL_KERNELS_ROADMAP.md
   - Plan integration with CTC checkpoint

---

## Files Reference

| File | Purpose |
|------|---------|
| `PERFORMANCE_PROOF.md` | Kokoro 50ms optimal latency proof |
| `reports/main/NOVEL_RESEARCH_ROADMAP.md` | Worker roadmap |
| `reports/main/NOVEL_TECHNIQUES_RESULTS.md` | Prosody beam search results |
| `MLX_MIGRATION_PLAN.md` | Overall migration status |
| `CUSTOM_METAL_KERNELS_ROADMAP.md` | Metal kernel implementation plan |
| `worker_status.json` | Current worker status |

---

## Session Summary

**Duration**: ~3 hours
**Primary Achievement**: Proved Kokoro TTS optimal performance at **50ms** for short text

**Key Insight**: MLX JIT compilation is the primary source of latency variance. Frame bucketing mitigates decoder variance; token bucketing would address predictor variance. Raw Metal is the path to sub-20ms synthesis but requires significant investment.

**Recommendation**: Complete CTC speculative decoding first (uses existing infrastructure), then evaluate Metal investment based on production latency requirements.
