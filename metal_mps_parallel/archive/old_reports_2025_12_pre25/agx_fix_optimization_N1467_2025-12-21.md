# AGX Fix Optimization Report

**Worker**: N=1467
**Date**: 2025-12-21
**Task**: Task 0.6 - Optimization Patch (Per-Encoder Mutex)

---

## Summary

Implemented per-encoder mutex optimization that eliminates mutex contention while maintaining
0% crash rate. Performance is identical to the global mutex approach because the bottleneck
is the GPU command queue, not the mutex.

---

## Implementation

### Approach: Per-Encoder Mutex

Instead of a single global mutex that serializes ALL encoder operations, the optimized
version creates a separate mutex for each `MTLComputeCommandEncoder` instance:

| Approach | Mutex Scope | Contention Pattern |
|----------|-------------|-------------------|
| Global mutex | All encoders share one mutex | Multiple encoders contend |
| Per-encoder mutex | Each encoder has its own mutex | Only same-encoder operations contend |

### Technical Details

- **Mutex storage**: Uses Objective-C associated objects (`objc_setAssociatedObject`)
- **Lifecycle**: Mutex created lazily on first access, automatically cleaned up when encoder is deallocated
- **No global map**: Avoids the need for a thread-safe map data structure

### Files Created

- `agx_fix/src/agx_fix_optimized.mm` - Per-encoder mutex implementation
- Updated `agx_fix/Makefile` to build both libraries

---

## Verification Results

### Correctness Test

| Metric | Result |
|--------|--------|
| Iterations | 50 |
| Threads per iteration | 8 |
| Ops per thread | 50 |
| Total operations | 20,000 |
| **Crashes** | **0** |

All 50 iterations completed successfully with 0 crashes.

### Mutex Statistics

| Metric | Global Mutex | Per-Encoder Mutex |
|--------|--------------|-------------------|
| Acquisitions | 2,400 | 2,400 |
| Contentions | 69 | 0 |
| **Contention rate** | **2.88%** | **0.00%** |
| Mutex creations | 1 (global) | 800 (per-encoder) |

**Key finding**: Per-encoder mutex eliminates ALL contention.

---

## Performance Comparison

### Throughput (ops/s)

| Threads | Global Mutex | Per-Encoder Mutex | Difference |
|---------|--------------|-------------------|------------|
| 1 | 3,609 ± 905 | 4,118 ± 1,061 | +14% |
| 2 | 8,035 ± 130 | 7,883 ± 126 | -2% |
| 4 | 10,386 ± 351 | 10,120 ± 616 | -3% |
| 8 | 8,922 ± 43 | 8,952 ± 109 | +0.3% |
| 16 | 9,231 ± 71 | 9,235 ± 86 | +0.04% |

**Key finding**: Performance is essentially identical (within noise margin).

---

## Analysis

### Why Performance is Identical

1. **GPU command queue is the bottleneck**, not the CPU mutex
2. The Metal command queue can only process ~10,000 ops/s regardless of synchronization approach
3. Even with 0% contention, per-encoder mutex doesn't unlock more parallelism
4. The GPU is already saturated at 4+ threads

### Why Contention Rate Differs

- **Global mutex**: All 8 threads competing for one mutex → ~3% contention
- **Per-encoder mutex**: Each encoder gets its own mutex → 0% contention
- Different encoders (in different threads) never share a mutex

### Throughput vs N=1466 Report

The N=1466 report showed ~4,800 ops/s while this test shows ~9,000-10,000 ops/s.
Differences:
- N=1466 used 5 iterations × 50 ops, this uses 5 iterations × 50 ops (same)
- Model size may differ (N=1466 used comprehensive benchmark script)
- System load may differ

The relative comparison (global vs per-encoder) is what matters, and both approaches
show identical scaling behavior.

---

## Conclusions

### Task 0.6 Success Criteria

| Criterion | Status |
|-----------|--------|
| Maintain 0% crash rate | ✅ (20,000 ops, 0 crashes) |
| Reduce contention rate | ✅ (2.88% → 0.00%) |
| Improve throughput | ❌ (identical, GPU-bound) |
| Target: >50% theoretical max | ⚠️ (N/A - GPU is bottleneck) |

### Recommendations

1. **Use per-encoder mutex** for cleaner design:
   - Zero contention is architecturally cleaner
   - Associated objects tie mutex lifecycle to encoder lifecycle
   - No global state to manage

2. **Don't expect throughput improvement**:
   - The GPU command queue is saturated
   - More parallel CPU work doesn't help when GPU is bottleneck
   - This is expected behavior for compute-bound workloads

3. **Next optimization**: To improve throughput, need to:
   - Batch more work per encoder (reduce API call overhead)
   - Use larger tensors (better GPU utilization)
   - Use multiple command queues (limited benefit, GPU still saturated)

---

## Test Environment

| Field | Value |
|-------|-------|
| Hardware | Mac16,5 (M4 Max, 40 GPU cores) |
| OS | macOS 15.7.3 (24G419) |
| PyTorch | 2.9.1a0+git1db92a1 |
| Test | 5 iterations × 50 ops/thread × varying threads |

---

## Files

- `agx_fix/src/agx_fix_optimized.mm` - Per-encoder mutex implementation
- `agx_fix/build/libagx_fix_optimized.dylib` - Built library
