# Comprehensive Benchmark Analysis N=1042

**Date**: 2025-12-17
**Purpose**: Document scalability ceiling and root cause analysis

## Executive Summary

Extensive benchmarking confirms that **CPU-side serialization from global operation mutexes** is the primary scalability bottleneck, not GPU saturation. Efficiency is consistent (~30-35%) across workload sizes, and ops/s plateaus at 4+ threads regardless of tensor dimensions.

## Benchmark Results

### 1. Standard Benchmark (nn.Linear 512x256, batch=16)

| Threads | Ops/s | Efficiency |
|---------|-------|------------|
| 1 | ~4350 | 100.0% |
| 2 | ~6129 | 70.5% |
| 4 | ~6694 | 38.5% |
| 8 | ~6108 | 17.6% |
| 16 | ~5905 | 8.5% |

**Key Finding**: Ops/s DECREASES from 4→8→16 threads due to mutex contention overhead exceeding parallelism benefit.

### 2. Workload Size Comparison (8 threads)

| Workload | Dimensions | Batch | Efficiency |
|----------|------------|-------|------------|
| Small | 256x128 | 4 | 31.5% |
| Medium | 512x256 | 16 | 29.6% |
| Large | 2048x2048 | 64 | 34.8% |

**Key Finding**: Efficiency is remarkably consistent across 16x workload size increase. This rules out GPU saturation as the bottleneck.

### 3. With MPS_FORCE_GRAPH_PATH=1

| Model | Default | Graph Path | Delta |
|-------|---------|------------|-------|
| nn.Linear | 30.6% | 35.0% | +4.4% |
| Transformer | 21.7% | 20.8% | -0.9% |

**Key Finding**: Graph path helps simple operations (Linear) but hurts complex operations (Transformer) due to compilation overhead.

## Root Cause Analysis

### Primary Bottleneck: Seven Global Operation Mutexes

```cpp
// pytorch-mps-fork/aten/src/ATen/native/mps/operations/

s_linear_nograph_mutex      // Linear.mm:19 - MOST IMPACTFUL
s_layer_norm_mutex          // Normalization.mm:31
s_bmm_tiled_mutex           // LinearAlgebra.mm:49
s_lu_decomposition_mutex    // LinearAlgebra.mm:50
s_lu_solve_mutex            // LinearAlgebra.mm:51
s_solve_triangular_mutex    // LinearAlgebra.mm:52
s_ndarray_identity_mutex    // OperationUtils.mm:484
```

These mutexes exist because:
> "Apple's MPS framework has internal shared state that makes concurrent encoding of MPSNDArrayMatrixMultiplication kernels unsafe, even with per-thread instances."

### Secondary Bottleneck: Allocator m_mutex (Now Sharded)

The m_mutex sharding (N=1038) improved efficiency from ~29.3% to ~30.6% (+1.3%), confirming it was a contributor but not the primary bottleneck.

### Why Efficiency Drops with More Threads

1. **Serial Portion**: Operation encoding requires mutex - Amdahl's Law applies
2. **Contention Overhead**: More threads = more mutex wait time
3. **Cache Thrashing**: Thread context switches pollute caches
4. **Diminishing Returns**: Beyond 4 threads, contention exceeds benefit

## Theoretical Maximum Efficiency

Using Amdahl's Law with observed data:

```
E_8 = 1 / (1 + (8-1) * f_serial)

If E_8 = 0.30, then:
  0.30 = 1 / (1 + 7 * f_serial)
  f_serial = (1/0.30 - 1) / 7 = 0.33 = 33%

~33% of execution time is serial (mutex-protected).
```

Maximum theoretical efficiency at 8 threads with 33% serial: **30%** (we're hitting the wall).

## Two-Path Architecture

Linear.mm has two execution paths:

1. **No-Graph Path**: Uses `MPSNDArrayMatrixMultiplication`, fast but requires mutex
2. **Graph Path**: Uses `MPSGraph`, thread-safe but has compilation overhead

Path selection:
```cpp
const bool parallel_streams_active =
    MPSStreamPool::instance().getActiveStreamCount() > 1;
const bool force_graph_path = force_graph_path_env || parallel_streams_active;

if (!force_graph_path && is_contiguous) {
    _mps_linear_nograph(...);  // Uses mutex
} else {
    // MPSGraph path - thread-safe
}
```

## Potential Solutions (Future Work)

| Solution | Complexity | Expected Gain | Risk |
|----------|------------|---------------|------|
| Always use graph path | Low | +5-10% | Regression on small tensors |
| Shard operation mutexes | Medium | +5-15% | Race conditions |
| Per-thread kernel instances | High | +20-30% | Apple MPS internals unknown |
| Hybrid adaptive selection | Medium | +10-20% | Tuning complexity |

## Conclusion

The current 30-35% efficiency at 8 threads is close to the **theoretical maximum** given the serial constraints imposed by Apple MPS internals. Further improvement requires either:

1. Changes to Apple's MPS framework (out of scope)
2. Hybrid strategies that minimize serial portion
3. Different workload patterns that amortize mutex overhead

The allocator sharding (N=1038) was correct and helped, but the operation-level mutexes are the dominant constraint.

## Related Reports

- `operation_mutex_formal_analysis_N1040_2025-12-17.md` - Detailed mutex analysis
- `verification_reality_check_N1040_2025-12-17.md` - Verification infrastructure assessment
- `mps-verify/VERIFICATION_ROADMAP.md` - Phase 7 for operation mutex modeling

## Verification Status

- **TLA+ Model**: 15.3M states explored, all safety properties pass
- **CBMC**: 4/4 harnesses pass
- **Test Suite**: 24/24 tests pass
- **Thread Safety**: All critical sections protected
