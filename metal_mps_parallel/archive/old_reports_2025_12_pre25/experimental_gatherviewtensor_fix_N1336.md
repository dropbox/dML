# Experimental Fix: MPSEncodingLock Extension to View.mm

**Worker**: N=1336
**Date**: 2025-12-19
**Status**: EXPERIMENTAL (requires rebuild to test)

## Summary

Extended `MPSEncodingLock` coverage to `gatherViewTensor()` and `scatterViewTensor()` functions in `View.mm` to address the residual 4% race condition in parallel `.contiguous()` operations.

## Background

Worker N=1335 identified the root cause of the `.contiguous()` race condition:
- The race occurs in `Placeholder::Placeholder()` which calls `gatherViewTensor()`
- `gatherViewTensor()` triggers Metal encoding operations NOT protected by `MPSEncodingLock`
- The existing lock only protects `executeMPSGraphOnSerialQueue()`, not pre-execution tensor binding

## Changes Made

**File**: `pytorch-mps-fork/aten/src/ATen/native/mps/operations/View.mm`

1. Added include: `#include <ATen/mps/MPSStream.h>`
2. Added `MPSEncodingLock` at start of `gatherViewTensor()` (line 85-88)
3. Added `MPSEncodingLock` at start of `scatterViewTensor()` (line 140-143)

## Verification Before Fix

Aggressive parallel test (8 threads, 50 iterations, larger tensors):
- Result: 48/50 passed (96%)
- Failure rate: **4%**

This matches the "residual 4.4% failure" predicted by N=1335's analysis.

## Expected Impact

After rebuild, the fix should:
- Eliminate the 4% residual race condition
- Achieve 100% parallel correctness without batching

## Trade-offs

**Performance Impact**:
- Adds lock contention to gather/scatter operations
- The lock is recursive (safe for nested calls)
- May decrease throughput for strided tensor operations
- **Benchmarking required after rebuild**

**Lock Scope**:
- Option A (chosen): Lock at function entry - protects entire function including `at::empty()`
- Alternative: Lock only around `dispatch_sync_with_rethrow` - smaller scope but may miss allocation races

## Testing Requirements

To verify this fix:
1. Rebuild pytorch-mps-fork: `python setup.py develop`
2. Run aggressive parallel test: `python3 tests/test_parallel_contiguous.py`
3. Expected: 50/50 pass (100%)
4. Run throughput benchmark to measure performance impact

## Files Modified

- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/View.mm`

## Patches Created

- `patches/039-gatherviewtensor-encoding-lock.patch` - Standalone patch
- `patches/cumulative-v2.9.1-to-mps-stream-pool.patch` - Updated cumulative patch

## Next Steps for Future Workers

1. Rebuild PyTorch fork: `cd pytorch-mps-fork && python setup.py develop`
2. Run the aggressive parallel test to verify 100% correctness
3. Benchmark throughput to quantify performance impact
4. If successful, consider submitting to upstream PyTorch

## References

- `reports/main/race_root_cause_analysis_N1335.md` - Root cause analysis
- `tests/minimal_mps_contiguous_race.mm` - C++ reproduction test
- `tests/test_batch_inference.py` - Batched inference tests
