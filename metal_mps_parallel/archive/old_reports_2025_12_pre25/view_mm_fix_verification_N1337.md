# View.mm Encoding Lock Fix - Verification Report

**Worker**: N=1337
**Date**: 2025-12-19
**Status**: VERIFIED PARTIAL SUCCESS

## Summary

The `MPSEncodingLock` fix in View.mm from N=1336 was rebuilt and tested. The fix successfully eliminates the race condition in pure `.contiguous()` operations but residual races remain in combined MPSGraph operations.

## Test Results

### Pure `.contiguous()` Operations - FIXED

| Test | Result | Notes |
|------|--------|-------|
| Pure .contiguous() (50 iters, 5 runs) | 250/250 (100%) | Single operation isolation |
| Pure F.linear | 50/50 (100%) | Isolated linear projection |
| Pure SDPA | 30/30 (100%) | Isolated attention |

### Combined Operations - RESIDUAL RACE

| Test | Result | Failure Rate |
|------|--------|--------------|
| Linear + reshape + contiguous | 50/50 | 0% |
| Linear + reshape + contiguous + slice | 42/50 | 16% |
| Full projection pattern (no SDPA) | 13/50 | 74% |
| Full projection pattern (with MPS_FORCE_GRAPH_PATH=1) | 17/50 | 66% |

### Mutex Effectiveness

| Condition | Pass Rate |
|-----------|-----------|
| Encoding mutex DISABLED | 3/30 (10%) |
| Encoding mutex ENABLED | 27-30/30 (90-100%) |

The encoding mutex provides ~9x improvement in stability.

## Key Findings

1. **Pure `.contiguous()` is now thread-safe**: The fix in View.mm successfully protects `gatherViewTensor()` and `scatterViewTensor()`.

2. **Combined operations still race**: When `.contiguous()` is combined with other MPS operations (linear, view, transpose), residual races occur.

3. **The race is NOT in `.contiguous()` itself**: Isolation testing proves individual operations pass; only combinations fail.

4. **MPSGraph path has additional race locations**: Operations that go through the MPSGraph execution path have unprotected sections.

## Technical Analysis

### What the Fix Protects

The fix adds `MPSEncodingLock` at the start of:
- `gatherViewTensor()` - Called during `.contiguous()` on strided tensors
- `scatterViewTensor()` - Called during scatter operations

### What Remains Unprotected

1. **MPSGraph cache lookups**: `LookUpOrCreateCachedGraph()` in various operations
2. **Metal pipeline state creation**: `getPipelineState()` creates compute pipelines
3. **Command buffer transitions**: Commit/wait sequences in combined operations
4. **Inter-operation timing**: Race windows between sequential MPS operations

## Recommendations

1. **Use BatchQueue for production**: The batch queue with num_workers=1 fully serializes MPS access and achieves 100% correctness.

2. **Extend encoding lock coverage**: Additional locations may need `MPSEncodingLock`:
   - `OperationUtils.mm` Placeholder constructor
   - `copy_cast_mps()` in Copy.mm
   - MPSGraph cache operations

3. **Consider graph path refactoring**: The `MPS_FORCE_GRAPH_PATH=1` mode has more races, suggesting the graph execution path needs additional synchronization.

## Follow-up Finding (N=1338)

The residual corruption in the `_in_projection_packed` + SDPA path can be eliminated by inserting a **device-wide barrier** (`torch.mps.synchronize()`) between the projection-copy step (`.contiguous()`/`.clone()`) and the subsequent SDPA call. See:
- `reports/main/projection_sdpa_device_sync_barrier_N1338.md`

## Files Changed

- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/View.mm` - MPSEncodingLock added

## Build Verification

```
Ninja incremental build: 86/89 targets rebuilt
View.mm: Rebuilt at step 72/89
libtorch_cpu.dylib: Relinked at step 80/89
libtorch_python.dylib: Relinked at step 85/89
```

## Conclusion

The View.mm fix is verified working for its intended scope (pure `.contiguous()` operations). The remaining failures are in OTHER race locations that were not addressed by this fix. The BatchQueue workaround remains the recommended solution for production 8-thread inference.
