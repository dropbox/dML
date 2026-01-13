# Phase 6 Continued: Thread-Safety Investigation (Worker N=4)

**Date**: 2025-12-12
**Worker**: N=4
**Status**: PARTIAL - Graph cache fix verified working, stream pool issues identified

## Summary

Implemented thread-local graph and kernel caches for MPSGraph operations. Verified that the graph path works for parallel softmax. However, discovered additional command buffer race conditions in the stream pool implementation that cause failures under stress testing.

## Changes Made

### 1. Thread-Local Graph Cache (OperationUtils.h/mm)
- Changed `MPSGraphCache::_instance_cache` from `static` to `static thread_local`
- Changed `MPSKernelCache::_instance_cache` from `static` to `static thread_local`
- Each thread now gets its own graph cache, preventing concurrent encoding of shared MPSGraph objects

### 2. Stream Pool Race Condition Fix (MPSStream.h/mm)
- Added `stream_creation_mutex_` to `MPSStreamPool` class
- Added double-checked locking to `ensureInitialized()` and `createStream()`
- Prevents race conditions when multiple threads try to create the same stream

### 3. Synchronization API Fixes (MPSHooks.mm, MPSGuardImpl.mm)
- Changed `deviceSynchronize()` from `getDefaultMPSStream()` to `getCurrentMPSStream()`
- Changed `commitStream()` from `getDefaultMPSStream()` to `getCurrentMPSStream()`
- Changed `synchronizeDevice()` from `getDefaultMPSStream()` to `getCurrentMPSStream()`
- Each thread now synchronizes its own stream when calling `torch.mps.synchronize()`

## Test Results

### Working
- **Softmax (non-contiguous)**: 4 threads x 20 iterations - PASS
  - Uses graph path due to non-contiguous weight tensor
  - Verifies thread-local graph cache is working
- **Basic tensor ops**: matmul, relu, etc. - PASS
  - These use stateless Metal kernels, not MPSGraph

### Failing
- **Softmax (stress test)**: 8 threads x 50 iterations - FAIL
  - Error: `_status < MTLCommandBufferStatusCommitted`
  - Indicates command buffer race condition despite per-thread streams
- **nn.Linear (contiguous)**: 4 threads x 10 iterations - FAIL
  - Uses `_mps_linear_nograph` path with MPSNDArrayMatrixMultiplication
  - Error: `Invalid KernelDAG, Number of core ops should be 1`
  - This is an Apple MPS framework limitation for concurrent kernel encoding

## Root Cause Analysis

### Issue 1: Graph Cache (FIXED)
- `MPSGraphCache` was a singleton returning shared `MPSGraph*` objects
- Multiple threads encoding to the same graph caused crashes
- **Fix**: Thread-local cache gives each thread its own graphs

### Issue 2: Synchronization APIs (FIXED)
- `torch.mps.synchronize()` was synchronizing the default stream (stream 0)
- This caused race conditions when threads had their own streams
- **Fix**: Use `getCurrentMPSStream()` instead of `getDefaultMPSStream()`

### Issue 3: Command Buffer Races (UNRESOLVED)
- Even with per-thread streams, stress testing causes command buffer errors
- The command buffer / encoder lifecycle may not be fully thread-safe
- Possible causes:
  1. Shared state in MPSDevice singleton
  2. Metal driver limitations with concurrent command queues
  3. Race in `commandBuffer()` or `commandEncoder()` methods

### Issue 4: MPSNDArrayMatrixMultiplication (UNRESOLVED)
- The no-graph Linear path uses `MPSNDArrayMatrixMultiplication`
- This kernel has internal thread-safety issues causing "Invalid KernelDAG"
- Possibly uses shared kernel DAG state internally

## Key Files Modified

| File | Change |
|------|--------|
| `OperationUtils.h` | Thread-local graph/kernel cache declarations |
| `OperationUtils.mm` | Thread-local cache initialization |
| `MPSStream.h` | Stream creation mutex, includes |
| `MPSStream.mm` | Double-checked locking for stream creation |
| `MPSHooks.mm` | Use getCurrentMPSStream() in sync methods |
| `MPSGuardImpl.mm` | Use getCurrentMPSStream() in synchronizeDevice |

## Next Steps for Worker N=5

### Priority 1: Debug Command Buffer Race
The stress test fails with `_status < MTLCommandBufferStatusCommitted`. Investigate:
1. Add logging to `commandBuffer()` and `commandEncoder()` to trace which thread/stream is involved
2. Check if `dispatch_sync(_serialQueue, ...)` is correctly isolating operations
3. Consider if Metal itself has limitations with concurrent command queues

### Priority 2: Linear No-Graph Path
The `_mps_linear_nograph` path fails with parallel execution. Options:
1. Keep the graph path for now (works in parallel with our fix)
2. Investigate if `MPSNDArrayMatrixMultiplication` can be made thread-safe
3. Create per-thread kernel instances instead of caching

### Test Command
```bash
source venv_mps_test/bin/activate && python3 -c "
import torch
import threading

errors = []
def worker(tid):
    try:
        for _ in range(50):
            x = torch.randn(4, 64, device='mps')
            y = torch.softmax(x, dim=-1)
            torch.mps.synchronize()
    except Exception as e:
        errors.append((tid, str(e)))

threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
for t in threads: t.start()
for t in threads: t.join()

print('PASS' if not errors else f'FAIL: {len(errors)} errors')
"
```

## Lessons Learned

1. Thread-local graph caches work for MPSGraph operations
2. Synchronization APIs must use per-thread streams, not the default stream
3. Even with per-thread streams, there may be deeper threading issues in the MPS stack
4. Apple's MPS kernel objects (like MPSNDArrayMatrixMultiplication) have internal thread-safety issues
5. The no-graph path for Linear is not thread-safe due to Apple framework limitations

## Files to Read

- `aten/src/ATen/mps/MPSStream.mm` - Stream pool implementation
- `aten/src/ATen/native/mps/operations/Linear.mm` - Two code paths (graph vs no-graph)
- `aten/src/ATen/native/mps/OperationUtils.h` - Graph cache definition
