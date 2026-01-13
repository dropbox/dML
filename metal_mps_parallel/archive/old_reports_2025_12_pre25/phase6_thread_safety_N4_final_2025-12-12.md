# Phase 6 Continued: Thread-Safety Investigation (Worker N=4 Final)

**Date**: 2025-12-12
**Worker**: N=4
**Status**: PARTIAL - Major progress, F.linear issue remains

## Summary

Made significant progress on MPS parallel thread-safety:
1. Thread-local graph/kernel caches implemented and verified working
2. Stream pool creation race condition fixed
3. Synchronization APIs fixed to use per-thread streams
4. Linear no-graph path disabled (uses thread-unsafe Apple MPS kernels)
5. Main thread initialization requirement discovered

## Changes Made

### 1. Thread-Local Graph Cache (OperationUtils.h/mm)
- Changed `MPSGraphCache::_instance_cache` from `static` to `static thread_local`
- Changed `MPSKernelCache::_instance_cache` from `static` to `static thread_local`
- Each thread now gets its own graph cache

### 2. Stream Pool Race Condition Fix (MPSStream.h/mm)
- Added `stream_creation_mutex_` to `MPSStreamPool` class
- Added double-checked locking to `ensureInitialized()` and `createStream()`

### 3. Synchronization API Fixes (MPSHooks.mm, MPSGuardImpl.mm)
- Changed sync methods from `getDefaultMPSStream()` to `getCurrentMPSStream()`

### 4. Linear.mm No-Graph Path Disabled
- Added `false &&` to disable MPSNDArrayMatrixMultiplication path
- This path has internal Apple MPS framework thread-safety issues

## Test Results

### Working (with main thread init)
- **matmul**: 4 threads x 50 iterations - PASS
- **Manual matmul + bias**: 4 threads x 20 iterations - PASS
- **nn.Linear (contiguous)**: 2 threads x 10 iterations - PASS

### Failing
- **F.linear**: 2+ threads - FAIL (even with unique tensor shapes)
- **nn.Linear**: 3+ threads - FAIL intermittently

## Critical Finding

**Main thread must initialize MPS before spawning workers:**
```python
_ = torch.zeros(1, device='mps')
torch.mps.synchronize()
# Now safe to spawn threads
```

## Unresolved Issues

### Issue 1: No-Graph Path Crashes
`_mps_linear_nograph` crashes with 2+ threads even though:
- Thread-local `MPSKernelCache` should give each thread its own `MPSNDArrayMatrixMultiplication` instance
- Each thread has its own stream with its own serial queue
- The same pattern (dispatch_sync_with_rethrow) works for graph operations

This suggests `MPSNDArrayMatrixMultiplication` has internal shared state in Apple's MPS framework that we cannot control from PyTorch.

### Issue 2: F.linear/torch.linear Crashes (CRITICAL FINDING)

**torch.linear crashes but manual equivalents work:**
- `x @ weight.T + bias` - WORKS in parallel (4t x 20i PASS)
- `torch.addmm(bias, x, weight.T)` - WORKS in parallel (4t x 20i PASS)
- `torch.linear(x, weight, bias)` - CRASHES with 2+ threads
- `F.linear(x, weight, bias)` - CRASHES with 2+ threads

This proves the issue is in `Linear.mm`'s implementation, not in:
- Thread-local graph cache
- Per-thread streams
- Underlying matmul or add operations
- MPSGraph encoding

The Linear.mm graph path must have some shared state that matmul doesn't have.

## Files Modified

| File | Change |
|------|--------|
| `OperationUtils.h` | Thread-local cache declarations |
| `OperationUtils.mm` | Thread-local cache initialization |
| `MPSStream.h` | Stream creation mutex |
| `MPSStream.mm` | Double-checked locking |
| `MPSHooks.mm` | getCurrentMPSStream() in sync |
| `MPSGuardImpl.mm` | getCurrentMPSStream() in sync |
| `Linear.mm` | Disabled no-graph path |

## Next Steps for Worker N=5

### CRITICAL: Focus on Linear.mm Graph Path

Since `x @ weight.T + bias` works but `torch.linear(x, weight, bias)` crashes, the issue is in Linear.mm specifically.

**Compare these code paths:**
- `LinearAlgebra.mm:mm_out_mps_impl()` - matmul (WORKS)
- `Linear.mm:_mps_linear()` graph path (CRASHES)

**Key differences to investigate:**
1. Linear.mm creates `weightTransposeTensor` in the graph - check if this causes issues
2. Linear.mm handles bias addition in the graph
3. Linear.mm uses manual NSMutableDictionary vs `dictionaryFromPlaceholders`
4. Linear.mm creates the output tensor before graph execution

**Recommended fix approach:**
Try rewriting the Linear.mm graph path to match the pattern that works:
```python
# Instead of custom Linear graph, use these ops which work:
y = torch.mm(x, weight.T)  # or torch.matmul
if bias is not None:
    y = y + bias
```

### Other items
1. **Fix MPSProfiler singleton** - Not thread-safe (line 825 in MPSProfiler.mm)
2. **Main thread init** - Document this requirement in PyTorch docs

## Working Test
```bash
source venv_mps_test/bin/activate && python3 -c "
import torch, threading
_ = torch.zeros(1, device='mps'); torch.mps.synchronize()
def worker(tid):
    for _ in range(50):
        x = torch.randn(32, 32, device='mps')
        y = torch.matmul(x, x)
        torch.mps.synchronize()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print('PASS')
"
```
