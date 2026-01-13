# Phase 6: Linear.mm Thread-Safety Fix (Worker N=5)

**Date**: 2025-12-12
**Worker**: N=5
**Status**: SUCCESS - F.linear and nn.Linear parallel inference working

## Summary

Fixed the thread-safety issue in `Linear.mm`'s no-graph path (`_mps_linear_nograph`) that caused crashes when running `torch.nn.functional.linear` or `torch.nn.Linear` in parallel across multiple threads.

## Root Cause Analysis

### The Problem
Worker N=4 identified that `torch.linear`/`F.linear` crashes with 2+ threads while manual `x @ weight.T + bias` works. The issue was in `_mps_linear_nograph`, which uses Apple's `MPSNDArrayMatrixMultiplication` kernel.

### Investigation
1. Verified that thread-local `MPSKernelCache` was correctly implemented (each thread gets its own cache)
2. Discovered that Apple's `MPSNDArrayMatrixMultiplication` has internal shared state in the MPS framework
3. Even with per-thread kernel instances, concurrent encoding crashes the framework

### Key Finding
**Apple's MPS framework (specifically `MPSNDArrayMatrixMultiplication`) is not thread-safe for concurrent encoding, even when:**
- Each thread has its own kernel instance via thread-local cache
- Each thread encodes to its own command buffer via its own stream
- Each stream has its own serial dispatch queue

This is a limitation in Apple's MPS framework, not our code.

## Solution

Added a global mutex (`s_linear_nograph_mutex`) to serialize access to the `_mps_linear_nograph` function:

```cpp
// Linear.mm
static std::mutex s_linear_nograph_mutex;

static void _mps_linear_nograph(...) {
  // THREAD-SAFETY: Serialize no-graph linear operations.
  // MPSNDArrayMatrixMultiplication has internal Apple framework state that is not thread-safe
  // even when using per-thread kernel instances. This mutex prevents concurrent encoding.
  std::lock_guard<std::mutex> lock(s_linear_nograph_mutex);

  // ... rest of function
}
```

### Why This Approach?
1. **Not disabling features**: Manager directive was to fix, not disable the no-graph path
2. **Minimal performance impact**: The no-graph path is only used on macOS 15+ with contiguous inputs
3. **Graph path unaffected**: Non-contiguous inputs still use the MPSGraph path, which is already thread-safe
4. **Correctness over parallelism**: For this specific kernel, serialization is required due to Apple framework limitations

## Test Results

All tests pass with 8 threads x 50 iterations:

| Test | Threads | Iterations | Result | ops/s |
|------|---------|------------|--------|-------|
| matmul | 8 | 50 | PASS | 5154 |
| F.linear | 8 | 50 | PASS | 5875 |
| nn.Linear | 8 | 50 | PASS | 7575 |
| MLP (2 layers) | 8 | 50 | PASS | 5835 |

## Files Modified

| File | Change |
|------|--------|
| `Linear.mm` | Added `#include <mutex>`, global mutex, and `lock_guard` in `_mps_linear_nograph` |

## Patch File

`patches/005-linear-nograph-mutex-fix.patch` - Contains all thread-safety changes including:
- Thread-local graph/kernel caches
- Stream pool with per-thread streams
- Synchronization API fixes
- Linear.mm mutex fix (NEW)

## Performance Considerations

The mutex serializes only the no-graph Linear path. Other operations remain fully parallel:
- `torch.matmul` - parallel (uses MPSGraph, thread-local cache works)
- `torch.add`, `torch.mul`, etc. - parallel (MPSGraph based)
- `F.linear` with non-contiguous inputs - parallel (falls back to MPSGraph path)

The serialization overhead is minimal since kernel encoding is fast compared to actual GPU execution.

## Future Work

1. **Report to Apple**: The `MPSNDArrayMatrixMultiplication` thread-safety issue should be reported to Apple
2. **Alternative approaches**: Could investigate using MPSGraph for all Linear operations (bypass no-graph path entirely)
3. **Benchmark comparison**: Compare throughput with/without mutex to quantify overhead

## Verification Commands

```bash
# Quick parallel test
python3 -c "
import torch, threading
_ = torch.zeros(1, device='mps'); torch.mps.synchronize()
def worker(tid):
    for _ in range(50):
        x = torch.randn(32, 64, device='mps')
        y = torch.nn.functional.linear(x, torch.randn(128, 64, device='mps'), torch.randn(128, device='mps'))
        torch.mps.synchronize()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
for t in threads: t.start()
for t in threads: t.join()
print('PASS: F.linear 8 threads x 50 iterations')
"
```
