# Critical Discovery: Stream Pool Never Properly Tested

**Worker**: N=34
**Date**: 2025-12-13 10:14 PST
**Updated**: 2025-12-13 10:35 PST (continued investigation)
**Severity**: CRITICAL

## Summary

The MPS stream pool code has **NEVER been properly tested**. All previous "passing" tests (N=1 through N=33) were running against the **baseline PyTorch without our modifications** due to a broken editable install.

## Root Cause

The virtual environment had TWO torch installations:
1. **Physical directory**: `venv_mps_test/lib/python3.14/site-packages/torch/` (baseline PyTorch v2.9.1)
2. **Editable install**: `__editable__.torch-2.9.1a0+gitad2a470.pth` pointing to `pytorch-mps-fork/`

Python's import system was finding the physical directory BEFORE the editable finder could redirect to our fork. Result: all test imports loaded baseline PyTorch.

## Evidence

1. Thread boundary test showed `PyTorch: 2.9.1a0+gitd38164a` (baseline) when it should show `gitad2a470` (fork)
2. After removing physical torch directory, version correctly shows `gitad2a470`
3. After fix, tests FAIL with Metal command buffer assertions
4. Baseline PyTorch has **no MPSStreamPool** - confirmed via `grep -c "MPSStreamPool"` returns 0

## Current Status

With the correct torch loaded (from our fork):
- **2/8 tests pass** (Benchmark, Thread Boundary WARN)
- **6/8 tests FAIL** with Metal assertions:
  - "Scheduled handler provided after commit call"
  - "commit command buffer with uncommitted encoder"
  - "_status < MTLCommandBufferStatusCommitted"

## Detailed Technical Analysis (N=34 Continued Session)

### What Works vs What Fails

| Scenario | Result |
|----------|--------|
| Single worker thread | PASS |
| 2 sequential workers | PASS |
| 2 parallel workers (barrier) | FAIL |
| Multiple iterations same process | FAIL |
| Fresh process each run | PASS |
| `torch.empty()` (allocation only) | PASS |
| `torch.zeros()` (blit encoder) | PASS |
| `torch.ones()` (MPSGraph) | FAIL |
| `torch.randn()` (MPSGraph) | FAIL |
| `t1 + t2` (MPSGraph arithmetic) | FAIL |

### Root Cause Hypothesis

The issue appears to be in **Apple's Metal/MPSGraph** when multiple threads simultaneously encode operations to **different** command buffers. Key observations:

1. **Blit operations work** - `torch.zeros()` uses direct Metal blit encoder
2. **MPSGraph operations fail** - `torch.ones()`, `torch.randn()`, arithmetic all use MPSGraph
3. **Sequential execution works** - same operations succeed when run sequentially
4. **Parallel execution fails** - simultaneous encoding crashes

### Investigation Findings

1. **Stream pool is thread-safe**:
   - TLS gives each thread its own stream
   - `std::call_once` handles main thread detection
   - `acquireSlot()` is mutex-protected
   - No data races in stream assignment

2. **Graph cache is thread-local**:
   - Each thread has its own `MPSGraphCache` instance
   - No shared graph state between threads

3. **Command buffers are per-stream**:
   - Each stream has its own command queue
   - Each stream creates its own command buffers

4. **The crash is in Metal framework code**:
   - Error occurs in `-[_MTLCommandBuffer addScheduledHandler:]`
   - Or `-[IOGPUMetalCommandBuffer validate]`
   - These are Apple's internal implementations

### Baseline PyTorch Comparison

The baseline PyTorch (before our changes) uses:
```cpp
MPSStream* getCurrentMPSStream() {
  return getDefaultMPSStream();  // Always returns singleton
}
```

All threads share ONE stream, so all operations serialize through that stream's serial queue. This is thread-safe but NOT parallel.

Our stream pool gives each thread its own stream, enabling true parallel execution, but this exposes Metal's threading limitations.

### Potential Fixes to Investigate

1. **Add global mutex around MPSGraph encoding**:
   - Sacrifice some parallelism to ensure safe encoding
   - Keep parallel execution at GPU level

2. **Serialize command buffer commits**:
   - Allow parallel encoding but serialize commits
   - Metal may only support serial commits from one queue

3. **Use single command queue with multiple encoders**:
   - All streams share one command queue
   - Each gets its own encoder section

4. **Check Apple's Metal threading documentation**:
   - Verify Metal's threading model for command buffers
   - May need to use dispatch barriers or explicit synchronization

## Files Modified

- `tests/test_thread_churn.py` - NEW: Thread churn test for freelist
- `tests/run_all_tests.sh` - Added Test 8 (Thread Churn)
- Removed: `venv_mps_test/lib/python3.14/site-packages/torch/` (conflicting baseline)
- Removed: `venv_mps_test/lib/python3.14/site-packages/functorch/`
- Removed: `venv_mps_test/lib/python3.14/site-packages/torchgen/`

## Test Command

```bash
source venv_mps_test/bin/activate
python -c "import torch; print(torch.__version__)"  # Must show gitad2a470
./tests/run_all_tests.sh
```

Expected: Currently 6 failures until thread-safety bugs are fixed.

## Recommendations for Next Worker (N=35)

1. **Verify fork is loaded**: `python -c "import torch; print(torch.__version__)"` must show `gitad2a470`
2. **Try adding global mutex** around MPSGraph's `encodeToCommandBuffer` call
3. **Research Apple Metal threading** - check if multiple command queues can encode simultaneously
4. **Minimal reproduction case**: 2 threads + barrier + `torch.ones(100,100, device='mps')`
5. **Consider separate command queue mutex** instead of per-stream serialization
