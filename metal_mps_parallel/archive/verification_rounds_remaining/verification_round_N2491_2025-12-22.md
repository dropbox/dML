# Verification Round N=2491 - BUG FOUND

**Date**: 2025-12-22
**Iteration**: N=2491
**Result**: **BUG FOUND** - Race condition in addCompletedHandler()

## Bug Report

### Error Message
```
-[_MTLCommandBuffer addCompletedHandler:]:976: failed assertion
`Completed handler provided after commit call'
```

### Location
`MPSStream.mm:addCompletedHandler()` lines 401-417

### Root Cause Analysis

**Flawed Code Flow:**
```cpp
MPSCommandBuffer* cb = _commandBuffer;  // [1] Local copy

if (cb) {
  MTLCommandBufferStatus status = [cb status];
  if (status != MTLCommandBufferStatusNotEnqueued) {
    cb = nil;  // [2] Set LOCAL variable to nil, _commandBuffer unchanged
  }
}

if (!cb) {
  cb = commandBuffer();  // [3] Returns _commandBuffer which is NOT nil!
}

[cb addCompletedHandler:block];  // [4] CRASH - buffer is committed
```

**The Bug:**
1. `_commandBuffer` is committed (status != NotEnqueued)
2. Local `cb` is set to nil, but `_commandBuffer` member variable is NOT changed
3. `commandBuffer()` returns `_commandBuffer` (which is still non-nil and committed)
4. `addCompletedHandler` called on committed buffer → Metal assertion failure

### Why It Wasn't Caught Before
- Rare timing condition - buffer must be committed between creation and handler addition
- More likely under heavy parallel load (4+ threads)
- Comprehensive test triggered it due to many concurrent operations

### Suggested Fix

**Option A**: Nil out `_commandBuffer` before calling `commandBuffer()`:
```cpp
if (cb) {
  MTLCommandBufferStatus status = [cb status];
  if (status != MTLCommandBufferStatusNotEnqueued) {
    [_commandBuffer release];  // Release committed buffer
    _commandBuffer = nil;      // Clear so commandBuffer() creates new one
    cb = nil;
  }
}
```

**Option B**: Modify `commandBuffer()` to check status:
```cpp
MPSCommandBuffer* MPSStream::commandBuffer() {
  std::lock_guard<std::recursive_mutex> lock(_streamMutex);
  if (_commandBuffer) {
    MTLCommandBufferStatus status = [_commandBuffer status];
    if (status != MTLCommandBufferStatusNotEnqueued) {
      [_commandBuffer release];
      _commandBuffer = nil;
    }
  }
  if (!_commandBuffer) {
    _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }
  return _commandBuffer;
}
```

### Reproduction
Run comprehensive stress test with 4+ threads and many concurrent operations.

## Conclusion

**BUG CONFIRMED** after 22 verification rounds and 66 rigorous attempts. This is a genuine race condition that causes Metal assertion failure under parallel load.

Priority: **HIGH** - causes crash under parallel inference
