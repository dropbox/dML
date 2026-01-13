# Phase 5: Build and Test Report

**Date**: 2025-12-12
**Worker**: N=2
**Status**: PASS - Parallel MPS inference working

## Summary

Successfully built PyTorch 2.9.1 with the MPS stream pool modifications and validated parallel inference across 8 concurrent threads.

## Build Results

- **PyTorch Version**: 2.9.1a0+gitd38164a
- **Build Target**: libtorch_cpu.dylib (191MB)
- **Build Time**: ~10 minutes initial, <1 minute incremental
- **Python Package**: torch-2.9.1a0+gitd38164a-cp314-cp314-macosx_15_0_arm64.whl

## Test Results

### Sequential Test
- Single thread, 3 iterations: PASS
- Basic MPS tensor operations work correctly

### Parallel Test (4 threads x 10 iterations)
- Operations: 40
- Errors: 0
- Throughput: 5,950 ops/sec
- **Result: PASS**

### Stress Test (8 threads x 50 iterations)
- Operations: 400
- Errors: 0
- Throughput: 17,795 ops/sec
- **Result: PASS**

## Bug Fixes Applied

### 1. MPSAllocator.mm - Thread-Safe Completion Handlers
**Problem**: `m_stream->addCompletedHandler()` used a cached default stream, causing race conditions when multiple threads used different streams.

**Fix**: Changed to `getCurrentMPSStream()->addCompletedHandler()` to use the calling thread's stream.

**Lines**: 369-371, 663-666

### 2. MPSProfiler.mm - Thread-Safe Profiler Handlers
**Problem**: `getDefaultMPSStream()` was used directly, causing handlers to be added to wrong command buffers.

**Fix**: Changed to `getCurrentMPSStream()` in both `addProfilerScheduledHandler()` and `addProfilerCompletedHandler()`.

**Lines**: 435-438, 474-487

### 3. MPSStream.mm - Auto-Stream Assignment
**Problem**: New threads defaulted to stream 0, causing all threads to share the same command buffer.

**Fix**: Modified `getCurrentStream()` to auto-assign streams from the pool:
- First thread (main): Gets stream 0 for backward compatibility
- Subsequent threads: Auto-acquire streams 1-31 from pool

**Lines**: 348-374

## Files Changed

| File | Lines Changed | Description |
|------|---------------|-------------|
| MPSStream.h | +107 | MPSStreamPool class declaration |
| MPSStream.mm | +136 | Pool implementation + auto-assign |
| MPSAllocator.mm | +9 | Thread-safe completion handlers |
| MPSProfiler.mm | +18 | Thread-safe profiler handlers |
| MPSGuardImpl.h | +25 | Pool-aware stream methods |

## Patch File

`patches/003-thread-safe-mps-stream-pool.patch` (423 lines)

This comprehensive patch includes all changes from the original stream pool implementation plus the thread-safety fixes.

## Key Design Decisions

1. **Auto-stream assignment**: New threads automatically get their own stream from the pool. This enables parallel execution without requiring explicit stream management from user code.

2. **Main thread detection**: The first thread to use MPS gets stream 0 (default) for backward compatibility with single-threaded code.

3. **Thread-local storage**: Each thread's current stream is stored in TLS (`thread_local`), enabling efficient lookup without locking.

4. **Round-robin allocation**: Streams 1-31 are allocated round-robin to new threads, distributing work across the pool.

## Validation

The error that was previously occurring:
```
-[_MTLCommandBuffer addScheduledHandler:]:787: failed assertion `Scheduled handler provided after commit call'
```

This has been eliminated by ensuring each thread uses its own stream and command buffer.

## Next Steps

1. **Phase 6**: More comprehensive testing with real models (NLLB, Kokoro TTS)
2. **Phase 7**: Performance benchmarking - measure actual speedup vs mutex-based serialization
3. **Phase 8**: Prepare upstream PR to pytorch/pytorch

## Files Created

- `tests/test_parallel_mps.py` - Comprehensive parallel test
- `tests/test_parallel_mps_simple.py` - Simple parallel test
- `tests/test_stream_assignment.py` - Stream assignment verification
- `patches/003-thread-safe-mps-stream-pool.patch` - Complete patch
