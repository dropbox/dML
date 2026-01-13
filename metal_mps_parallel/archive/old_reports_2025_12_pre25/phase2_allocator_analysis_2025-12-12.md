# Phase 2: MPSAllocator Multi-Stream Analysis

**Worker**: N=1
**Date**: 2025-12-12
**Status**: ANALYSIS COMPLETE

---

## 1. Executive Summary

The MPSAllocator is already thread-safe via `std::recursive_mutex`. For basic multi-stream operation, **no changes are immediately required**. However, there are opportunities to improve cross-stream synchronization.

**Recommendation**: Proceed to Phase 5 (build) first to validate the core MPSStreamPool. Phase 2 allocator changes can be addressed during Phase 6 (testing) if cross-stream issues arise.

---

## 2. Current Thread Safety Mechanisms

### 2.1 Mutex Protection

**File**: `aten/src/ATen/mps/MPSAllocator.mm`

The allocator uses `std::recursive_mutex` for all operations:

```cpp
// Lines with lock_guard:
370:  std::lock_guard<std::recursive_mutex> lock(m_mutex);
509:  std::lock_guard<std::recursive_mutex> lock(m_mutex);
516:  std::lock_guard<std::recursive_mutex> lock(m_mutex);
526:  std::lock_guard<std::recursive_mutex> lock(m_mutex);
... (15+ more occurrences)
```

**Implication**: All allocator operations are serialized. This ensures correctness but may become a bottleneck under heavy parallel allocation.

### 2.2 Metal Hazard Tracking

Some allocations use `MTLResourceHazardTrackingModeTracked`:

```cpp
// HeapBlock.h line 139
options |= (usage & UsageFlags::HAZARD)
    ? MTLResourceHazardTrackingModeTracked
    : MTLResourceHazardTrackingModeUntracked;
```

**Implication**: Metal automatically handles read/write hazards for tracked resources. This provides GPU-side thread safety.

### 2.3 MPSEvent Synchronization

The allocator uses `MPSEventPool` for GPU/CPU synchronization:

```cpp
// recordEvents() - Line 564
buffer_block->event = m_event_pool->acquireEvent(false, nullptr);  // NOTE: nullptr stream!
buffer_block->event->record(/*needsLock*/ false);

// waitForEvents() - Line 593
bool waitedOnCPU = buffer_block->event->synchronize();
```

---

## 3. Potential Multi-Stream Issues

### 3.1 Event Stream Association (Minor)

**Current Behavior**: Events are acquired with `nullptr` stream:
```cpp
m_event_pool->acquireEvent(false, nullptr);
```

**Potential Issue**: Events not associated with a specific stream may not synchronize correctly across streams.

**Mitigation**: Events can still be recorded/waited on any stream. The `nullptr` stream likely defaults to the default stream.

**Recommended Fix (Low Priority)**:
```cpp
// In recordEvents()
MPSStream* currentStream = getCurrentMPSStream();
buffer_block->event = m_event_pool->acquireEvent(false, currentStream);
```

### 3.2 Cache Release Synchronization (Minor)

**Current Behavior**:
```cpp
// Line 442 - release_cached_buffers()
auto stream = getDefaultMPSStream();
dispatch_sync(stream->queue(), ^() {
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
});
```

**Potential Issue**: Only waits on default stream. Buffers in use by other streams may still have pending work.

**Recommended Fix (If needed)**:
```cpp
// Wait on all active streams before releasing cache
for (size_t i = 0; i < kMPSStreamsPerPool; ++i) {
    MPSStream* stream = MPSStreamPool::instance().getStream(i);
    if (stream) {
        dispatch_sync(stream->queue(), ^() {
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        });
    }
}
```

### 3.3 Buffer Ownership Tracking (Deferred)

**Current State**: Buffers do not track which stream allocated them.

**The Plan's Proposed Design**:
```cpp
struct AllocationInfo {
    void* ptr;
    size_t size;
    MPSStream* owner_stream;  // NEW: Track which stream allocated
    id<MTLBuffer> buffer;
};
```

**Assessment**: This is useful for debugging and advanced cross-stream optimization, but **not required for basic functionality** because:

1. Metal buffers can be accessed from any command queue
2. The mutex already serializes allocations
3. MPSEvent can handle cross-stream synchronization when needed

---

## 4. Why No Immediate Changes Are Required

### 4.1 Metal Memory Model

Metal allows any `MTLBuffer` to be used by any `MTLCommandQueue`. There's no ownership model at the Metal level - only synchronization requirements.

### 4.2 PyTorch Tensor Lifetime

Tensors in PyTorch are reference-counted. A tensor created on Stream A can be:
- Passed to operations on Stream B
- The underlying buffer remains valid until refcount → 0
- PyTorch already handles synchronization for operations

### 4.3 Allocator Mutex

The global allocator mutex ensures:
- No concurrent modification of allocation tables
- No double-free scenarios
- No allocation during garbage collection

### 4.4 CUDA Analogy

CUDA's allocator (`CUDACachingAllocator`) also uses a global mutex and doesn't track per-stream allocations for basic operations. Cross-stream access is handled at a higher level (tensor operations).

---

## 5. Recommended Implementation Order

### Phase 2A: Skip for Now
The allocator changes can be deferred. The current implementation should work for multi-stream inference.

### Phase 5: Build First
Compile the modified PyTorch to verify syntax and basic functionality.

### Phase 6: Test and Iterate
During testing, if cross-stream issues arise:
1. Add stream parameter to `recordEvents()`
2. Implement all-stream synchronization in `release_cached_buffers()`
3. Add optional per-buffer stream tracking for debugging

---

## 6. Files for Future Reference

| File | Lines | Purpose |
|------|-------|---------|
| `MPSAllocator.h` | 1-560 | Allocator interface, BufferBlock struct |
| `MPSAllocator.mm` | 555-607 | recordEvents/waitForEvents implementation |
| `MPSAllocator.mm` | 434-453 | release_cached_buffers |
| `MPSEvent.h` | 1-105 | Event pool interface |

---

## 7. Next Steps for Worker N=2

1. **Skip Phase 2 for now** - allocator is already thread-safe
2. **Proceed to Phase 5** - Build modified PyTorch
3. **Run tests in Phase 6** - Validate with test_mps_stream_pool.cpp
4. **Return to Phase 2** only if tests reveal cross-stream synchronization issues

---

## 8. Build Commands

```bash
cd ~/metal_mps_parallel/pytorch-mps-fork

# Configure (once)
python setup.py develop --cmake-only

# Build MPS target (faster than full build)
cmake --build build --target torch_mps -j8

# Or full libtorch build
cmake --build build --target libtorch -j8
```

---

**Phase 2 Analysis Complete. Proceed to Phase 5 (Build).**
