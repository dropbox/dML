# Phase 0: Research and Setup Report

**Worker**: N=0
**Date**: 2025-12-12
**Status**: COMPLETE

---

## 1. Environment Setup

### PyTorch Fork
- **Location**: `~/metal_mps_parallel/pytorch-mps-fork/`
- **Version**: v2.9.1
- **Actual Git Hash**: `d38164a545b4a4e4e0cf73ce67173f70574890b6`
- **Plan's Hash** (incorrect): `5811a8d7da873dd699ff6687092c225caffcf1bb`

**Note**: The plan also mentioned `git checkout v2.5.1` which was a typo. The correct version is v2.9.1.

---

## 2. CUDA Stream Pool Analysis

**Source Files**:
- `c10/cuda/CUDAStream.h` (269 lines)
- `c10/cuda/CUDAStream.cpp` (381 lines)

### Key Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CUDA Stream Pool Design                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Per-Device Pools (lazily initialized):                         │
│  ├── Default Pool: 1 stream (ID 0, the NULL stream)             │
│  ├── Low Priority Pool: 32 streams (round-robin allocation)     │
│  └── High Priority Pool: 32 streams (round-robin allocation)    │
│                                                                  │
│  Thread-Local Current Streams:                                   │
│  └── thread_local std::unique_ptr<StreamId[]> current_streams   │
│                                                                  │
│  Key Constants:                                                  │
│  └── kStreamsPerPool = 32 (1 << 5)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `getStreamFromPool(priority, device)` | Get stream from pool (round-robin via atomic counter) |
| `getCurrentCUDAStream(device)` | Get thread-local current stream |
| `setCurrentCUDAStream(stream)` | Set thread-local current stream |
| `getDefaultCUDAStream(device)` | Get default stream (ID 0) |

### Round-Robin Allocation

```cpp
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
    auto raw_idx = counter++;
    return raw_idx % kStreamsPerPool;  // 32 streams
}
```

### Stream ID Encoding (64-bit)

```
-- 54 bits --  -- 5 bits -----  -- 4 bits --     --1 bit --
zeros          stream id index  StreamIdType     Ext/native stream
```

### Key Insight
CUDA streams are thread-safe. The same stream CAN be used by multiple threads. The pool distributes work across streams to enable parallelism, not to provide exclusive access.

---

## 3. MPS Backend Analysis

**Primary Source Files**:
- `aten/src/ATen/mps/MPSStream.h` (159 lines)
- `aten/src/ATen/mps/MPSStream.mm` (293 lines)
- `aten/src/ATen/mps/MPSGuardImpl.h` (183 lines)
- `aten/src/ATen/mps/MPSEvent.h` (106 lines)
- `aten/src/ATen/mps/MPSAllocator.h` (560+ lines)

### THE PROBLEM: Singleton Pattern

**MPSStream.mm lines 273-290:**
```objc
MPSStream* MPSStreamImpl::_stream = nullptr;

MPSStream* MPSStreamImpl::getInstance() {
    if (_stream == nullptr) {
        _stream = new MPSStream(Stream(Stream::UNSAFE,
                                       c10::Device(DeviceType::MPS, 0), 0));
    }
    return _stream;
}

MPSStream* getCurrentMPSStream() {
    return getDefaultMPSStream();  // Returns singleton!
}

MPSStream* getDefaultMPSStream() {
    return MPSStreamImpl::getInstance();  // Returns singleton!
}
```

**Result**: ALL threads share ONE `MPSStream` with ONE `MTLCommandQueue` and ONE `MPSCommandBuffer`. Concurrent access crashes.

### MPSStream Internal Structure

Each `MPSStream` owns:
- `MTLCommandQueue_t _commandQueue` - Metal command queue
- `MPSCommandBuffer_t _commandBuffer` - Current command buffer
- `dispatch_queue_t _serialQueue` - Serializes operations

**Serialization via dispatch_sync**:
```objc
void MPSStream::fill(...) {
    dispatch_sync(_serialQueue, ^() {  // BLOCKS ALL THREADS
        @autoreleasepool {
            // ... work ...
        }
    });
}
```

### MPSGuardImpl - No Pool Support

**MPSGuardImpl.h lines 70-86:**
```cpp
Stream getStream(Device d) const override {
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
}

Stream getNewStream(Device, int priority = 0) const override {
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
}

Stream getDefaultStream(Device d) const override {
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
}
```

ALL methods return the DEFAULT stream. No pool support exists.

### MPSEvent - Already Pool-Capable

`MPSEventPool` exists and supports:
- Event recording on streams
- Waiting for events
- Cross-stream synchronization

This is good - we can reuse this for cross-stream sync.

### MPSAllocator - Based on CUDA

- Modeled after `CUDACachingAllocator`
- Uses heap-based allocation
- Has `MPSEventPtr` for sync
- Will need review for multi-stream safety

---

## 4. Files to Modify

### Primary Targets

| File | Changes |
|------|---------|
| `aten/src/ATen/mps/MPSStream.h` | Add `MPSStreamPool` class, thread-local support |
| `aten/src/ATen/mps/MPSStream.mm` | Implement pool, change `getCurrentMPSStream()` |
| `aten/src/ATen/mps/MPSGuardImpl.h` | Update to use pool streams |
| `aten/src/ATen/mps/MPSGuardImpl.mm` | Implement pool-aware guards |

### Secondary Targets

| File | Changes |
|------|---------|
| `aten/src/ATen/mps/MPSAllocator.mm` | Verify thread safety, add stream tracking if needed |
| `aten/src/ATen/mps/MPSEvent.mm` | Cross-stream synchronization |

### New Files (per plan)

| File | Purpose |
|------|---------|
| `aten/src/ATen/mps/MPSStreamPool.h` | Pool interface (optional, could be in MPSStream.h) |
| `aten/src/ATen/mps/MPSStreamPool.mm` | Pool implementation |

---

## 5. Proposed Implementation (Phase 1)

### MPSStreamPool Design

```cpp
class MPSStreamPool {
public:
    static MPSStreamPool& instance();

    // Get stream from pool (round-robin)
    MPSStream* getStreamFromPool();

    // Thread-local current stream
    static MPSStream* getCurrentStream();
    static void setCurrentStream(MPSStream* stream);

private:
    static constexpr int kStreamsPerPool = 32;
    std::array<std::unique_ptr<MPSStream>, kStreamsPerPool> streams_;
    std::atomic<uint32_t> next_stream_idx_{0};

    static thread_local MPSStream* current_stream_;
};
```

### Key Changes to getCurrentMPSStream()

**Before (singleton):**
```cpp
MPSStream* getCurrentMPSStream() {
    return getDefaultMPSStream();
}
```

**After (pool-aware):**
```cpp
MPSStream* getCurrentMPSStream() {
    return MPSStreamPool::getCurrentStream();
}
```

### Backward Compatibility

- `getDefaultMPSStream()` still returns stream 0 (for single-threaded code)
- New threads automatically get pool streams
- No API changes required for users

---

## 6. Risk Analysis

| Risk | Mitigation |
|------|------------|
| Command buffer race conditions | Each stream has own command queue + buffer |
| Memory allocation conflicts | Allocator already thread-safe (mutex protected) |
| Cross-stream tensor access | Use MPSEvent for synchronization |
| Performance regression (single-thread) | Stream 0 remains default, no overhead |

---

## 7. Next Steps for Worker N=1

1. **Create MPSStreamPool class** in `MPSStream.h`
2. **Implement pool** in `MPSStream.mm`:
   - 32 pre-allocated streams (or lazy init)
   - Round-robin allocation via atomic counter
   - Thread-local current stream storage
3. **Update `getCurrentMPSStream()`** to use pool
4. **Keep `getDefaultMPSStream()`** returning stream 0 for compatibility
5. **Test basic compilation** before moving to Phase 2

---

## 8. Verification Commands

```bash
# Verify fork location
ls ~/metal_mps_parallel/pytorch-mps-fork/aten/src/ATen/mps/

# Verify version
cd ~/metal_mps_parallel/pytorch-mps-fork && git describe --tags

# Key files exist
ls -la ~/metal_mps_parallel/pytorch-mps-fork/aten/src/ATen/mps/MPSStream.*
ls -la ~/metal_mps_parallel/pytorch-mps-fork/c10/cuda/CUDAStream.*
```

---

**Phase 0 Complete. Ready for Phase 1 implementation.**
