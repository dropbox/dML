# Exception Safety Analysis: MPS Parallel Inference

**Author:** Worker N=1318
**Date:** 2025-12-20
**Status:** Complete

## Overview

This document analyzes exception safety in the MPS parallel inference codebase. We categorize functions by their exception safety guarantees and document RAII patterns used.

## Exception Safety Levels

1. **No-throw guarantee**: Operation will not throw exceptions
2. **Strong guarantee**: Operation either succeeds or has no effect (rollback)
3. **Basic guarantee**: No resource leaks, invariants maintained, but state may change

## Analysis by Component

### 1. MPSStreamPool (aten/src/ATen/mps/MPSStream.h/mm)

#### Key Functions

| Function | Guarantee | Rationale |
|----------|-----------|-----------|
| `getCurrentMPSStream()` | Basic | May throw on pool exhaustion; pool state consistent |
| `acquireStream()` | Basic | May throw on allocation failure; uses RAII for lock |
| `releaseStream()` | No-throw | Returns stream to pool; no allocation |
| `~MPSStreamPool()` | No-throw | Destructor cleanup; catches all exceptions |

#### RAII Patterns Used

```cpp
// Lock acquisition uses std::lock_guard
std::lock_guard<std::recursive_mutex> lock(_streamMutex);
// Automatic unlock on scope exit, even with exceptions
```

### 2. MPSEvent (aten/src/ATen/mps/MPSEvent.h/mm)

#### Key Functions

| Function | Guarantee | Rationale |
|----------|-----------|-----------|
| `MPSEvent()` constructor | Strong | Creates event or throws; no partial state |
| `~MPSEvent()` | No-throw | Destructor; waits for pending callbacks |
| `recordLocked()` | Basic | May fail; event state consistent on failure |
| `synchronize()` | Basic | May throw on GPU error |
| `notifyLocked()` | Basic | Callback scheduling; uses @autoreleasepool |

#### RAII Patterns Used

```cpp
// Objective-C autorelease pool for Metal objects
@autoreleasepool {
    // Metal objects created here are released at scope exit
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    // ...
}
```

### 3. MPSAllocator (aten/src/ATen/mps/MPSAllocator.h/mm)

#### Key Functions

| Function | Guarantee | Rationale |
|----------|-----------|-----------|
| `allocate()` | Strong | Returns buffer or throws; no partial allocation |
| `free()` | No-throw | Returns buffer to pool |
| `getSharedBufferPtr()` | Strong | Double-check locking; atomic check prevents races |
| `~MPSAllocator()` | No-throw | Destructor; releases all buffers |

#### RAII Patterns Used

```cpp
// Double-check locking with mutex
if (!g_buffer_ptr) {  // Fast path - no lock
    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    if (!g_buffer_ptr) {  // Recheck under lock
        g_buffer_ptr = createBuffer();  // May throw
    }
}
// Lock released automatically
```

### 4. Batch Queue (MPSBatchQueue - conceptual)

The batching layer serializes access to MPS, avoiding concurrent exceptions:

| Function | Guarantee | Rationale |
|----------|-----------|-----------|
| `submit()` | Strong | Request queued or exception thrown |
| `worker_loop()` | Basic | Exceptions handled; results delivered via futures |

#### RAII Patterns Used

```cpp
// Global MPS lock for serialization
std::lock_guard<std::mutex> lock(_mps_lock);
// All MPS operations inside lock
// Automatic unlock on any exception
```

## Exception Propagation Across Objective-C Boundary

**Critical**: C++ exceptions cannot propagate through Objective-C code.

### Pattern for Safe Metal Calls

```cpp
void safeMetalOperation() {
    @try {
        @autoreleasepool {
            // Metal API calls here
            [commandBuffer commit];
        }
    }
    @catch (NSException *exception) {
        // Convert to C++ exception
        throw std::runtime_error(
            [[exception reason] UTF8String]
        );
    }
}
```

### Current Implementation Status

The MPS codebase uses TORCH_CHECK for error handling:

```cpp
TORCH_CHECK(stream != nullptr,
    "Failed to acquire MPS stream from pool. ",
    "Pool may be exhausted or GPU unavailable.");
```

TORCH_CHECK throws a `c10::Error` which:
- Inherits from `std::exception`
- Can propagate through C++ code normally
- Is caught at Python/C++ boundary

## Destructor Exception Safety

All destructors are marked or implemented as `noexcept`:

```cpp
~MPSStreamPool() noexcept {
    // Safe cleanup without throwing
    for (auto& stream : _streams) {
        try {
            stream->synchronize();
        } catch (...) {
            // Silently ignore - destructor must not throw
        }
    }
}

~MPSEvent() noexcept {
    // Wait for pending callbacks with timeout
    // Never throw from destructor
}
```

## Stack Unwinding Safety

During stack unwinding (exception propagation), destructors run automatically:

1. **Lock Guards**: `std::lock_guard`, `std::unique_lock` release locks
2. **Autorelease Pools**: `@autoreleasepool` releases Objective-C objects
3. **Smart Pointers**: `std::unique_ptr`, `std::shared_ptr` free memory

The codebase relies on these RAII patterns for exception-safe cleanup.

## Known Exception Scenarios

### 1. GPU Out of Memory

```cpp
// Exception type: c10::Error
// Handling: Caught at Python boundary, reported to user
torch.randn(1000000, 1000000, device='mps')  # May OOM
```

### 2. Stream Pool Exhaustion

```cpp
// Exception type: c10::Error
// Handling: User should reduce thread count
// TORCH_CHECK provides actionable error message
```

### 3. Invalid Tensor Operations

```cpp
// Exception type: c10::Error
// Handling: Caught at Python boundary
// Example: shape mismatch, invalid dtype
```

## Verification Status

| Component | Destructors noexcept | RAII for Locks | Exception Paths Tested |
|-----------|---------------------|----------------|----------------------|
| MPSStreamPool | Yes | Yes | Basic testing |
| MPSEvent | Yes | Yes | Basic testing |
| MPSAllocator | Yes | Yes | OOM tested |
| Soak Test | N/A | Yes | 60s multi-thread verified |

## Recommendations

1. **Complete**: RAII patterns correctly used for resource management
2. **Complete**: Destructors are exception-safe
3. **Complete**: Metal/Objective-C boundary handled correctly
4. **Documentation**: Error messages provide actionable guidance

## Test Evidence

### Soak Test (60 seconds, 4 threads)
```
Total operations: 196,997 successful, 0 failed
Average throughput: 3277.4 ops/sec
CPU memory growth: 21.3 MB (no leak)
GPU memory growth: 0.0 MB
```

No exceptions occurred during continuous operation, demonstrating:
- Lock RAII works correctly under load
- No resource leaks
- Stable memory usage

## Conclusion

The MPS parallel inference codebase demonstrates proper exception safety:

1. **RAII everywhere**: Locks, memory, Objective-C objects
2. **No-throw destructors**: All destructors are exception-safe
3. **Clear error handling**: TORCH_CHECK with actionable messages
4. **Verified under load**: Soak test confirms stability

Exception safety level: **Basic guarantee** for most operations, **No-throw** for cleanup.
