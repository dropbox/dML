# Apple Framework Assumptions

This document enumerates all assumptions our verification makes about Apple's closed-source
Metal, MPS, and Objective-C runtime frameworks. These assumptions cannot be formally verified
because Apple does not publish source code or formal specifications.

## Purpose

Our TLA+/Apalache/CBMC verification proves correctness **assuming** Apple frameworks behave
as documented. This document:

1. Explicitly lists each assumption
2. Provides evidence supporting each assumption
3. Describes runtime checks that validate assumptions in practice

## MTLCommandQueue Thread Safety

### Assumption A1: Command Queues Are Thread-Safe

**Statement**: Multiple threads can call `commandBuffer` on different `MTLCommandQueue`
instances concurrently without corruption.

**Evidence**:
- Apple documentation: "Command queues are thread-safe."
  [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/metal_best_practices_guide/resource_management)
- MLX library uses multiple command queues concurrently (proven in production)
- Our stress tests run 8+ threads with separate queues without crashes

**Runtime Check**:
```python
# tests/verify_command_queue_thread_safety.py
def test_concurrent_command_queues():
    queues = [device.newCommandQueue() for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(submit_work, q) for q in queues]
        # All must complete without error
        for f in futures:
            f.result()
```

**Risk if False**: Memory corruption, undefined behavior.

---

## MTLCommandBuffer Thread Safety

### Assumption A2: Command Buffers Are NOT Thread-Safe

**Statement**: A single `MTLCommandBuffer` must not be accessed from multiple threads
simultaneously. However, different command buffers can be accessed concurrently.

**Evidence**:
- Apple documentation: "Don't write to a resource while a command buffer that reads from
  it is in flight."
- Observed crashes when violating this (the AGX driver race we're fixing)

**Our Mitigation**: Each stream owns its command buffer; mutex serializes access.

**Runtime Check**: `AGX_FIX_VERBOSE=1` logs concurrent access attempts.

---

## MTLSharedEvent Semantics

### Assumption A3: signaledValue Is Atomic and Monotonic

**Statement**: `MTLSharedEvent.signaledValue` updates atomically. Once a value V is
observed, all values < V have also been signaled.

**Evidence**:
- Apple documentation: "The event's signaled value is monotonically increasing."
- Metal API design implies acquire-release semantics at signal points
- Empirical testing with 8-thread synchronization shows no reordering

**Runtime Check**:
```cpp
// In MPSEvent.mm
assert(old_value <= new_value && "Event value must be monotonic");
```

**Risk if False**: Synchronization failures, lost GPU work, data races.

---

### Assumption A4: Listener Callbacks Execute Sequentially Per Event

**Statement**: For a single `MTLSharedEvent`, listener blocks are executed in signal
order, not concurrently.

**Evidence**:
- Observed behavior: callbacks for value N always complete before callbacks for N+1
- No documentation contradicts this
- Would be a fundamental API flaw if false

**Runtime Check**:
```cpp
thread_local bool in_callback = false;
void callback_handler() {
    assert(!in_callback && "Callbacks must not nest");
    in_callback = true;
    // ... handle ...
    in_callback = false;
}
```

---

## Objective-C Runtime

### Assumption A5: objc_msgSend Is Thread-Safe for Different Objects

**Statement**: Concurrent `objc_msgSend` calls on different receiver objects are safe.

**Evidence**:
- Apple documentation: "The Objective-C runtime is thread-safe."
- Universal iOS/macOS practice

**Risk if False**: Entire Objective-C ecosystem would be broken.

---

### Assumption A6: Class Method Swizzling Is Atomic

**Statement**: `method_setImplementation` atomically replaces the IMP pointer.

**Evidence**:
- Apple runtime source (publicly available) uses atomic operations
- Millions of apps use runtime swizzling safely

**Our Usage**: `agx_fix/src/agx_fix_v2_5.mm` swizzles at library load time
(constructor), before any encoder methods are called.

---

## MPSGraph Thread Safety

### Assumption A7: MPSGraph Instances Are NOT Thread-Safe

**Statement**: A single `MPSGraph` instance must not be executed concurrently
from multiple threads.

**Evidence**:
- No Apple documentation claims thread safety
- PyTorch MPS backend serializes graph execution per stream
- Observed crashes when violating this

**Our Mitigation**: Per-stream MPSGraph instances, never shared.

---

### Assumption A8: MPSGraphCompilationDescriptor Is Immutable After Creation

**Statement**: Once created, compilation descriptors don't change and can be safely
referenced from multiple threads.

**Evidence**:
- Immutable descriptor pattern is standard in Apple APIs
- No observed issues with descriptor sharing

---

## Memory Ordering (ARM)

### Assumption A9: Apple Silicon Implements ARM v8.x Memory Model

**Statement**: Apple M-series chips implement the ARMv8 memory model with:
- Data-dependency ordering
- Acquire-release semantics for appropriate instructions
- No store-to-load forwarding between different addresses

**Evidence**:
- Apple states M-series is ARM-compatible
- ARMv8 is formally specified
- Our CBMC harnesses assume ARM PSO model

**Risk if False**: Subtle memory ordering bugs possible.

---

## AGX Driver Behavior

### Assumption A10: AGX Encoder Operations Access _impl Pointer

**Statement**: Metal encoder methods (setBuffer, dispatchThreads, etc.) read from
a context structure via the `_impl` pointer. If `_impl` is NULL, the operation crashes.

**Evidence**:
- Crash dumps show fault addresses as small offsets (0x98, 0x5c8) from NULL
- Reverse engineering confirms `_impl` pattern
- Our fix (checking `_impl` before calling original) prevents crashes

**This is the bug we're fixing**: The AGX driver NULLs `_impl` AFTER releasing
a lock, creating a race window.

---

### Assumption A11: destroyImpl Lock Behavior

**Statement**: `AGXG16XFamilyComputeContext destroyImpl` holds a lock while
freeing resources, then NULLs `_impl` after releasing the lock.

**Evidence**:
- Disassembly analysis (`agx_patch/create_patch.py`)
- TLA+ model (`agx_patch/AGXRaceFix.tla`) matches crash behavior
- Binary patch that reorders NULL-before-unlock eliminates crashes

---

## Verification Coverage

| Assumption | TLA+ | CBMC | Runtime Check |
|------------|------|------|---------------|
| A1 Command Queue Thread Safety | MPSStreamPool.tla | - | stress tests |
| A2 Command Buffer Not Thread-Safe | MPSCommandBuffer.tla | command_buffer_harness.c | mutex guard |
| A3 Event signaledValue Atomic | MPSEvent.tla | - | monotonicity assert |
| A4 Listener Callbacks Sequential | MPSEvent.tla | event_pool_harness.c | callback guard |
| A5 objc_msgSend Thread-Safe | AGXObjCRuntime.tla | - | - |
| A6 Swizzling Atomic | - | - | constructor timing |
| A7 MPSGraph Not Thread-Safe | MPSGraphCache.tla | graph_cache_harness.c | per-stream |
| A8 Descriptor Immutable | - | - | - |
| A9 ARM Memory Model | MPSMemoryOrdering.tla | memory_ordering_harness.c | - |
| A10 _impl Access Pattern | AGXDylibFix.tla | - | is_impl_valid() |
| A11 destroyImpl Lock Order | AGXRaceFix.tla | - | binary patch |

## What If Assumptions Are Wrong?

### Detection
- Crashes with distinct patterns (NULL+offset for A10, PAC trap for A6)
- Data corruption observable in tensor values
- Deadlocks detectable via heartbeat monitoring

### Mitigation
- Conservative fallback to single-threaded mode
- Runtime validation with asserts
- Crash logging (`crash_logs/`) for post-mortem analysis

## Maintaining This Document

When adding new assumptions:
1. Add to this document with evidence and runtime checks
2. Add corresponding TLA+ property if applicable
3. Add CBMC harness if C-level verification is possible
4. Update the coverage table

Last updated: 2025-12-23 (N=1306)
