# Apple Framework Assumptions

## Purpose

Our formal verification (TLA+, Iris/Coq, CBMC, TSA) proves properties of **our code**.
However, we depend on Apple's closed-source frameworks behaving correctly.
This document explicitly lists those assumptions.

**If any assumption is false, our proofs may not hold on all platforms.**

---

## Runtime-Verified Assumptions

These assumptions are checked by `verification/run_platform_checks`:

### A.001: MTLSharedEvent Atomicity

**Statement**: `MTLSharedEvent.signaledValue` updates are atomic and immediately visible to all threads.

**Source**: Apple Metal documentation states shared events provide CPU-GPU synchronization.

**Verification**: Runtime test with 8 threads doing 1000 increments each. Pass condition: no lost updates.

**Risk if false**: Lost signals cause synchronization failures, potential data races in dependent code.

**Status**: VERIFIED (M4 Max, macOS 15.7.3)

### A.002: MTLCommandQueue Thread Safety

**Statement**: Multiple threads can submit command buffers to **different** MTLCommandQueue instances concurrently without corruption.

**Source**: Apple Metal Best Practices Guide: "Command queues are thread-safe"

**Verification**: Runtime test with 8 queues, each receiving 100 command buffers concurrently.

**Risk if false**: GPU command corruption, crashes, undefined behavior.

**Status**: VERIFIED (M4 Max, macOS 15.7.3)

### A.003: Sequential Consistency Memory Ordering

**Statement**: `std::memory_order_seq_cst` provides global total order on Apple Silicon.

**Source**: ARM64 memory model guarantees, Apple Silicon AArch64 compliance.

**Verification**: Dekker's algorithm test with 100,000 critical section entries per thread.

**Risk if false**: Race conditions in "correct" lock-free code.

**Status**: VERIFIED (M4 Max, macOS 15.7.3)

### A.004: Unified Memory Coherency

**Statement**: CPU and GPU see consistent view of `MTLResourceStorageModeShared` buffers after appropriate synchronization.

**Source**: Apple Silicon unified memory architecture documentation.

**Verification**: CPU read-after-write test for 1024 elements.

**Risk if false**: Data corruption, stale reads, subtle computation errors.

**Status**: VERIFIED (M4 Max, macOS 15.7.3)

### A.005: @autoreleasepool Semantics

**Statement**: `@autoreleasepool` correctly releases all autoreleased objects at scope exit.

**Source**: Clang ARC documentation, Objective-C memory management specification.

**Verification**: 100 iterations creating and abandoning Metal objects in autoreleasepool.

**Risk if false**: Memory leaks, resource exhaustion.

**Status**: VERIFIED (M4 Max, macOS 15.7.3)

### A.006: Stream Isolation

**Statement**: Operations on different MTLCommandQueues do not interfere with each other's data.

**Source**: Implied by Metal's command queue abstraction.

**Verification**: 4 streams writing unique patterns to separate buffers, verifying no cross-contamination.

**Risk if false**: Silent data corruption between concurrent operations.

**Status**: VERIFIED (M4 Max, macOS 15.7.3)

---

## Trusted Assumptions (Not Runtime-Verified)

These assumptions cannot be easily verified at runtime but are critical:

### AF.001: Metal Driver Stability

**Statement**: The Metal driver (AGX firmware) does not crash or hang under concurrent GPU load.

**Source**: Trust in Apple's quality assurance.

**Verification**: Empirical (our tests don't crash).

**Risk if false**: System hangs, GPU reset required.

**Status**: TRUSTED

### AF.002: MTLDevice Registry ID Uniqueness

**Statement**: Each MTLDevice has a unique `registryID` that persists for the device lifetime.

**Source**: Apple documentation.

**Verification**: Not verified (no multi-device test setup).

**Risk if false**: Device confusion in multi-GPU scenarios (irrelevant for most Apple Silicon).

**Status**: TRUSTED

### AF.003: Callback Block Safety

**Statement**: Objective-C blocks captured for Metal callbacks remain valid until the callback fires.

**Source**: Objective-C block semantics, ARC specification.

**Verification**: Implicitly tested by our callback-using tests.

**Risk if false**: Use-after-free in callback handlers.

**Status**: TRUSTED (with careful coding patterns)

---

## Assumptions Known to be FALSE

These document cases where Apple's frameworks do NOT behave as one might expect:

### AF.X01: MPS Framework Thread Safety (SINGLE OPERATIONS)

**Statement**: Individual MPS operations (LayerNorm, SDPA) can be executed concurrently from multiple threads.

**Status**: **FALSE** - This is why this project exists.

**Evidence**: Crashes at 4+ threads without our batching workaround.

**Our Solution**: Request batching serializes access to MPS operations while allowing concurrent user threads.

### AF.X02: MPSGraph Concurrent Compilation

**Statement**: Multiple MPSGraph instances can compile concurrently.

**Status**: **FALSE** - Graph compilation appears to have shared state.

**Evidence**: Race conditions observed during concurrent graph creation.

**Our Solution**: Single worker thread for graph operations.

---

## Platform-Specific Assumptions

### Hardware Variations

| Assumption | M1 | M2 | M3 | M4 | Ultra |
|------------|----|----|----|----|-------|
| A.001 Event Atomicity | Untested | Untested | Untested | VERIFIED | Untested |
| A.002 Queue Safety | Untested | Untested | Untested | VERIFIED | Untested |
| A.003 Memory Ordering | Untested | Untested | Untested | VERIFIED | Untested |

### Potential Platform Risks

1. **M3+ Dynamic Caching**: May affect threadgroup memory behavior (untested).
2. **Ultra dual-die**: May affect memory coherency across UltraFusion (untested).
3. **Low-memory (8GB)**: May affect OOM behavior (untested).
4. **Older macOS**: Metal framework may behave differently (untested).

---

## Version-Specific Assumptions

### macOS Versions

| Version | Metal | Status | Notes |
|---------|-------|--------|-------|
| 13.x | 3.0 | Untested | Baseline Metal 3 |
| 14.x | 3.1 | Untested | May have fixes/changes |
| 15.x | 3.2 | VERIFIED | Current development platform |

### PyTorch Versions

Our patches are developed against PyTorch v2.9.1. Behavior with other versions is not verified.

---

## How to Verify on Your Platform

Run the runtime assumption verification:

```bash
cd verification
make
./run_platform_checks
```

Expected output should show all checks PASS. If any check fails:

1. **Document the failure** in reports/main/
2. **Do not run production workloads** until investigated
3. **Report to project maintainers** with platform details

---

## Adding New Assumptions

When adding code that depends on undocumented Apple behavior:

1. Add the assumption to this document
2. If possible, add a runtime verification check
3. Document the risk if the assumption is false
4. Note which platforms have been verified

---

## References

- [Apple Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [Metal Programming Guide](https://developer.apple.com/documentation/metal/)
- [ARM Architecture Reference Manual](https://developer.arm.com/documentation/ddi0487/latest)
- [Clang ARC Documentation](https://clang.llvm.org/docs/AutomaticReferenceCounting.html)

---

## Changelog

- 2025-12-19: Initial document created (N=1317)
  - 6 runtime-verified assumptions
  - 3 trusted assumptions
  - 2 known-false assumptions documented
