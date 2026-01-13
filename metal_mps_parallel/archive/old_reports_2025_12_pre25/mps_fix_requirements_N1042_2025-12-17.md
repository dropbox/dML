# What Apple Needs to Fix in MPS

**Date**: 2025-12-17
**Status**: MPS is closed-source; requires Apple engineering action
**Bug Report**: `APPLE_RADAR_FB123456.md` (ready for submission)

## Executive Summary

MetalPerformanceShaders has **internal global mutable state** that prevents concurrent kernel encoding. Apple needs to either:
1. Fix the internal state management to be thread-safe
2. Provide thread-safe alternative APIs
3. Document the limitations and recommend alternatives

## The Root Cause

### What We Know

From PyTorch's MPS backend comments and crash analysis:

```cpp
// pytorch-mps-fork/aten/src/ATen/native/mps/operations/Linear.mm:15-19
// THREAD-SAFETY: Global mutex for MPSNDArrayMatrixMultiplication encoding.
// Apple's MPS framework has internal shared state that makes concurrent encoding
// of MPSNDArrayMatrixMultiplication kernels unsafe, even with per-thread instances.
static std::mutex s_linear_nograph_mutex;
```

### The Problem

Even when users create **separate kernel instances per thread**:

```objc
// Thread 1
MPSNDArrayMatrixMultiplication *kernel1 = [[MPSNDArrayMatrixMultiplication alloc]
    initWithDevice:device];
[kernel1 encodeToCommandBuffer:cmdBuf1 ...];

// Thread 2 (concurrent)
MPSNDArrayMatrixMultiplication *kernel2 = [[MPSNDArrayMatrixMultiplication alloc]
    initWithDevice:device];
[kernel2 encodeToCommandBuffer:cmdBuf2 ...];  // CRASH!
```

The crash occurs because MPS kernels share **internal global state** that isn't thread-safe.

### Crash Location (from stack traces)

```
Thread 1: EXC_BAD_ACCESS
  MPSSetResourcesOnCommandEncoder + 0x...
  -[MPSNDArrayMatrixMultiplication encodeToCommandEncoder:...]
  ...

Thread 2: EXC_BAD_ACCESS
  MPSSetResourcesOnCommandEncoder + 0x...  // Same internal function
```

## What Apple Needs to Change

### Option 1: Fix Internal State Management (Preferred)

Apple should make MPS kernel instances **truly independent**:

```objc
// Current (broken): Kernel instances share internal state
@implementation MPSNDArrayMatrixMultiplication {
    // Internal state that's actually global/shared (BUG)
    static _MPSInternalEncoder* s_shared_encoder;  // PROBLEM
}

// Fixed: Each instance has independent state
@implementation MPSNDArrayMatrixMultiplication {
    _MPSInternalEncoder* _encoder;  // Per-instance, not shared
}
```

**Technical Requirements**:
1. Remove all static/global mutable state from kernel implementations
2. Use per-instance encoders and scratch buffers
3. Ensure `MTLDevice` and `MTLCommandQueue` access is thread-safe

### Option 2: Provide Thread-Safe API

Add explicit thread-safe variants:

```objc
// New API option
@interface MPSNDArrayMatrixMultiplication : MPSKernel

// Existing (not thread-safe for concurrent encoding)
- (instancetype)initWithDevice:(id<MTLDevice>)device;

// NEW: Thread-safe variant
- (instancetype)initWithDevice:(id<MTLDevice>)device
                    threadSafe:(BOOL)threadSafe;

// Or: Concurrent encoding method
- (void)encodeConcurrentlyToCommandBuffer:(id<MTLCommandBuffer>)cmdBuf
                             sourceArrays:(NSArray<MPSNDArray*>*)srcs
                         destinationArray:(MPSNDArray*)dst;
@end
```

### Option 3: Per-Queue Kernel Isolation

Allow binding kernel instances to specific command queues:

```objc
// New API option
@interface MPSNDArrayMatrixMultiplication : MPSKernel

// Associate kernel with a specific queue (enables concurrency)
- (instancetype)initWithDevice:(id<MTLDevice>)device
                  commandQueue:(id<MTLCommandQueue>)queue;

@end
```

This would let each thread use kernels bound to their own queue.

### Option 4: Document and Recommend MPSGraph

Apple could officially document that:
1. MPS kernels are not thread-safe for concurrent encoding
2. MPSGraph should be used for concurrent workloads
3. MPSGraph has thread-safe graph execution

```
// Recommended documentation update
/**
 * @discussion MPSNDArrayMatrixMultiplication encodes work to a command buffer.
 *
 * THREAD SAFETY: This kernel is NOT safe for concurrent encoding from
 * multiple threads, even with separate kernel instances. For concurrent
 * workloads, use MPSGraph instead.
 */
```

## Affected MPS Classes

Based on our analysis, these MPS classes have thread-safety issues:

| Class | Issue | PyTorch Workaround |
|-------|-------|-------------------|
| `MPSNDArrayMatrixMultiplication` | Internal shared state | `s_linear_nograph_mutex` |
| `MPSNDArrayIdentity` | Internal shared state | `s_ndarray_identity_mutex` |
| `MPSMatrixDecompositionLU` | Internal shared state | `s_lu_decomposition_mutex` |
| `MPSMatrixSolveLU` | Internal shared state | `s_lu_solve_mutex` |
| `MPSMatrixSolveTriangular` | Internal shared state | `s_solve_triangular_mutex` |
| Metal compute kernels (via MPS) | Dispatch threading issues | `s_layer_norm_mutex` |

## How to Report to Apple

### 1. File Feedback Assistant Report

**URL**: https://feedbackassistant.apple.com

**Report**: Use `APPLE_RADAR_FB123456.md` as template

**Category**: Developer Technologies > Metal > MetalPerformanceShaders

### 2. WWDC Labs (if available)

Request a Metal/MPS lab appointment to discuss with Apple engineers directly.

### 3. Apple Developer Forums

Post to: https://developer.apple.com/forums/tags/metalperformanceshaders

### 4. DTS (Developer Technical Support)

If you have a paid Apple Developer account, submit a technical support incident.

## Evidence Package for Apple

When submitting, include:

1. **Reproduction Code**: `APPLE_RADAR_FB123456.md` contains minimal repro
2. **Crash Logs**: Stack traces showing `MPSSetResourcesOnCommandEncoder` crash
3. **Environment**: macOS version, Xcode version, Apple Silicon model
4. **Impact Statement**: "This blocks thread-safe ML inference in PyTorch/TensorFlow/etc."
5. **MLX Reference**: "Note: Apple's own MLX framework avoids MPS for this reason"

## Likelihood of Fix

| Factor | Assessment |
|--------|------------|
| Impact | High (affects all ML frameworks using MPS) |
| MLX avoids MPS | Apple knows about this |
| Fix complexity | Medium (requires internal refactoring) |
| Apple priority | Unknown |

**Realistic expectation**: This is a deep architectural issue. Fix might require:
- macOS 16+ (next major release)
- MPS 2.0 or equivalent rewrite
- May never be fixed if Apple considers MLX the solution

## Our Recommended Path

Given uncertainty about Apple fixing MPS:

1. **Submit radar anyway** - Document the issue officially
2. **Implement Steel integration** - Don't wait for Apple
3. **Contribute to MLX** - Help improve the alternative
4. **Document for users** - Explain MPS limitations in PyTorch docs

## Conclusion

Fixing MPS requires Apple engineering effort on closed-source code. We can:
1. Report the bug with comprehensive evidence
2. Work around it with Steel kernels (MLX approach)
3. Document limitations for users

Apple's own ML team chose option 2 (avoid MPS entirely). This suggests they may not prioritize fixing MPS for concurrent workloads.
