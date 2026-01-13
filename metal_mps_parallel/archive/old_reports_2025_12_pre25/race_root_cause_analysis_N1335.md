# Phase 3.7: Root Cause Analysis - ATen MPS Race Condition

**Worker**: N=1335
**Date**: 2025-12-19
**Phase**: 3.7 - Deep code path analysis
**Previous Investigation**: N=1334 (Metal API isolation), N=1333 (C++ reproduction)

## Executive Summary

Through detailed code path analysis, I have identified the **specific location** of the race condition that causes `.contiguous()` failures in parallel MPS inference. The race occurs in the **Placeholder constructor** when binding ATen tensors to MPSGraph inputs/outputs.

## Key Finding

**The race condition is in `Placeholder::Placeholder()` in `OperationUtils.mm:542-642`**, specifically when processing non-contiguous tensors that require gathering.

## Race Condition Mechanism

### Call Stack

```
.contiguous() call
  └── Copy.mm::copy_kernel_mps() or similar copy operation
      └── gatherViewTensor(src, dst) [OperationUtils.mm:562]
          └── Placeholder::Placeholder() [OperationUtils.mm:542]
              ├── at::empty() [line 87 of View.mm] - Buffer allocation
              └── mpsStream->commandEncoder() [line 98] - Metal encoding
```

### The Problem

1. **MPSEncodingLock** only protects graph execution (`executeMPSGraphOnSerialQueue`)
2. **Placeholder constructor** is called BEFORE graph execution to prepare input/output bindings
3. **Placeholder can trigger Metal operations** via `gatherViewTensor()` which:
   - Allocates buffers via `at::empty()` → MPS allocator
   - Gets command encoder via `mpsStream->commandEncoder()` → Metal command buffer
   - Encodes Metal compute kernels → Metal encoding
4. These operations are **NOT** protected by MPSEncodingLock
5. Multiple threads calling `.contiguous()` concurrently can race in the Placeholder constructor

### Unprotected Code Path

```cpp
// OperationUtils.mm:557-572 - INSIDE Placeholder constructor
if ((!src.is_contiguous() || src.storage_offset()) && gatherTensorData) {
    // ...
    _tensor = gatherViewTensor(src, emptyShell);  // ← RACE HERE
    if (!_tensor.has_storage()) {
        _tensor = src.clone(MemoryFormat::Contiguous);  // ← ALSO RACE
    }
    srcBuf = getMTLBufferStorage(_tensor);
}
```

### gatherViewTensor Race Points

```cpp
// View.mm:84-133
Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst) {
    // Race point 1: Buffer allocation
    output = at::empty(src.sizes(), ...);  // ← NOT PROTECTED

    // Race point 2: Metal encoding
    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
        id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();  // ← NOT PROTECTED
        // ... Metal kernel encoding ...
    });
}
```

## Why Previous Findings Are Consistent

| Finding | Explanation |
|---------|-------------|
| Bare Metal API passes 100% | No Placeholder involved - direct Metal API use is thread-safe |
| MPSEncodingLock helps (19x reduction) | Protects executeMPSGraph path but not Placeholder path |
| Residual 4.4% failure with lock | Placeholder path is not protected by MPSEncodingLock |
| Thread-local MPSGraphCache is safe | Cache itself is thread-local, but Placeholder triggers operations during lookup miss |

## Thread-Safety Architecture Analysis

### Protected Layers

| Component | Protection Mechanism | Race-Free |
|-----------|---------------------|-----------|
| MPSGraphCache | Thread-local (per-thread instance) | Yes |
| MPSKernelCache | Thread-local (per-thread instance) | Yes |
| executeMPSGraph | MPSEncodingLock + stream mutex | Yes |
| Buffer allocation | Pool mutex | Yes |
| MPSNDArrayIdentity | s_ndarray_identity_mutex | Yes |

### Unprotected Layers

| Component | Issue | Race Location |
|-----------|-------|---------------|
| Placeholder constructor | gatherViewTensor() | OperationUtils.mm:562 |
| Placeholder constructor | src.clone() | OperationUtils.mm:569 |
| gatherViewTensor | at::empty() + commandEncoder() | View.mm:87,98 |

## Proposed Fix

The fix would involve extending `MPSEncodingLock` protection to the Placeholder constructor:

```cpp
// Option A: Acquire lock in Placeholder constructor
Placeholder::Placeholder(MPSGraphTensor* mpsGraphTensor,
                         const Tensor& src, ...) {
    // Acquire encoding lock if we might trigger Metal operations
    MPSEncodingLock encodingLock;  // ← ADD THIS

    if ((!src.is_contiguous() || src.storage_offset()) && gatherTensorData) {
        _tensor = gatherViewTensor(src, emptyShell);
        // ...
    }
}
```

```cpp
// Option B: Acquire lock in gatherViewTensor
Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst) {
    MPSEncodingLock encodingLock;  // ← ADD THIS

    Tensor output = at::empty(...);
    // ...
}
```

### Trade-offs

| Option | Pros | Cons |
|--------|------|------|
| A (Placeholder) | Protects all paths | Locks even when not needed |
| B (gatherViewTensor) | Minimal lock scope | May miss other paths |

## Why BatchQueue Works

Our BatchQueue workaround (num_workers=1) is effective because:

1. **Single worker serializes ALL MPS operations** including Placeholder construction
2. **No concurrent gatherViewTensor calls** possible with single worker
3. **Effectively extends serialization** to cover the unprotected Placeholder path

This is why we achieve 10/10 correctness with batched inference at 8 user threads.

## Verification Status

| Suite | Result |
|-------|--------|
| TSA | 0 warnings (4 files) |
| Batch inference | 10/10 tests pass (8 threads, 1 worker) |
| Parallel correctness (batched) | 10/10 (100%) |

## Conclusions

1. **Root cause identified**: Race in `Placeholder::Placeholder()` during non-contiguous tensor handling
2. **Specific functions**: `gatherViewTensor()` and `src.clone()` trigger unprotected Metal operations
3. **MPSEncodingLock gap**: Protects graph execution but not pre-execution tensor binding
4. **BatchQueue workaround**: Correctly addresses the issue by serializing all paths
5. **Potential upstream fix**: Extend MPSEncodingLock protection to Placeholder constructor

## Files Analyzed

- `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm` - Placeholder constructor
- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/View.mm` - gatherViewTensor
- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Copy.mm` - Copy operations
- `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm` - MPSEncodingLock definition
- `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.h` - MPSGraphCache (thread-local)

## Next Steps (Optional)

1. [ ] Create isolated reproduction that triggers specifically Placeholder race
2. [ ] Test fix by adding MPSEncodingLock to gatherViewTensor
3. [ ] Measure performance impact of extended locking
4. [ ] Consider filing PyTorch issue with this specific localization
