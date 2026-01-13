# Apple Feedback Report: MPS Thread-Safety Issues

**Feedback ID**: FB123456 (placeholder - submit at https://feedbackassistant.apple.com)
**Date**: 2024-12-14
**Product**: macOS / Metal Performance Shaders
**Classification**: Bug / Crash
**Reproducibility**: Always (3+ threads)

---

## Summary

`MPSNDArrayMatrixMultiplication` and certain Metal compute kernels crash when encoded concurrently from 3+ threads, even when each thread uses its own kernel instance and separate `MTLCommandQueue`. This prevents thread-safe parallel inference in machine learning frameworks like PyTorch.

---

## Environment

- **macOS Version**: 14.x / 15.x (Sonoma / Sequoia)
- **Hardware**: Apple Silicon (M1/M2/M3/M4)
- **Xcode Version**: 15.x / 16.x
- **Framework**: MetalPerformanceShaders.framework

---

## Issue 1: MPSNDArrayMatrixMultiplication Internal Shared State

### Description

`MPSNDArrayMatrixMultiplication` has internal mutable state that is not thread-safe. Concurrent encoding from 3+ threads causes crashes in `MPSSetResourcesOnCommandEncoder`, even when:

1. Each thread creates its own `MPSNDArrayMatrixMultiplication` instance
2. Each thread uses its own `MTLCommandQueue`
3. Encoding is serialized via mutex (still crashes)

### Steps to Reproduce

```objc
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <dispatch/dispatch.h>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    // Create separate command queues per thread
    dispatch_queue_t queues[4];
    id<MTLCommandQueue> cmdQueues[4];
    for (int i = 0; i < 4; i++) {
        cmdQueues[i] = [device newCommandQueue];
        queues[i] = dispatch_queue_create(NULL, DISPATCH_QUEUE_SERIAL);
    }

    // Matrix dimensions
    NSUInteger M = 512, N = 512, K = 512;

    // Create input/output buffers (shared is fine, issue is in kernel state)
    id<MTLBuffer> bufA = [device newBufferWithLength:M*K*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithLength:K*N*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:M*N*sizeof(float) options:MTLResourceStorageModeShared];

    dispatch_group_t group = dispatch_group_create();

    // Launch 4 concurrent threads - CRASHES
    for (int t = 0; t < 4; t++) {
        dispatch_group_async(group, queues[t], ^{
            // Each thread creates its own kernel instance
            MPSNDArrayMatrixMultiplication *matmul = [[MPSNDArrayMatrixMultiplication alloc]
                initWithDevice:device
                sourceCount:2];

            for (int iter = 0; iter < 100; iter++) {
                @autoreleasepool {
                    id<MTLCommandBuffer> cmdBuf = [cmdQueues[t] commandBuffer];

                    // Create descriptors for this iteration
                    MPSNDArrayDescriptor *descA = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:@[@(M), @(K)]];
                    MPSNDArrayDescriptor *descB = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:@[@(K), @(N)]];
                    MPSNDArrayDescriptor *descC = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:@[@(M), @(N)]];

                    MPSNDArray *arrA = [[MPSNDArray alloc] initWithBuffer:bufA offset:0 descriptor:descA];
                    MPSNDArray *arrB = [[MPSNDArray alloc] initWithBuffer:bufB offset:0 descriptor:descB];
                    MPSNDArray *arrC = [[MPSNDArray alloc] initWithBuffer:bufC offset:0 descriptor:descC];

                    // CRASH OCCURS HERE at 3+ threads
                    [matmul encodeToCommandBuffer:cmdBuf
                                     sourceArrays:@[arrA, arrB]
                                 destinationArray:arrC];

                    [cmdBuf commit];
                    [cmdBuf waitUntilCompleted];
                }
            }
        });
    }

    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    return 0;
}
```

### Expected Behavior

Each thread should be able to encode matrix multiplications concurrently using its own kernel instance and command queue without crashes.

### Actual Behavior

Crash in `MPSSetResourcesOnCommandEncoder` or similar internal MPS function when 3+ threads encode simultaneously.

**Crash signature**:
```
Thread 3 Crashed:
0   MetalPerformanceShaders    MPSSetResourcesOnCommandEncoder + 0x...
1   MetalPerformanceShaders    -[MPSNDArrayMatrixMultiplication encodeToCommandBuffer:...] + 0x...
```

### Workaround

Use `MPSGraph` instead of `MPSNDArrayMatrixMultiplication`. The graph-based path appears to be thread-safe:

```objc
MPSGraph *graph = [[MPSGraph alloc] init];
MPSGraphTensor *a = [graph placeholderWithShape:@[@(M), @(K)] dataType:MPSDataTypeFloat32 name:@"a"];
MPSGraphTensor *b = [graph placeholderWithShape:@[@(K), @(N)] dataType:MPSDataTypeFloat32 name:@"b"];
MPSGraphTensor *c = [graph matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:@"c"];
// Graph execution IS thread-safe with separate graph instances per thread
```

---

## Issue 2: Metal Compute Kernels Crash at 4+ Threads

### Description

Custom Metal compute kernels (used for operations like LayerNorm) crash intermittently when dispatched from 4+ concurrent threads, even with separate command queues and pipeline states.

### Steps to Reproduce

```objc
// Simplified - actual issue occurs in complex compute shader dispatch
// Each thread:
// 1. Gets its own MTLComputePipelineState (from shared MTLLibrary)
// 2. Uses its own MTLCommandQueue
// 3. Encodes compute commands

// At 4+ threads, intermittent crashes occur in Metal's internal state
```

### Observed Failure Rate (before mitigation)

| Threads | Success Rate |
|---------|--------------|
| 2       | 100%         |
| 4       | ~60%         |
| 6       | ~40%         |
| 8       | ~60%         |

### Workaround

Serialize compute kernel encoding with a process-wide mutex. This sacrifices parallelism for correctness.

---

## Impact

These issues prevent machine learning frameworks from achieving parallel inference on Apple Silicon. PyTorch, which uses MPS for GPU acceleration on Mac, cannot safely run multiple inference threads without crashes or serialization.

**Affected use cases**:
- Web servers handling concurrent ML inference requests
- Batch processing with thread pools
- Real-time applications requiring parallel model execution

---

## Suggested Fix

1. **MPSNDArrayMatrixMultiplication**: Ensure internal state is either:
   - Thread-local (per-instance)
   - Protected by internal synchronization
   - Documented as not thread-safe (so callers can serialize)

2. **Metal Compute Dispatch**: Review internal state management in `MTLComputeCommandEncoder` or related paths for thread-safety when multiple command queues dispatch concurrently.

3. **Documentation**: If concurrent encoding is not supported, document this clearly in the MPS and Metal headers.

---

## Related Projects

This issue was discovered while implementing parallel MPS inference for PyTorch:
- Project: https://github.com/dropbox/dML/metal_mps_parallel
- PyTorch issue: (link if filed)

---

## Attachments

- Minimal reproduction code (above)
- Crash logs (can be provided on request)
- PyTorch MPS parallel inference patch showing workarounds

---

## Contact

(Your contact information for Apple to follow up)
