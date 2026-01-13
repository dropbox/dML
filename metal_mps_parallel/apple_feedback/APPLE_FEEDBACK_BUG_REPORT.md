# Apple Feedback: Metal Command Encoder Thread Safety Issue

**Created by Andrew Yates**

**Feedback Category**: Metal Framework (MetalPerformanceShaders / AGX Driver)
**Severity**: Major - Prevents multi-threaded ML inference on Apple Silicon
**Affects**: All Metal-based ML frameworks (PyTorch MPS, MLX, CoreML)
**macOS Version**: 15.7.3 (24G419)
**Hardware**: Apple M4 Max (tested), affects all Apple Silicon

---

## CORRECTION (2025-12-20): Issue is in MPS, Not Metal Driver

**Original claim**: The serialization bug is in Metal/AGX driver.

**Corrected finding**: Raw Metal API achieves **62% efficiency** at 8 threads when used correctly
(per-thread command buffers + shared queue). The serialization causing poor ML performance is in
**MetalPerformanceShaders (MPS)** or framework integration layers, NOT the Metal driver.

| Test Configuration | 8-Thread Efficiency |
|--------------------|---------------------|
| Raw Metal (shared queue, per-thread buffers) | **62%** |
| PyTorch MPS (event sync) | 34% |
| PyTorch MPS (device sync) | 14% |

**Evidence**: `fixes/metal_interpose/test_raw_metal.m` demonstrates that Metal API scales well
when used correctly. The "bare Metal" reproduction below shares a SINGLE command buffer across
threads - this is incorrect usage and not representative of the actual issue.

**Recommendation**: This Apple Feedback report may not be applicable. The serialization
affecting ML frameworks appears to be in MPS internal synchronization or framework code.

---

## Summary (Original - See Correction Above)

Metal's command encoder has a thread safety issue that causes assertion failures when multiple threads attempt GPU operations concurrently. This fundamentally limits the ability to perform parallel ML inference on Apple Silicon GPUs.

**Error Message**:
```
-[AGXG16XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:]:1091:
failed assertion 'A command encoder is already encoding to this command buffer'
```

This occurs in Apple's AGX driver (`AGXG16XFamilyCommandBuffer`), below both the MPS and MLX framework layers.

**Note**: This error only occurs with incorrect usage (sharing a single command buffer across threads).
Per-thread command buffers work correctly.

---

## Impact

1. **MLX crashes** with 2+ Python threads performing matrix operations
2. **PyTorch MPS** requires mutex serialization, limiting multi-thread efficiency to ~13% at 8 threads (vs single-op baseline)
3. **All Apple ML frameworks** are affected since they use Metal command buffers

This prevents developers from fully utilizing Apple Silicon's powerful GPU cores for parallel ML workloads.

---

## Reproduction Steps

### Minimal MLX Reproduction (crashes immediately)

```python
#!/usr/bin/env python3
"""
Minimal reproduction of Metal command encoder race condition
Requires: pip install mlx
Hardware: Any Apple Silicon Mac
"""
import mlx.core as mx
import threading

def worker(tid):
    for _ in range(20):
        a = mx.random.normal((1024, 1024))
        b = mx.random.normal((1024, 1024))
        c = a @ b
        mx.eval(c)

threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Complete")  # Never reached - crashes with assertion
```

**Result**: Crash with `'A command encoder is already encoding to this command buffer'`

### PyTorch MPS (works with mutex, limited efficiency)

```python
#!/usr/bin/env python3
"""
PyTorch MPS works but requires internal serialization
"""
import torch
import threading

device = torch.device('mps')

def worker(tid):
    for _ in range(20):
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = torch.mm(a, b)
    torch.mps.synchronize()

threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Complete")  # Works, but with serialized encoder access
```

### Bare Metal API Reproduction (No ML Framework)

This reproduction uses **only Metal APIs** (no MPS, no MLX, no PyTorch), proving the bug is in the Metal/AGX layer:

```objc
// Compile: clang -fobjc-arc -framework Foundation -framework Metal -o metal_thread_test metal_thread_test.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> sharedBuffer = [queue commandBuffer];

        dispatch_group_t group = dispatch_group_create();

        // 4 threads accessing SAME command buffer - will crash
        for (int t = 0; t < 4; t++) {
            dispatch_group_async(group, dispatch_get_global_queue(0, 0), ^{
                id<MTLComputeCommandEncoder> encoder = [sharedBuffer computeCommandEncoder];
                [NSThread sleepForTimeInterval:0.001];
                [encoder endEncoding];
            });
        }

        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        [sharedBuffer commit];
    }
    return 0;
}
```

**Result**: Immediate crash with `'A command encoder is already encoding to this command buffer'`

**Key Finding**: Per-thread command buffers work fine. The bug only manifests when multiple threads access the **same** command buffer. This is the root cause of all ML framework issues - they internally share command buffers for efficiency.

---

## Technical Analysis

### Root Cause Location

The crash occurs at:
```
-[AGXG16XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:]
```

This is Apple's AGX driver class, not in the ML framework layer. The race condition is:

1. Thread A calls `[commandBuffer computeCommandEncoder]`
2. Thread B calls `[commandBuffer computeCommandEncoder]` on same buffer
3. AGX tries to coalesce/reuse encoder state
4. Both threads get the same encoder reference
5. Assertion fails: "encoder is already encoding"

### Why This Is a Metal-Level Issue

- **Both MLX and PyTorch MPS hit the same error** - the issue is below the framework layer
- MLX uses Apple's own Steel kernels, yet still crashes
- The error originates in `AGXG16XFamilyCommandBuffer`, not `MPS*` classes
- The MLX team is implementing mutex locks (same workaround we did for PyTorch)

### MLX GitHub Issues Confirming This

- Issue #2133: "Thread safety: Ongoing issue" - StreamContext, cache, graph eval not thread-safe
- PR #2104: "Metal thread safety" - MLX team adding mutex locks
- Issue #2067: "[BUG] thread issues with evaluation"

---

## Efficiency Measurements

With mutex serialization (the only way to avoid crashes):

| Threads | MPS Efficiency | MLX (vanilla) |
|---------|---------------|---------------|
| 1 | 100% (baseline) | 100% |
| 2 | ~55% | CRASH |
| 4 | ~30% | CRASH |
| 8 | ~13% | CRASH |

The ~13% efficiency vs single-op baseline at 8 threads is due to the GPU command queue bottleneck. Threading plateaus at ~3,800 ops/s regardless of thread count. Use thread pools to avoid thread creation overhead, but batching (~373x more efficient at batch 256) is recommended for throughput.

---

## Suggested Fix Approaches

### Option 1: Per-Thread Command Encoders (Preferred)

Instead of reusing/coalescing encoders, create fresh encoders per invocation:

```objc
// Current (problematic):
- (id<MTLComputeCommandEncoder>)getEncoder:(id<MTLCommandBuffer>)cmdBuf {
    // May return cached/shared encoder
    return [cmdBuf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
}

// Fixed:
- (id<MTLComputeCommandEncoder>)getEncoder:(id<MTLCommandBuffer>)cmdBuf {
    // Always create fresh encoder, no sharing
    return [[cmdBuf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] retain];
}
```

### Option 2: Thread-Safe Encoder Pool

Implement a proper thread-local encoder cache in the AGX driver:

```objc
// In AGXG16XFamilyCommandBuffer:
static thread_local id<MTLComputeCommandEncoder> t_encoder;

- (id<MTLComputeCommandEncoder>)threadSafeEncoder {
    if (!t_encoder || t_encoder.commandBuffer != self) {
        t_encoder = [self computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    }
    return t_encoder;
}
```

### Option 3: Internal Locking (Last Resort)

Add internal mutex protection in the coalescing path:

```objc
- (BOOL)tryCoalescingPreviousComputeCommandEncoderWithConfig:...nextEncoderClass:... {
    @synchronized(self) {  // Add thread safety
        // Existing coalescing logic
    }
}
```

---

## Test Environment

- **Hardware**: Apple M4 Max, 40-core GPU, 128GB unified memory
- **macOS**: 15.7.3 (24G419)
- **MLX**: 0.30.0
- **PyTorch**: 2.9.1a0+git1038e7b (custom MPS patches for thread safety)

---

## Attachments

1. `metal_bare_thread_race.m` - **Bare Metal reproduction (no ML frameworks)** - Most important
2. `mlx_crash_reproduction.py` - Minimal MLX crash reproduction
3. `pytorch_mps_workaround.py` - PyTorch with mutex workaround
4. `efficiency_benchmark.py` - Throughput scaling measurements
5. `crash_log.txt` - Full crash stacktrace

---

## Conclusion

The Metal command encoder race condition is a significant limitation for ML workloads on Apple Silicon. With Apple's push toward on-device AI (Apple Intelligence, CoreML, MLX), fixing this would enable much better parallel inference performance.

Both PyTorch and MLX teams are working around this with application-level mutexes, but a proper fix in Metal would benefit the entire Apple ML ecosystem.

Thank you for your attention to this issue.
