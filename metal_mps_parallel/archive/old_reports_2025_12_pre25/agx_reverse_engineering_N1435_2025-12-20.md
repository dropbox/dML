# AGX Driver Reverse Engineering Analysis

**Worker**: N=1435
**Date**: 2025-12-20
**Classification**: RESEARCH - Apple Bug Report Evidence

---

## Executive Summary

Reverse engineering of Apple's AGXMetalG16X driver (version 329.2) reveals that the crashes occur due to NULL pointer dereferences of context objects in resource management functions. The root cause appears to be a race condition where a `ComputeContext` object is either:
1. Freed by one thread while another thread is still using it
2. Never properly initialized due to concurrent access during setup

---

## Crash Site 1: useResourceCommon

### Symbol
```
AGX::ContextCommon<...>::useResourceCommon(IOGPUMetalResource*, eAGXAllocationListResourceFlags, ...)
```

### Location
```
Address: 0x26430c (function start)
Crash:   0x264370 (offset +100 from start)
```

### Faulting Instruction
```asm
0000000000264370    ldr    x0, [x20, #0x5c8]    ; Load resourceList from context
```

### Analysis

The function prologue:
```asm
000000000026430c    pacibsp                          ; Pointer auth
0000000000264310    stp    x26, x25, [sp, #-0x50]!  ; Save registers
0000000000264314    stp    x24, x23, [sp, #0x10]
0000000000264318    stp    x22, x21, [sp, #0x20]
000000000026431c    stp    x20, x19, [sp, #0x30]
0000000000264320    stp    x29, x30, [sp, #0x40]
0000000000264324    add    x29, sp, #0x40
0000000000264328    mov    x22, x3               ; arg3 -> x22
000000000026432c    mov    x19, x2               ; arg2 -> x19
0000000000264330    mov    x21, x1               ; arg1 (resource) -> x21
0000000000264334    mov    x20, x0               ; self (context) -> x20
```

At crash time, `x20 = 0x0` (NULL), meaning the `self` pointer (first argument) was NULL.

### ContextCommon Structure (Inferred)
```cpp
class ContextCommon {
    // ...
    void* mtlResourceList;       // offset 0x5c8 - MTLResourceList*
    void* ioResourceList;        // offset 0x5d8 - IOGPUResourceList*
    void* resourceGroupUsage;    // offset 0x638 - ResourceGroupUsage*
    // ...
};
```

### Race Condition Hypothesis

The context object is passed as `self` (x0) to this function. When x0 is NULL:

1. **Premature Deallocation**: Thread A frees the context while Thread B still holds a reference
2. **Use-After-Free**: Thread B calls `useResourceCommon` on the freed context
3. **NULL Dereference**: The memory is zeroed or reused, causing x0 = NULL

---

## Resource Management Flow

The crash occurs during resource tracking for GPU operations:

```asm
; After NULL check failure, the code would have:
0000000000264370    ldr    x0, [x20, #0x5c8]    ; Load MTLResourceList from context
0000000000264374    cbz    x0, 0x264380          ; Skip if list is NULL
0000000000264378    mov    x1, x24               ; resource to add
000000000026437c    bl     _MTLResourceListAddResource  ; Add to Metal list

0000000000264380    ldr    x0, [x20, #0x638]    ; Load ResourceGroupUsage
0000000000264384    mov    x1, x24
0000000000264388    mov    x2, x22
000000000026438c    bl     setAndBindResource    ; Bind resource

0000000000264398    ldr    x0, [x20, #0x5d8]    ; Load IOGPUResourceList
...
00000000002643a4    bl     _IOGPUResourceListAddResource
```

This shows the context object maintains THREE resource tracking lists:
- `MTLResourceList` at offset 0x5c8
- `ResourceGroupUsage` at offset 0x638
- `IOGPUResourceList` at offset 0x5d8

---

## Crash Site 2: allocateUSCSpillBuffer (0x184 offset)

### Symbol
```
AGX::SpillInfoGen3<...>::allocateUSCSpillBuffer(AGXSpillDesc*, ...)
```

### Analysis

This crash occurs during shader register spill buffer allocation. The fault address 0x184 indicates access to a different structure, likely `SpillInfo` or a related allocation descriptor.

The WRITE fault (esr 0x92000046) indicates the driver was attempting to STORE data when it crashed, not load. This suggests:
1. An allocation was being tracked
2. The tracking structure pointer was NULL
3. The driver attempted to write allocation metadata at offset 0x184

---

## Crash Site 3: prepareForEnqueue (0x98 offset)

### Symbol
```
AGX::ComputeContext<...>::prepareForEnqueue(bool)
```

### Location
```
Address: 0x3ceb70 (from nm)
```

### Analysis

This is the kernel enqueue preparation function. The 0x98 offset suggests access to an early field in the context, possibly:
- Command buffer state
- Encoder state
- Pipeline configuration

---

## Root Cause Analysis

### The Pattern

All three crashes share a common pattern:
1. A **context object** is expected to be valid
2. The context is accessed via a **pointer that is NULL**
3. The crash happens at a **field offset** from the NULL pointer

### The Bug

Apple's AGX driver has **insufficient synchronization** between:
1. **Context lifecycle management** (creation/destruction)
2. **Context usage** (encoding, resource tracking, kernel dispatch)

When multiple command encoders operate concurrently:
- Thread A may destroy/invalidate a context
- Thread B still has a reference to that context
- Thread B accesses the (now-invalid) context and crashes

### Evidence

1. **Multiple crash sites**: Three different functions, three different offsets
2. **Same NULL pattern**: x20 (context pointer) is consistently 0x0
3. **Resource management context**: Crashes during resource list operations
4. **Concurrency required**: Only crashes under multi-threaded load

---

## Inferred AGX Internal Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AGX Metal Driver                          │
├─────────────────────────────────────────────────────────────┤
│  ComputeContext (per command encoder)                        │
│  ├── MTLResourceList (offset 0x5c8)                         │
│  ├── IOGPUResourceList (offset 0x5d8)                       │
│  ├── ResourceGroupUsage (offset 0x638)                      │
│  └── SpillInfo (tracks shader register spills)              │
├─────────────────────────────────────────────────────────────┤
│  Shared State (NEEDS MUTEX?)                                 │
│  ├── Pipeline state cache                                    │
│  ├── Spill buffer pool                                       │
│  └── Context registry                                        │
└─────────────────────────────────────────────────────────────┘
```

The driver appears to assume contexts are thread-local, but when multiple threads create and use contexts on different command queues, the shared backing state races.

---

## Recommendations for Apple Bug Report

### Title
```
AGX Metal Driver: Race condition in ComputeContext causes NULL pointer crash under concurrent encoding
```

### Severity
**Critical** - Causes application crash (SIGSEGV)

### Reproduction
```bash
# Multi-threaded Metal compute shader dispatch
# Crashes ~55% of attempts within first second
MPS_DISABLE_ENCODING_MUTEX=1 python3 tests/benchmark_comprehensive_final.py
```

### Impact
- Affects all applications using Metal compute shaders from multiple threads
- Requires application-level mutex to work around
- Limits parallel GPU utilization

### Suggested Fix
Add proper synchronization to:
1. `useResourceCommon` - check context validity before use
2. `allocateUSCSpillBuffer` - lock spill buffer allocation
3. `prepareForEnqueue` - synchronize context state access

Or mark ComputeContext as not thread-safe and document that encoders must be used from single thread.

---

## Technical Details

### Driver Version
- **Binary**: `/System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X`
- **Version**: 329.2
- **Platform**: macOS 15.7.3 (24G419)
- **Hardware**: Mac16,5 (M4 Max)

### Key Symbols
| Function | Address | Purpose |
|----------|---------|---------|
| `useResourceCommon` | 0x26430c | Resource list management |
| `setComputePipelineState:` | 0x3c61b0 | Pipeline state setup |
| `prepareForEnqueue` | 0x3ceb70 | Kernel dispatch preparation |
| `allocateUSCSpillBuffer` | 0x6b6f58 | Shader spill allocation |

### Structure Offsets (ComputeContext)
| Offset | Type | Purpose |
|--------|------|---------|
| 0x5c8 | MTLResourceList* | Metal resource tracking |
| 0x5d8 | IOGPUResourceList* | IOGPU resource tracking |
| 0x638 | ResourceGroupUsage* | Resource group binding |

---

## Conclusion

The AGX driver has race conditions in context lifecycle management. When multiple threads encode Metal compute commands concurrently, contexts can be invalidated while still in use, causing NULL pointer dereferences.

Our workaround (global encoding mutex) is correct and necessary until Apple fixes the driver.

---

## References

- Crash Analysis 1: `reports/crash_reports/CRASH_ANALYSIS_2025-12-20_173618.md`
- Crash Analysis 2: `reports/crash_reports/CRASH_ANALYSIS_2025-12-20_174241.md`
- TLA+ Verification: `reports/main/tla_verification_complete_N1435_2025-12-20.md`
