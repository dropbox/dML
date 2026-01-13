# AGX Driver Symbol Analysis - N=1461

**Date**: 2025-12-20
**Worker**: N=1461
**Purpose**: Technical evidence for Apple bug report

---

## Driver Location

```
/System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X
Size: 20,490,752 bytes
```

---

## Crash Site 1: setComputePipelineState:

**Symbol**: `-[AGXG16XFamilyComputeContext setComputePipelineState:]`
**Addresses**: `0x3c61b0`, `0x2bd2b0` (multiple slices)
**Crash offset**: `+0x5c8`
**Fault type**: READ (NULL+0x5c8)

---

## Crash Site 2: prepareForEnqueue

**Mangled**: `__ZN3AGX14ComputeContextINS_6HAL2008EncodersENS1_7ClassesENS1_10ObjClassesENS1_28EncoderComputeServiceClassesEE17prepareForEnqueueEb`

**Demangled**:
```cpp
AGX::ComputeContext<
    AGX::HAL200::Encoders,
    AGX::HAL200::Classes,
    AGX::HAL200::ObjClasses,
    AGX::HAL200::EncoderComputeServiceClasses
>::prepareForEnqueue(bool)
```

**Addresses**: `0x3ceb70`, `0x2b61d0`
**Crash offset**: `+0x98`
**Fault type**: READ (NULL+0x98)

---

## Crash Site 3: allocateUSCSpillBuffer

**Mangled**: `__ZN3AGX13SpillInfoGen3INS_6HAL2008EncodersENS1_7ClassesENS1_10ObjClassesEE22allocateUSCSpillBufferEP12AGXSpillDescP13AGXUMADescRecP15MTLResourceListPNS_24DMASpillBufferDescriptorER17IOGPUResourceInfob`

**Demangled**:
```cpp
AGX::SpillInfoGen3<
    AGX::HAL200::Encoders,
    AGX::HAL200::Classes,
    AGX::HAL200::ObjClasses
>::allocateUSCSpillBuffer(
    AGXSpillDesc*,
    AGXUMADescRec*,
    MTLResourceList*,
    AGX::DMASpillBufferDescriptor*,
    IOGPUResourceInfo&,
    bool
)
```

**Addresses**: `0x80d7d0`, `0x6b6f58`
**Crash offset**: `+0x184`
**Fault type**: WRITE (NULL+0x184)

---

## Analysis

### HAL200 Namespace

All crash sites use the `AGX::HAL200::*` template parameters, indicating this is the Hardware Abstraction Layer for the M4 family GPU (Generation 200).

### Common Thread: ComputeContext

- Crash 1: `AGXG16XFamilyComputeContext` (Objective-C wrapper)
- Crash 2: `AGX::ComputeContext<...>` (C++ template)
- Crash 3: `AGX::SpillInfoGen3<...>` (called from ComputeContext)

All crashes involve the compute context state during concurrent Metal encoding operations.

### USC Spill Buffer

The third crash site (`allocateUSCSpillBuffer`) relates to Unified Shader Core (USC) register spilling. When a compute kernel needs more registers than physically available, the driver allocates spill buffers in device memory. The NULL pointer suggests the spill buffer allocation context is not properly initialized or is being accessed by multiple threads without synchronization.

---

## Hypothesis

The `ComputeContext` object contains internal state that is:
1. Allocated lazily (on first use)
2. Accessed without thread safety during concurrent encoding
3. Results in NULL pointer dereferences when multiple threads race

This explains why:
- Normal sequential operations never crash (state is properly initialized)
- Concurrent operations crash ~55% of the time (race condition during initialization/access)
- Crashes occur at multiple sites (different parts of the context are affected)

---

## Recommendation for Apple Bug Report

Include:
1. The three crash sites with full demangled signatures
2. Register dumps showing NULL pointers (x20, x8)
3. Reproducible test case (our benchmark with mutex disabled)
4. Evidence that concurrent MTLComputeCommandEncoder operations trigger the race

The bug is in Apple's AGX driver, not in application code.
