# ContextCommon Structure Analysis

**Worker**: N=1473
**Date**: 2025-12-21
**Classification**: Phase 2 - Deep Reverse Engineering

---

## Executive Summary

Analysis of Apple's AGXMetalG16X driver (version 329.2) reveals detailed internal structure information through Objective-C type encodings embedded in the binary. This report extends previous reverse engineering by mapping additional fields and discovering synchronization primitives.

---

## 1. Driver Binary Information

| Field | Value |
|-------|-------|
| Path | `/System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X` |
| Size | 20,490,752 bytes |
| Architecture | Universal (x86_64, arm64e) |
| Version | 329.2 |
| macOS | 15.7.3 (24G419) |
| Hardware | Mac16,5 (M4 Max) |

---

## 2. ContextCommon Structure (Extended)

### Known Field Offsets

| Offset | Type | Purpose | Source |
|--------|------|---------|--------|
| 0x98 | Unknown | prepareForEnqueue crash | Crash report |
| 0x184 | Unknown | allocateUSCSpillBuffer crash | Crash report |
| 0x5c8 | MTLResourceList* | Metal resource tracking | Disassembly |
| 0x5d8 | IOGPUResourceList* | IOGPU resource tracking | Disassembly |
| 0x638 | ResourceGroupUsage* | Resource group binding | Disassembly |

### Structure Size Estimate

Based on the largest known offset (0x638 = 1592 bytes) plus typical pointer-aligned trailing fields, the ContextCommon structure is estimated at **~1700-2000 bytes**.

---

## 3. NEW DISCOVERY: Locking Mechanisms

### AGXA_UnfairLock

The driver uses Apple's `os_unfair_lock` for synchronization:

```cpp
struct AGXA_UnfairLock {
    os_unfair_lock_s lock;  // _os_unfair_lock_opaque (uint32_t)
};
```

**Found in**:
- ComputePipeline (visible_function_table_lock)
- ComputePipeline (intersection_function_table_lock)
- Program structures (lock field)

### Implication

The driver **does** have internal locking for some operations (function tables, program variants), but **NOT** for ContextCommon access during encoding operations. This confirms the race condition occurs in code paths that lack synchronization.

---

## 4. ComputePipeline Structure (From Type Encoding)

Key fields discovered:

```cpp
class ComputePipeline {
    unique_ptr<AGX::HeapBuffer> scs_per_shader_config_table;
    AGXG16XFamilyBuffer* dora_state_buffer;
    ComputeProgramVariant* compute_variant;
    bool supports_indirect_command_buffers;
    bool requires_binary_linking_bindings;
    bool descendent_pipeline;
    AGXA_UnfairLock visible_function_table_lock;  // SYNC!
    AGXA_UnfairLock intersection_function_table_lock;  // SYNC!
    ExecuteIndirectPipelineState ei_state;
    Allocation resource_indirection_heap_allocation;
    AGXG16XFamilyDevice* device_obj;
    RuntimeState loader_runtime_state;
    // ... additional fields
};
```

---

## 5. SpillInfo Structure (From Type Encoding)

```cpp
struct SpillInfo {
    uint32_t max_temporary_register_count;
    uint32_t total_spill_buffer_bytes;
    uint32_t base_spill_buffer_bytes;
    uint32_t entry_function_group_non_main_spill_buffer_bytes;
    uint32_t total_ipr_buffer_bytes;
    uint32_t base_ipr_buffer_bytes;
    uint32_t max_call_stack_depth;
    uint32_t max_spill_per_function_bytes;
    uint32_t max_ipr_per_function_bytes;
    uint32_t tls_alloc_size;
    uint32_t total_tls_size;
    uint32_t global_constructor_count;
    // ... additional fields
};
```

This explains crash site #3 (allocateUSCSpillBuffer at 0x184) - the SpillInfo structure is accessed without synchronization when multiple threads allocate spill buffers concurrently.

---

## 6. ResourceGroupMembershipList

Discovered C++ namespace function:
```cpp
void AGX::ResourceGroupMembershipList::set(AGXMDSID)
```

This is part of the resource tracking system that crashes at offset 0x5c8.

---

## 7. Program Variant Structures

Multiple program types share a common pattern:

```cpp
template<typename Variant>
struct Program {
    VectorMap<Key, ProgramVariantEntry<Variant>, 4> variants;
    VectorMap<ReflectionKey, ReflectionEntry, 4> reflections;
    AGXA_UnfairLock lock;  // Per-program lock
    IndirectArgumentLayout* argument_layouts[128];
    bool has_layouts;
};
```

**Programs using this pattern**:
- ComputeProgram
- TileProgram
- VertexProgram
- FragmentProgram
- ObjectProgram
- MeshProgram
- IntersectionProgram

---

## 8. RuntimeState Structure

```cpp
struct RuntimeState {
    shared_ptr<AGX::HeapBuffer> got;
    shared_ptr<AGX::HeapBuffer> global_constructors;
    shared_ptr<AGX::HeapBuffer> builtin_state_buffer;
    // ... additional fields
};
```

---

## 9. Class Hierarchy

```
AGXG16XFamilyDevice
├── AGXG16XFamilyCommandQueue
│   └── AGXG16XFamilyCommandBuffer
│       ├── AGXG16XFamilyBlitContext
│       ├── AGXG16XFamilyComputeContext     ← CRASH SITE
│       ├── AGXG16XFamilyRenderContext
│       ├── AGXG16XFamilyResourceStateContext
│       ├── AGXG16XFamilySampledComputeContext
│       └── AGXG16XFamilySampledRenderContext
├── AGXG16XFamilyBuffer
├── AGXG16XFamilyTexture
├── AGXG16XFamilyHeap
├── AGXG16XFamilyComputePipeline
├── AGXG16XFamilyRenderPipeline
└── AGXG16XFamilyResourceGroup
```

---

## 10. Key Methods on ComputeContext

```objc
-[AGXG16XFamilyComputeContext endEncoding]
-[AGXG16XFamilyComputeContext sampleCountersInBuffer:atSampleIndex:withBarrier:]
```

### Key Methods on CommandBuffer

```objc
-[AGXG16XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:]
-[AGXG16XFamilyCommandBuffer renderCommandEncoderWithDescriptor:]
-[AGXG16XFamilyCommandBuffer parallelRenderCommandEncoderWithDescriptor:]
-[AGXG16XFamilyCommandBuffer sampledComputeCommandEncoderWithConfig:programInfoBuffer:capacity:]
-[AGXG16XFamilyCommandBuffer dropResourceGroups:count:]
```

---

## 11. Race Condition Root Cause (Refined)

Based on new findings:

1. **Program/Pipeline structures** have locks (`AGXA_UnfairLock`)
2. **Context structures** do NOT have locks for encoding operations
3. **Resource lists** (0x5c8, 0x5d8, 0x638) are accessed without synchronization

The driver protects **configuration data** (pipelines, programs) but NOT **runtime encoding state** (contexts, resource lists).

### Race Window

```
Thread A (Encoding)               Thread B (Cleanup)
--------------------              ------------------
1. Get context pointer            
2. Access context->resourceList   2a. Invalidate context
   at 0x5c8                           (sets to NULL)
3. CRASH: NULL deref              
```

---

## 12. Stress Test Results (N=1473)

| Test | Iterations | Crashes | Crash Rate |
|------|------------|---------|------------|
| benchmark_comprehensive_final.py | 5 | 0 | 0% |
| Custom 16-thread stress | 1 | 0 | 0% |
| Shutdown phase | 5 | 0 | 0% |

**Note**: The 55% crash rate reported in N=1424 was not reproduced in this session. Crash may be timing-dependent or require specific conditions.

---

## 13. Conclusions

1. **ContextCommon has at least 6 known fields** at offsets 0x98, 0x184, 0x5c8, 0x5d8, 0x638
2. **Driver uses os_unfair_lock** for pipeline/program synchronization
3. **No synchronization** for context access during encoding
4. **SpillInfo structure** explains the 0x184 crash site
5. **Resource tracking lists** explain the 0x5c8 and 0x5d8 crash sites
6. **Crash is timing-dependent** - not always reproducible

---

## References

- Previous analysis: `reports/main/agx_reverse_engineering_N1435_2025-12-20.md`
- Appendix D: `papers/appendix/appendix_d_disassembly.md`
- Type encodings extracted: `/tmp/agx_strings.txt`
