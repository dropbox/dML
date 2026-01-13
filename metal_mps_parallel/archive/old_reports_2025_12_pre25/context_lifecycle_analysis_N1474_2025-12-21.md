# Context Lifecycle Analysis (ComputeContext / ContextCommon)

**Worker**: N=1474  
**Date**: 2025-12-21  
**Classification**: Phase 2 - Deep Reverse Engineering (Task 2.2)  
**Driver**: AGXMetalG16X 329.2 (macOS 15.7.3)  

---

## Executive Summary

Static analysis of AGXMetalG16X shows that `AGXG16XFamilyComputeContext` (the driver’s Objective‑C compute encoder) owns a single pointer ivar, `self->_impl`, which points at the underlying C++ compute context (`AGX::ComputeContext<HAL200...>`). That `_impl` is the practical “ContextCommon root” used by encoding paths.

**Key lifecycle finding**: `-[AGXG16XFamilyComputeContext deferredEndEncoding]` **tail-calls `destroyImpl`**, which **sets `self->_impl = NULL`** and returns the `_impl` backing allocation to an internal pool (or frees it). There is no visible synchronization protecting concurrent reads of `self->_impl` in encoding methods (e.g., `setComputePipelineState:`).

This explains the canonical NULL‑base faults:
- `NULL + 0x5c8` during resource tracking (`mtlResourceList` access)
- `NULL + 0x98` in `prepareForEnqueue`
- `NULL + 0x184` in `allocateUSCSpillBuffer`

All are consistent with a context pointer becoming invalid (cleared/recycled) while another thread is still executing an encoding/flush path that expects it to be valid.

---

## 1. Anchoring Symbols and IVARs

### 1.1 `AGXG16XFamilyComputeContext._impl`

The ivar offset for `AGXG16XFamilyComputeContext._impl` is stored at:
- `_OBJC_IVAR_$_AGXG16XFamilyComputeContext._impl` (arm64e): `0x0000000000758300`

Multiple methods load the ivar offset like:
```asm
adrp   x8, 1179         ; 0x758000
ldrsw  x21, [x8, #0x300] ; -> ivar offset for _impl (points at 0x758300)
ldr    x8,  [x0, x21]    ; x8 = self->_impl
```

### 1.2 `AGXG16XFamilyCommandBuffer._previousComputeCommandEncoder`

The command buffer caches a previous compute encoder:
- `_OBJC_IVAR_$_AGXG16XFamilyCommandBuffer._previousComputeCommandEncoder` (arm64e): `0x0000000000758098`

This is the “recycling” link between command buffers and compute contexts.

---

## 2. Lifecycle Walkthrough (arm64e)

### 2.1 Creation: CommandBuffer → ComputeContext (init)

Entry:
- `-[AGXG16XFamilyCommandBuffer computeCommandEncoderWithConfig:]` @ `0x24d694`

Behavior:
1. Attempts reuse via `tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:`.
2. If coalescing fails, calls `commitEncoder`, then allocates a new `AGXG16XFamilyComputeContext` and initializes it via `initWithCommandBuffer:config:`.

Key snippet (coalescing decision):
```asm
bl  "_objc_msgSend$tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:"
cbz w0, <allocate_new_encoder>
...
ldr x0, [commandBuffer, _previousComputeCommandEncoder]
bl "_objc_msgSend$reInitWithCommandBuffer:"
```

Initialization:
- `-[AGXG16XFamilyComputeContext initWithCommandBuffer:config:]` @ `0x2be1f4`

Key behaviors observed:
- `_impl` is obtained from an internal device pool guarded by `os_unfair_lock` **or** allocated via `calloc`.
- If pulled from pool, it is aggressively zeroed:
  - `bzero(self->_impl, pool_allocation_size)`
- The pointer is then stored into `self->_impl`.
- Resource‑tracking fields are initialized, including the canonical crash offset:
  - Writes to `_impl + 0x5c8` (MTLResourceList)
  - Writes to `_impl + 0x5d8` (IOGPUResourceList)
  - Writes to `_impl + 0x638` (ResourceGroupUsage)

This establishes:
1. `_impl` has a pool-backed lifetime separate from the ObjC wrapper object.
2. Reuse involves `bzero`, making stale references catastrophically unsafe.

### 2.2 Use: setComputePipelineState: dereferences `_impl + 0x5c8`

Primary anchor for crash site #1:
- `-[AGXG16XFamilyComputeContext setComputePipelineState:]` @ `0x2bd2b0`

Core sequence:
```asm
ldrsw  x8,  [0x758000 + 0x300]  ; _impl ivar offset
ldr    x20, [x0, x8]            ; x20 = self->_impl
ldr    x0,  [x20, #0x5c8]       ; x0 = _impl->mtlResourceList
cbz    x0,  <skip_add>
bl     _MTLResourceListAddResource
...
b      AGX::ComputeContext<HAL200...>::setPipelineCommon(ComputePipeline*)
```

If `self->_impl` is cleared concurrently, the `ldr x0, [x20, #0x5c8]` becomes the `NULL + 0x5c8` fault.

### 2.3 End-of-life: deferredEndEncoding → destroyImpl → `_impl = NULL` + pool return

The invalidation path is explicit and frequent.

#### 2.3.1 `deferredEndEncoding` ends the pass, then destroys `_impl`

- `-[AGXG16XFamilyComputeContext deferredEndEncoding]` @ `0x2bd8dc`

Key behavior:
1. If `_impl` exists, calls C++ `endComputePass(...)`.
2. Tail-calls `destroyImpl`.

```asm
ldrsw  x21, [0x758000 + 0x300]  ; _impl ivar offset
ldr    x8,  [self, x21]         ; self->_impl
cbz    x8,  <return>
...
bl  AGX::ComputeContext<HAL200...>::endComputePass(...)
...
b   _objc_msgSend$destroyImpl   ; tail-call destroyImpl
```

This means **`deferredEndEncoding` is a hard “context teardown” operation**, not merely a flush.

#### 2.3.2 `destroyImpl` returns backing memory and clears `self->_impl`

- `-[AGXG16XFamilyComputeContext destroyImpl]` @ `0x2bdd1c`

End-of-function is the key invalidation write:
```asm
str xzr, [x19, x24]  ; self->_impl = NULL
ret
```

Before that store:
- It pushes the `_impl` allocation back into a fixed-size freelist guarded by `os_unfair_lock`, or `free()`s it when the freelist is full.

So there are two closely-related invalidation hazards:
1. `self->_impl` cleared to NULL (crash-as-NULL when a racing thread reloads the pointer).
2. The allocation returned to pool and later `bzero`’d on reuse (crash-as-zeroed-substructure when a racing thread holds a stale pointer).

### 2.4 CommandBuffer invalidation triggers

#### 2.4.1 `commitEncoder` ends and releases previous encoders

- `-[AGXG16XFamilyCommandBuffer commitEncoder]` @ `0x24cc34`

For `_previousComputeCommandEncoder`:
1. Calls `deferredEndEncoding`
2. Releases the ObjC object
3. Clears the ivar to NULL

```asm
ldr x0, [self, _previousComputeCommandEncoder]
bl  _objc_msgSend$deferredEndEncoding
bl  _objc_release
str xzr, [self, _previousComputeCommandEncoder]
```

#### 2.4.2 CommandBuffer `dealloc` releases previous compute encoder

- `-[AGXG16XFamilyCommandBuffer dealloc]` @ `0x250440`

It releases `_previousComputeCommandEncoder` (and other cached encoders), again making “who else is still using it?” a correctness requirement.

---

## 3. Call Graph (Relevant Slice)

```
MTLCommandBuffer (client)
  |
  v
-[AGXG16XFamilyCommandBuffer computeCommandEncoderWithConfig:]
  |-> tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:
  |     (reads prevEncoder->_impl; modifies internal state)
  |-> if reuse:
  |     prevEncoder reInitWithCommandBuffer: self
  |     return prevEncoder
  `-> else:
        -[AGXG16XFamilyCommandBuffer commitEncoder]
           -> prevComputeEncoder deferredEndEncoding
                -> AGX::ComputeContext<HAL200>::endComputePass(...)
                -> destroyImpl (self->_impl = NULL; return/free allocation)
           -> release prevComputeEncoder
        alloc/init new encoder (initWithCommandBuffer:config:)

MTLComputeCommandEncoder (client)
  |
  v
-[AGXG16XFamilyComputeContext setComputePipelineState:]
  -> load self->_impl
  -> deref _impl->mtlResourceList @ +0x5c8
  -> AGX::ComputeContext<HAL200>::setPipelineCommon(...)
```

---

## 4. Exact Race Window (Static)

The minimal race that explains the observed crash signatures:

```
Thread A (encoding)                               Thread B (cleanup)
-------------------                               -----------------
1) x20 = self->_impl
2) ldr x0, [x20, #0x5c8]  <-- expects valid

                                                  1) commitEncoder / dealloc
                                                  2) prevEncoder deferredEndEncoding
                                                     -> destroyImpl
                                                     -> self->_impl = NULL
                                                     -> return allocation to pool
```

If Thread A reloads `self->_impl` after Thread B clears it, it becomes NULL and immediately faults at `NULL + 0x5c8` (or other offsets depending on which method races).

If Thread A keeps a stale `_impl` pointer while Thread B returns the allocation to the pool, a subsequent init path can `bzero()` it, producing “NULL-like” internal pointers and additional crash sites (`+0x98`, `+0x184`) during later flush/enqueue operations.

---

## 5. Conclusions

1. `AGXG16XFamilyComputeContext._impl` is the lifecycle-critical pointer and is cleared in `destroyImpl` without visible synchronization.
2. `deferredEndEncoding` is a hard teardown that tail-calls `destroyImpl`; it is invoked by `commitEncoder` and during command buffer teardown.
3. The driver uses `os_unfair_lock` to guard freelists and some configuration structures, but there is no comparable protection around the runtime encoding state pointer (`_impl`) or command-buffer cached encoder pointers under concurrent access.

---

## References

- `reports/main/agx_symbol_analysis_N1461_2025-12-20.md` (multi-slice addresses for crash symbols)
- `reports/main/context_common_structure_N1473_2025-12-21.md` (ContextCommon field mapping and locking findings)
- `papers/appendix/appendix_d_disassembly.md` (disassembly appendix)

