# Appendix D: Disassembly and Reverse Engineering

This appendix contains the disassembly analysis of Apple's AGXMetalG16X driver.

---

## D.1 Driver Information

| Field | Value |
|-------|-------|
| Binary | `/System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X` |
| Version | 329.2 |
| Platform | macOS 15.7.3 (24G419) |
| Hardware | Mac16,5 (M4 Max) |
| UUID | 0ce453ff-9f17-321f-bc18-383573a48221 |

---

## D.2 Crash Site 1: useResourceCommon

### Symbol
```
AGX::ContextCommon<...>::useResourceCommon(IOGPUMetalResource*, eAGXAllocationListResourceFlags, ...)
```

### Location
```
Function start: 0x26430c
Crash offset:   0x264370 (+100 bytes)
```

### Disassembly

```asm
; Function prologue - save registers and set up frame
000000000026430c    pacibsp                          ; Pointer authentication
0000000000264310    stp    x26, x25, [sp, #-0x50]!  ; Save callee-saved regs
0000000000264314    stp    x24, x23, [sp, #0x10]
0000000000264318    stp    x22, x21, [sp, #0x20]
000000000026431c    stp    x20, x19, [sp, #0x30]
0000000000264320    stp    x29, x30, [sp, #0x40]    ; Save frame pointer & LR
0000000000264324    add    x29, sp, #0x40           ; Set up frame pointer

; Save arguments to callee-saved registers
0000000000264328    mov    x22, x3                  ; arg3 -> x22
000000000026432c    mov    x19, x2                  ; arg2 (flags) -> x19
0000000000264330    mov    x21, x1                  ; arg1 (resource) -> x21
0000000000264334    mov    x20, x0                  ; self (context) -> x20

; ... (intermediate code) ...

; THE CRASH SITE - accessing context->mtlResourceList
0000000000264370    ldr    x0, [x20, #0x5c8]        ; Load MTLResourceList*
;                   ^^^ CRASH: x20 = 0x0 (NULL), offset 0x5c8
;                       Results in fault at address 0x5c8

; If load succeeded, would continue:
0000000000264374    cbz    x0, 0x264380             ; Skip if list is NULL
0000000000264378    mov    x1, x24                  ; resource to add
000000000026437c    bl     _MTLResourceListAddResource

; Load IOGPUResourceList
0000000000264380    ldr    x0, [x20, #0x638]        ; Load ResourceGroupUsage
0000000000264384    mov    x1, x24
0000000000264388    mov    x2, x22
000000000026438c    bl     setAndBindResource

; Load third resource list
0000000000264398    ldr    x0, [x20, #0x5d8]        ; Load IOGPUResourceList
; ...
00000000002643a4    bl     _IOGPUResourceListAddResource
```

### Analysis

The function expects `self` (x0/x20) to be a valid `ContextCommon*`. When another thread frees the context while this thread is still encoding, x20 becomes NULL, causing the crash at offset 0x5c8.

---

## D.3 ContextCommon Structure (Inferred)

Based on the disassembly, the `ContextCommon` class has the following layout:

```cpp
class ContextCommon {
    // ... earlier fields (0x000 - 0x5c7) ...

    // Resource tracking lists
    MTLResourceList*      mtlResourceList;       // offset 0x5c8
    IOGPUResourceList*    ioResourceList;        // offset 0x5d8
    // ... gap ...
    ResourceGroupUsage*   resourceGroupUsage;    // offset 0x638

    // ... additional fields ...
};
```

### Known Offsets

| Offset | Type | Purpose | Evidence |
|--------|------|---------|----------|
| 0x98 | Unknown | Crash in prepareForEnqueue | Crash report #3 |
| 0x184 | Unknown | Crash in allocateUSCSpillBuffer | Crash report #2 |
| 0x5c8 | MTLResourceList* | Metal resource tracking | Disassembly + Crash #1 |
| 0x5d8 | IOGPUResourceList* | IOGPU resource tracking | Disassembly |
| 0x638 | ResourceGroupUsage* | Resource group binding | Disassembly |

---

## D.4 Crash Site 2: allocateUSCSpillBuffer

### Symbol
```
AGX::SpillInfoGen3<...>::allocateUSCSpillBuffer(AGXSpillDesc*, ...)
```

### Location
```
Function start: 0x6b6f58
Crash offset:   +192 (0x184 fault address)
```

### Analysis

This crash occurs during shader register spill buffer allocation. The WRITE fault (esr 0x92000046) indicates the driver was attempting to store data when it crashed:

1. An allocation was being tracked
2. The tracking structure pointer was NULL
3. The driver attempted to write allocation metadata at offset 0x184

---

## D.5 Crash Site 3: prepareForEnqueue

### Symbol
```
AGX::ComputeContext<...>::prepareForEnqueue(bool)
```

### Location
```
Function address (arm64e): 0x2b61d0
Function address (x86_64): 0x3ceb70
Fault offset: 0x98
```

### Analysis

This is the kernel enqueue preparation function. The 0x98 offset suggests access to an early field in the context, possibly:
- Command buffer state
- Encoder state
- Pipeline configuration

---

## D.6 Key Symbols

| Symbol | Address | Purpose |
|--------|---------|---------|
| `useResourceCommon` | 0x26430c (arm64e) / 0x3642f0 (x86_64) | Resource list management |
| `setComputePipelineState:` | 0x2bd2b0 (arm64e) / 0x3c61b0 (x86_64) | Pipeline state setup |
| `prepareForEnqueue` | 0x2b61d0 (arm64e) / 0x3ceb70 (x86_64) | Kernel dispatch preparation |
| `allocateUSCSpillBuffer` | 0x6b6f58 (arm64e) / 0x80d7d0 (x86_64) | Shader spill allocation |
| `dispatchThreads:` | Various | Kernel dispatch entry point |

---

## D.7 Race Condition Root Cause

### The Pattern

All crashes share:
1. A context object expected to be valid
2. Context accessed via a NULL pointer
3. Crash at a field offset from NULL

### The Bug

Apple's AGX driver has insufficient synchronization between:
1. **Context lifecycle** (creation/destruction)
2. **Context usage** (encoding, resource tracking, dispatch)

### Race Window

```
Thread A                          Thread B
--------                          --------
1. Create context
2. Start encoding
3. ...                           3a. Destroy context (no check!)
4. Use context (CRASH)              (context now invalid)
```

Thread B can invalidate Thread A's context while Thread A is still using it.

---

## D.8 Inferred AGX Architecture

```
+-------------------------------------------------------------+
|                    AGX Metal Driver                          |
+-------------------------------------------------------------+
|  ComputeContext (per command encoder)                        |
|  +-- MTLResourceList (offset 0x5c8)                         |
|  +-- IOGPUResourceList (offset 0x5d8)                       |
|  +-- ResourceGroupUsage (offset 0x638)                      |
|  +-- SpillInfo (tracks shader register spills)              |
+-------------------------------------------------------------+
|  Shared State (NEEDS MUTEX?)                                 |
|  +-- Pipeline state cache                                    |
|  +-- Spill buffer pool                                       |
|  +-- Context registry                                        |
+-------------------------------------------------------------+
```

The driver appears to assume contexts are thread-local, but when multiple threads create and use contexts on different command queues, shared backing state races.

---

## D.9 Extraction Commands

```bash
# Extract symbols
nm -p /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X \
  | grep -E "(useResource|setComputePipeline|prepareFor|allocateUSC)"

# Disassemble specific function
otool -tv /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X \
  | sed -n '/useResourceCommon/,/^_/p' | head -100

# Get driver version
otool -L /System/Library/Extensions/AGXMetalG16X.bundle/Contents/MacOS/AGXMetalG16X
```
