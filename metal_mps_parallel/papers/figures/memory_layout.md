# AGX ContextCommon Memory Layout and Crash Sites

## Figure 7: ContextCommon Structure Layout (Inferred from Crashes)

```
                AGX::ContextCommon Object Layout
                ═══════════════════════════════════

    Offset    Size   Field                    Source
    ──────────────────────────────────────────────────
    0x000     8      vtable_ptr               (standard C++)
    0x008     ?      unknown fields...
    :         :      :
    :         :      [~1400 bytes unknown]
    :         :      :
    ┌─────────────────────────────────────────────────┐
    │ 0x098     8    ??? (CRASH SITE 2)              │◄── prepareForEnqueue
    │                NULL dereference                │    reads this field
    └─────────────────────────────────────────────────┘
    :         :      :
    :         :      [~236 bytes unknown]
    :         :      :
    ┌─────────────────────────────────────────────────┐
    │ 0x184     8    ??? (CRASH SITE 3)              │◄── allocateUSCSpillBuffer
    │                NULL dereference                │    writes to this field
    └─────────────────────────────────────────────────┘
    :         :      :
    :         :      [~1092 bytes unknown]
    :         :      :
    ┌─────────────────────────────────────────────────┐
    │ 0x5C8     8    MTLResourceList*                │◄── CRASH SITE 1
    │                Metal resource tracking         │    useResourceCommon
    │                                                │    reads this field
    └─────────────────────────────────────────────────┘
    0x5D0     8    (padding/unknown)
    ┌─────────────────────────────────────────────────┐
    │ 0x5D8     8    IOGPUResourceList*              │    IOGPU resource list
    └─────────────────────────────────────────────────┘
    :         :      :
    :         :      [~96 bytes unknown]
    :         :      :
    ┌─────────────────────────────────────────────────┐
    │ 0x638     8    ResourceGroupUsage*             │    Resource group binding
    └─────────────────────────────────────────────────┘
    :         :      :
    0x???     ?    Total size unknown


NOTE: When context pointer is NULL, crash occurs at these absolute addresses:
      NULL + 0x098 = 0x98   → KERN_INVALID_ADDRESS
      NULL + 0x184 = 0x184  → KERN_INVALID_ADDRESS
      NULL + 0x5C8 = 0x5C8  → KERN_INVALID_ADDRESS
```

## Figure 8: Three Crash Sites in AGXMetalG16X Driver

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         CRASH SITE 1: setComputePipelineState:                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  Call Stack:                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │ PyTorch → MPS → MPSMatrixMultiplication → Metal →                       │  ║
║  │   -[AGXG16XFamilyComputeContext setComputePipelineState:] →             │  ║
║  │     AGX::ContextCommon::useResourceCommon()                 ◄── CRASH   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  Disassembly at crash:                                                         ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │ 0x264334:  mov    x20, x0               ; self (context) → x20          │  ║
║  │            ...                                                           │  ║
║  │ 0x264370:  ldr    x0, [x20, #0x5c8]     ; CRASH: x20 = NULL             │  ║
║  │            ─────────────────────────────────────────────────────────    │  ║
║  │                      ▲                                                   │  ║
║  │                      │                                                   │  ║
║  │            SIGSEGV: NULL + 0x5c8 = 0x5c8 (unmapped)                     │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  Register dump at crash:                                                       ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │ x20 = 0x0000000000000000   (NULL - the context pointer)                 │  ║
║  │ x21 = 0x0000600001234000   (valid resource pointer)                     │  ║
║  │ pc  = 0x00000001a2264370   (instruction at crash)                       │  ║
║  │ lr  = 0x00000001a2263e7c   (return address)                             │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════╗
║                         CRASH SITE 2: prepareForEnqueue                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  Call Stack:                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │ PyTorch → MPS → commitAndWait → Metal →                                 │  ║
║  │   -[AGXMTLCommandBuffer commit] →                                       │  ║
║  │     prepareForEnqueue()                                     ◄── CRASH   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  Fault address: NULL + 0x98 = 0x98                                            ║
║  Access type: READ                                                             ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════╗
║                         CRASH SITE 3: allocateUSCSpillBuffer                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  Call Stack:                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │ PyTorch → MPS → dispatchEncoder → Metal →                               │  ║
║  │   -[AGXG16XFamilyComputeContext setComputePipelineState:] →             │  ║
║  │     allocateUSCSpillBuffer()                                ◄── CRASH   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  Fault address: NULL + 0x184 = 0x184                                          ║
║  Access type: WRITE                                                            ║
║                                                                                ║
║  This site writes to the context structure during buffer allocation.          ║
║  When context is NULL, write to 0x184 triggers SIGSEGV.                       ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

## Figure 9: How NULL Pointer Reaches Driver

```
                    Normal Flow                     Buggy Flow
                    ═══════════                     ══════════

    Application        │                                │
         │             │                                │
         ▼             │                                │
    ┌─────────────┐    │                     ┌──────────────────────┐
    │ Create      │    │                     │ Thread B destroys    │
    │ Context     │    │                     │ Thread A's context   │
    └──────┬──────┘    │                     └──────────┬───────────┘
           │           │                                │
           ▼           │                                ▼
    ┌─────────────┐    │                     ┌──────────────────────┐
    │ context_ptr │    │                     │ context_ptr = NULL   │
    │ = 0x12340   │    │                     │ (object freed)       │
    │ (valid)     │    │                     │                      │
    └──────┬──────┘    │                     └──────────┬───────────┘
           │           │                                │
           ▼           │                                ▼
    ┌─────────────┐    │                     ┌──────────────────────┐
    │ Begin       │    │                     │ Thread A continues   │
    │ Encoding    │    │                     │ with stale pointer   │
    └──────┬──────┘    │                     └──────────┬───────────┘
           │           │                                │
           ▼           │                                ▼
    ┌─────────────┐    │                     ┌──────────────────────┐
    │ Call        │    │                     │ Call useResource     │
    │ useResource │    │                     │ with context = NULL  │
    │ (x20 = ptr) │    │                     │ (x20 = 0x0)          │
    └──────┬──────┘    │                     └──────────┬───────────┘
           │           │                                │
           ▼           │                                ▼
    ┌─────────────┐    │                     ╔══════════════════════╗
    │ ldr x0,     │    │                     ║ ldr x0,              ║
    │ [x20,#0x5c8]│    │                     ║ [x20, #0x5c8]        ║
    │             │    │                     ║                      ║
    │ x0 = valid  │    │                     ║ x20 = NULL           ║
    │ resource    │    │                     ║ → 0x0 + 0x5c8        ║
    └──────┬──────┘    │                     ║ → SIGSEGV            ║
           │           │                     ╚══════════════════════╝
           ▼           │
         SUCCESS       │
```
