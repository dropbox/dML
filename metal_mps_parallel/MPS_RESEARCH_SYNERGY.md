# MPS Research Synergy: How Binary Analysis Helps Everything

**Key Insight**: Reverse engineering MPS isn't just about fixing Apple's code - it dramatically improves ALL our options.

---

## The Synergy Map

```
                    ┌─────────────────────────────────────┐
                    │      MPS Binary Research            │
                    │  (Ghidra analysis of internals)     │
                    └─────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Option 2: Steel    │  │  Option 3: Apple    │  │  Formal Verification│
│  Integration        │  │  Bug Report         │  │  (TLA+/Lean/CBMC)   │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
           │                        │                        │
           ▼                        ▼                        ▼
   Know exactly what       Specific code paths      Model the ACTUAL
   patterns to avoid       and global variables     bug, not guesses
```

---

## How Research Helps Option 2 (Steel Integration)

### Without Research (Guessing)
```cpp
// We THINK the problem is global state, but we're not sure exactly what
// Our Steel implementation might accidentally repeat the same mistake

struct SteelGEMM {
    // Are we sure this is safe? We're just guessing based on crashes...
    ThreadLocalEncoder encoder;  // Hope this is right?
};
```

### With Research (Knowing)
```cpp
// We KNOW MPS uses a global encoder cache at offset 0x1234 in the binary
// We KNOW it's accessed without synchronization in encodeToCommandBuffer:
// We KNOW the exact race window is during resource binding

// So our Steel implementation explicitly avoids this pattern:
struct SteelGEMM {
    // Per-instance encoder - we KNOW this is what MPS should have done
    // because we saw they use a shared one at _g_encoder_cache
    InstanceLocalEncoder encoder;

    // Explicit barrier here because MPS crashes at this exact point
    void encode() {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // Critical!
        // ...
    }
};
```

### Formal Verification Benefit

```tla+
\* TLA+ model based on ACTUAL MPS internals (not guesses)

\* We discovered MPS has this global state:
VARIABLES
    g_encoder_cache,     \* Found at offset 0x1234
    g_resource_table,    \* Found at offset 0x5678
    g_scratch_buffer     \* Found at offset 0x9ABC

\* We can model the EXACT race condition:
MPSRaceCondition ==
    \E t1, t2 \in Threads :
        /\ t1 # t2
        /\ pc[t1] = "binding_resources"    \* From disassembly
        /\ pc[t2] = "binding_resources"
        /\ g_encoder_cache' = ???          \* CONFLICT!

\* And prove Steel avoids it:
SteelSafe ==
    \A t1, t2 \in Threads :
        encoder[t1] # encoder[t2]  \* Per-thread, no sharing
```

---

## How Research Helps Option 3 (Apple Bug Report)

### Without Research (Vague Report)
```
Title: MPS crashes with multiple threads

Description: When I use MPSNDArrayMatrixMultiplication from multiple
threads, it sometimes crashes. Please fix.

Steps: Run attached code with 4 threads.

[Apple likely deprioritizes - not actionable]
```

### With Research (Detailed Report)
```
Title: Thread-safety violation in MPSNDArrayMatrixMultiplication due to
       unsynchronized access to global encoder cache

Description:
MPSNDArrayMatrixMultiplication uses a global encoder cache at internal
offset +0x1234 (symbol: _g_mps_encoder_cache) that is accessed without
synchronization during encodeToCommandBuffer:.

Root Cause Analysis:
1. Thread 1 calls encodeToCommandBuffer: at 0x7fff12345678
2. Internal function MPSSetResourcesOnCommandEncoder accesses g_encoder_cache
3. Thread 2 concurrently accesses same cache
4. Race condition corrupts encoder state
5. Crash in MPSSetResourcesOnCommandEncoder+0x42

Disassembly of problematic code (MetalPerformanceShaders+0x12345):
    adrp    x0, _g_mps_encoder_cache@PAGE
    ldr     x0, [x0, _g_mps_encoder_cache@PAGEOFF]
    ; NO LOCK ACQUISITION HERE - BUG
    bl      _MPSBindResourcesToEncoder

Proposed Fix:
Add mutex acquisition before line +0x12348:
    bl      _pthread_mutex_lock
    adrp    x0, _g_mps_encoder_cache@PAGE
    ...
    bl      _pthread_mutex_unlock

Or better: Use per-instance encoder cache instead of global.

Note: Apple's own MLX framework avoids MPS entirely, using custom
Metal kernels (Steel GEMM) that don't have this issue. This suggests
Apple engineers are aware of MPS threading limitations.

[Apple engineers can immediately locate and fix the bug]
```

---

## How Research Helps Formal Verification

### Accurate Models

| Without Research | With Research |
|------------------|---------------|
| "Assume there's some global state" | "Global at offset 0x1234, type MTLEncoder*" |
| "Guess the race window" | "Race between instructions +0x42 and +0x56" |
| "Hope our model is right" | "Model matches actual binary behavior" |

### CBMC Harnesses

```cpp
// With research, we can model the ACTUAL MPS behavior:

// Discovered structure from reverse engineering
struct MPSInternalState {
    void* encoder_cache;      // offset +0x00
    void* resource_table;     // offset +0x08
    uint32_t flags;           // offset +0x10
    // ... discovered layout
};

void cbmc_harness_mps_race() {
    // Model the actual race we found
    MPSInternalState* global_state = get_mps_global();  // From analysis

    // Thread 1 path (from disassembly)
    void* enc1 = global_state->encoder_cache;
    bind_resources(enc1, ...);  // NO LOCK

    // Thread 2 path (concurrent)
    void* enc2 = global_state->encoder_cache;  // SAME POINTER
    bind_resources(enc2, ...);  // RACE!

    __CPROVER_assert(enc1 != enc2, "Encoders should be independent");
    // FAILS - proving the bug exists
}
```

---

## Research Outputs Feed Everything

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MPS Binary Research Outputs                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Global State Map                                                │
│     - _g_encoder_cache (offset, type, usage)                        │
│     - _g_resource_table (offset, type, usage)                       │
│     - Thread-unsafe access patterns                                 │
│                                                                     │
│  2. Race Condition Analysis                                         │
│     - Exact instruction sequences                                   │
│     - Race windows in nanoseconds                                   │
│     - Reproduction probability                                      │
│                                                                     │
│  3. Architecture Documentation                                      │
│     - MPS encoder lifecycle                                         │
│     - Resource binding flow                                         │
│     - Why per-instance doesn't help                                 │
│                                                                     │
│  4. Comparison with MLX                                             │
│     - What MLX does differently                                     │
│     - Why Steel is thread-safe                                      │
│     - Proof that Apple knows about this                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Steel Design  │      │ Apple Report  │      │ TLA+/Lean     │
│               │      │               │      │ Specs         │
│ - Avoid exact │      │ - Cite exact  │      │ - Model exact │
│   patterns    │      │   offsets     │      │   behavior    │
│ - Test same   │      │ - Show        │      │ - Prove Steel │
│   scenarios   │      │   disassembly │      │   avoids bug  │
└───────────────┘      └───────────────┘      └───────────────┘
```

---

## Updated Research Plan

### Phase 1: Binary Analysis (Enables Everything)
1. Extract MPS from dyld cache
2. Load into Ghidra, identify key symbols
3. Find global state variables
4. Trace race condition path
5. **Output**: Technical document with offsets, types, disassembly

### Phase 2: Parallel Workstreams (All Benefit from Phase 1)

| Workstream | Uses Research Output | Deliverable |
|------------|---------------------|-------------|
| Steel Integration | Avoid discovered patterns | Thread-safe kernels |
| Apple Report | Cite specific code locations | Actionable bug report |
| Formal Verification | Model actual internals | Accurate TLA+/Lean specs |
| Publication | Document everything | Research paper/blog |

### Phase 3: Validation
- Steel passes same scenarios that crash MPS
- Apple acknowledges bug report
- Formal proofs verified against actual behavior

---

## Conclusion

**One research effort, four benefits:**

1. **Steel is CORRECT** - We know exactly what to avoid
2. **Apple can FIX** - We give them exact code locations
3. **Proofs are ACCURATE** - Based on real internals, not guesses
4. **Publication is VALUABLE** - Novel technical contribution

The research isn't just nice-to-have - it's the foundation that makes everything else rigorous and trustworthy.
