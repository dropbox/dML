# MPS Research Plan: Binary Analysis + Formal Verification + Upstream Contributions

**Goal**: Use formal verification to analyze Metal/MPS thread-safety, contribute PyTorch patches, AND submit bug reports to Apple.

**Outcome**: Thread-safe parallel MPS inference. Threading is safe but plateaus at ~3,800 ops/s regardless of thread count (GPU command queue bottleneck). Batching is ~373x more efficient for throughput.

**Status**: Phase 0 complete - Steel kernel approach invalidated. N=1400+ correction: No Metal driver bug; overhead is from thread creation (use thread pools for 0% overhead).

---

## Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MPS Research → Dual Upstream Impact                      │
│                        (Updated: Steel path invalidated)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────┐                              │
│                    │  MPS Binary Research    │                              │
│                    │  (Ghidra + Dynamic)     │                              │
│                    └───────────┬─────────────┘                              │
│                                │                                            │
│                    ┌───────────▼─────────────┐                              │
│                    │  Formal Verification    │                              │
│                    │  (TLA+ / Lean / CBMC)   │                              │
│                    └───────────┬─────────────┘                              │
│                                │                                            │
│              ┌─────────────────┴─────────────────┐                          │
│              │                                   │                          │
│              ▼                                   ▼                          │
│   ┌─────────────────────┐             ┌─────────────────────┐              │
│   │  PyTorch Upstream   │             │   Apple Upstream    │              │
│   │                     │             │                     │              │
│   │  • Mutex patches    │             │  • Bug report filed │              │
│   │  • Thread-safe MPS  │             │  (via Feedback Asst)│              │
│   │  • ~13% vs baseline │             │  • MLX issue shared │              │
│   └─────────────────────┘             └─────────────────────┘              │
│                                                                             │
│   Impact: ALL PyTorch users          Impact: Awareness of Metal/AGX         │
│   on Apple Silicon benefit           limitation for Apple to fix            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Validate Steel Hypothesis - COMPLETE (N=1055)

**STATUS: HYPOTHESIS INVALIDATED**

### 0.1 Benchmark Results

MLX v0.30.0 was tested on Apple M4 Max with Python threading:

| Test | Result |
|------|--------|
| Single-threaded baseline | PASS (445-1834 ops/s) |
| 2+ threads | **CRASH** - Metal assertion failure |

**Error Message** (identical for all multi-threaded tests):
```
-[AGXG16XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:nextEncoderClass:]:1091:
failed assertion 'A command encoder is already encoding to this command buffer'
```

### 0.2 Key Finding: Issue is in Metal/AGX, Not Framework

The crash occurs in Apple's AGX driver (`AGXG16XFamilyCommandBuffer`), proving:
- The thread safety issue is in Apple's closed-source Metal layer
- Both MPS and MLX hit the same limitation
- Steel kernels won't help because they use the same Metal command buffers

### 0.3 MLX Has Same Issues (Confirmed)

GitHub research confirms MLX team is aware:
- **Issue #2133**: "Thread safety: Ongoing issue" - StreamContext, cache, graph eval not thread-safe
- **PR #2104**: "Metal thread safety" - Adding mutex locks (same approach as our MPS patches)
- **Issue #2067**: "[BUG] thread issues with evaluation"

### 0.4 Efficiency Comparison

| Threads | MPS (our patches) | MLX v0.30.0 |
|---------|-------------------|-------------|
| 1 | WORKS | WORKS |
| 2 | ~55% efficiency | CRASH |
| 4 | ~30% efficiency | CRASH |
| 8 | ~13% efficiency | CRASH |

**Conclusion:** Our MPS patches are AHEAD of MLX in thread safety.

### 0.5 Decision

**DO NOT proceed with Steel integration for threading.** N=1400+ correction: The ~13% efficiency vs baseline is due to threading overhead (thread creation), not a Metal limitation. Use thread pools for 0% overhead.

**Report:** `reports/main/mlx_benchmark_N1055_2025-12-17.md`

---

## Phase 1: MPS Binary Analysis (5-7 commits)

### 1.1 Environment Setup

```bash
# Research machine setup (with SIP considerations)
mkdir -p ~/mps-research/{extracted,analysis,findings}

# Tools needed
brew install ghidra  # Or download from ghidra-sre.org
brew install radare2  # Alternative disassembler
pip install frida-tools  # Dynamic instrumentation
```

### 1.2 MPS Extraction

```bash
# The dyld shared cache contains MPS framework
# Location: /System/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e

# RECOMMENDED: Use Ghidra's built-in dyld cache loader
# File -> Import -> select dyld_shared_cache_arm64e
# Ghidra will parse and extract frameworks automatically

# ALTERNATIVE: Build dsc_extractor from Apple's dyld source
# git clone https://github.com/apple-oss-distributions/dyld
# cd dyld && xcodebuild -project dyld.xcodeproj -scheme dsc_extractor -configuration Release
# ./build/Release/dsc_extractor /System/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e ~/mps-research/extracted

# NOTE: dyld_shared_cache_util is NOT distributed by Apple - don't use
```

### 1.3 Static Analysis Targets

| Symbol | Purpose | Priority |
|--------|---------|----------|
| `MPSNDArrayMatrixMultiplication` | Main matmul class | HIGH |
| `MPSSetResourcesOnCommandEncoder` | Crash location | HIGH |
| `_g_*` symbols | Global state | HIGH |
| `MPSNDArrayIdentity` | Also has mutex | MEDIUM |
| `MPSMatrixDecompositionLU` | Also has mutex | MEDIUM |

### 1.4 Analysis Methodology

```python
# Ghidra script to find global state
# mps_analysis.py

from ghidra.program.model.symbol import SymbolType

def find_global_state():
    """Find all global/static variables in MPS"""
    globals = []
    sym_table = currentProgram.getSymbolTable()

    for sym in sym_table.getAllSymbols(True):
        if sym.getSymbolType() == SymbolType.LABEL:
            name = sym.getName()
            if name.startswith("_g_") or name.startswith("_s_"):
                globals.append({
                    'name': name,
                    'address': sym.getAddress(),
                    'refs': len(list(sym.getReferences()))
                })

    return sorted(globals, key=lambda x: x['refs'], reverse=True)

def find_race_window():
    """Identify unsynchronized access to global state"""
    # Look for patterns:
    # 1. Load from global (adrp + ldr)
    # 2. No lock acquisition (bl _pthread_mutex_lock)
    # 3. Modification or use
    pass
```

### 1.5 Dynamic Analysis

**NOTE: Frida requires SIP disabled for system frameworks.**
Options:
- `csrutil disable` in Recovery Mode (temporary for research)
- Run in VM without SIP
- Use DTrace instead (works with SIP): `sudo dtrace -n 'objc$target:MPSNDArrayMatrixMultiplication::entry { printf("%s", probefunc); }' -p <pid>`
- Skip dynamic analysis and rely on static analysis only

```javascript
// Frida script to trace MPS execution
// mps_trace.js
// REQUIRES: SIP disabled or running in VM

// Correct ObjC hooking syntax (NOT Module.findExportByName)
if (ObjC.available) {
    var MPSMatMul = ObjC.classes.MPSNDArrayMatrixMultiplication;
    if (MPSMatMul) {
        Interceptor.attach(MPSMatMul["- encodeToCommandBuffer:sourceArrays:destinationArray:"].implementation, {
            onEnter: function(args) {
                this.self = args[0];
                this.cmdBuf = args[2];
                console.log("[Thread " + Process.getCurrentThreadId() + "] encode START");
                console.log("  self: " + this.self);
                console.log("  cmdBuf: " + this.cmdBuf);
            },
            onLeave: function(retval) {
                console.log("[Thread " + Process.getCurrentThreadId() + "] encode END");
            }
        });
    }
}
```

### 1.6 Deliverables

1. **MPS Internal Architecture Document**
   - Class hierarchy
   - Global state inventory
   - Thread-unsafe patterns

2. **Race Condition Technical Report**
   - Exact code locations
   - Disassembly snippets
   - Reproduction timeline

3. **Comparison with MLX**
   - MLX has the same Metal/AGX issue (crashes at 2+ threads)
   - Both frameworks need mutex-based workaround (MLX PR #2104)

---

## Phase 2: Formal Verification of Findings (8-12 commits)

### 2.1 TLA+ Model of MPS Bug

```tla+
---------------------------- MODULE MPSThreadUnsafe ----------------------------
\* Formal model of MPS race condition based on binary analysis

CONSTANTS
    Threads,                \* Set of thread IDs, e.g., {1, 2, 3, 4}
    Encoders,               \* Set of encoder IDs (model global encoder pool)
    NULL,                   \* Null value for uninitialized state
    GLOBAL_ENCODER_ADDR     \* Discovered address: 0x???? (documentation only)

VARIABLES
    pc,                     \* Program counter per thread
    global_encoder,         \* The problematic global state
    encoder_in_use,         \* Is encoder currently being used
    thread_encoder          \* What each thread thinks it has

vars == <<pc, global_encoder, encoder_in_use, thread_encoder>>

TypeInvariant ==
    /\ pc \in [Threads -> {"idle", "get_encoder", "bind_resources", "encode", "done"}]
    /\ global_encoder \in Encoders \cup {NULL}
    /\ encoder_in_use \in BOOLEAN
    /\ thread_encoder \in [Threads -> Encoders \cup {NULL}]

Init ==
    /\ pc = [t \in Threads |-> "idle"]
    /\ global_encoder \in Encoders  \* Some encoder exists globally
    /\ encoder_in_use = FALSE
    /\ thread_encoder = [t \in Threads |-> NULL]

\* Thread starts work
Start(t) ==
    /\ pc[t] = "idle"
    /\ pc' = [pc EXCEPT ![t] = "get_encoder"]
    /\ UNCHANGED <<global_encoder, encoder_in_use, thread_encoder>>

\* MPS behavior (from disassembly) - NO LOCK!
GetEncoder(t) ==
    /\ pc[t] = "get_encoder"
    /\ thread_encoder' = [thread_encoder EXCEPT ![t] = global_encoder]
    /\ pc' = [pc EXCEPT ![t] = "bind_resources"]
    /\ UNCHANGED <<global_encoder, encoder_in_use>>

BindResources(t) ==
    /\ pc[t] = "bind_resources"
    /\ encoder_in_use' = TRUE
    /\ pc' = [pc EXCEPT ![t] = "encode"]
    /\ UNCHANGED <<global_encoder, thread_encoder>>

Encode(t) ==
    /\ pc[t] = "encode"
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ encoder_in_use' = FALSE
    /\ UNCHANGED <<global_encoder, thread_encoder>>

Next ==
    \E t \in Threads:
        \/ Start(t)
        \/ GetEncoder(t)
        \/ BindResources(t)
        \/ Encode(t)

Spec == Init /\ [][Next]_vars

\* INVARIANT: MPS claims to be thread-safe (this SHOULD be violated)
MPSHasRace ==
    \E t1, t2 \in Threads :
        /\ t1 # t2
        /\ pc[t1] = "bind_resources"
        /\ pc[t2] = "bind_resources"
        /\ thread_encoder[t1] = thread_encoder[t2]  \* Same encoder!

\* This WILL be violated by TLC - proving the bug exists
MPSSafe == ~MPSHasRace
=============================================================================
```

### 2.2 TLA+ Model of Fix

```tla+
---------------------------- MODULE MPSFixed ----------------------------
\* Formal model of how MPS SHOULD work

VARIABLES
    pc,
    per_thread_encoder,     \* Each thread gets its own encoder
    mutex_holder            \* OR: mutex protects global encoder

\* Option A: Per-thread encoders (MLX approach)
GetEncoderFixed_PerThread(t) ==
    /\ pc[t] = "get_encoder"
    /\ per_thread_encoder' = [per_thread_encoder EXCEPT
         ![t] = CreateNewEncoder()]  \* Fresh encoder per thread
    /\ pc' = [pc EXCEPT ![t] = "bind_resources"]

\* Option B: Mutex protection (minimal fix)
GetEncoderFixed_Mutex(t) ==
    /\ pc[t] = "get_encoder"
    /\ mutex_holder = NULL
    /\ mutex_holder' = t
    /\ pc' = [pc EXCEPT ![t] = "bind_resources"]

\* THEOREM: Fixed version is safe
FixedSafe ==
    \A t1, t2 \in Threads :
        t1 # t2 =>
            per_thread_encoder[t1] # per_thread_encoder[t2]
=============================================================================
```

### 2.3 Lean 4 Proof of Fix Correctness

```lean
-- Lean 4 proof that our fix is correct
-- Run with: lake build

import Mathlib.Data.Finset.Basic

namespace MPS.Fix

-- Domain types (opaque - we only care about identity)
opaque Thread : Type
opaque Encoder : Type

-- Decidable equality for our types (needed for Finset)
instance : DecidableEq Thread := inferInstance
instance : DecidableEq Encoder := inferInstance

/-- The bug: multiple threads share an encoder -/
structure MPSBuggyState where
  global_encoder : Encoder
  threads_using : Finset Thread
  thread_encoder : Thread → Encoder
  all_use_global : ∀ t, t ∈ threads_using → thread_encoder t = global_encoder

/-- Buggy MPS allows race: two threads have the same encoder -/
theorem mps_has_race (s : MPSBuggyState) (t1 t2 : Thread)
    (h1 : t1 ∈ s.threads_using) (h2 : t2 ∈ s.threads_using) (hne : t1 ≠ t2) :
    s.thread_encoder t1 = s.thread_encoder t2 := by
  rw [s.all_use_global t1 h1, s.all_use_global t2 h2]
  -- Both threads use global_encoder, hence same encoder = RACE!

/-- The fix: per-thread encoders -/
structure MPSFixedState where
  thread_encoders : Thread → Encoder
  encoders_independent : ∀ t1 t2, t1 ≠ t2 → thread_encoders t1 ≠ thread_encoders t2

/-- Fixed MPS is race-free -/
theorem mps_fixed_no_race (s : MPSFixedState) (t1 t2 : Thread) (hne : t1 ≠ t2) :
    s.thread_encoders t1 ≠ s.thread_encoders t2 :=
  s.encoders_independent t1 t2 hne

/-- Mutex-based workaround (actual implementation in PyTorch patches) -/
-- NOTE: Phase 0 invalidated Steel approach. The actual fix uses mutexes.
-- Our PyTorch patches serialize command buffer access via mutex, achieving
-- thread safety at the cost of ~30% efficiency ceiling at 8 threads.
-- This is the ONLY viable approach until Apple fixes Metal/AGX.

end MPS.Fix
```

### 2.4 CBMC Verification

```cpp
// CBMC harness proving the bug exists and fix works
// Compile: cbmc --unwind 10 mps_race_harness.c

#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <stdatomic.h>

// Model of MPS internal state (from binary analysis)
struct MPSInternalState {
    void* global_encoder;       // Offset 0x???? from analysis
    int encoder_refcount;
    atomic_bool encoder_in_use; // Atomic for proper race detection
};

static struct MPSInternalState g_mps_state;

// Model encoder usage - detects concurrent access using atomic CAS
void use_encoder(void* encoder) {
    // Atomic compare-and-swap: only succeed if encoder_in_use was false
    // This properly detects races (unlike non-atomic check-then-set)
    bool expected = false;
    bool success = atomic_compare_exchange_strong(&g_mps_state.encoder_in_use,
                                                   &expected, true);

    // CBMC assertion: CAS should succeed (no concurrent usage)
    assert(success && "RACE DETECTED: encoder already in use by another thread!");

    // Simulate work
    __CPROVER_assume(encoder != NULL);

    // Release encoder
    atomic_store(&g_mps_state.encoder_in_use, false);
}

// Model of buggy MPS behavior (NO LOCK)
void* mps_buggy_encode(void* arg) {
    // From disassembly: no lock before accessing global encoder
    void* my_encoder = g_mps_state.global_encoder;

    // RACE CONDITION: multiple threads use same encoder concurrently
    use_encoder(my_encoder);

    return NULL;
}

// Model of fixed behavior (WITH LOCK)
pthread_mutex_t fix_mutex = PTHREAD_MUTEX_INITIALIZER;
void* mps_fixed_encode(void* arg) {
    pthread_mutex_lock(&fix_mutex);
    void* my_encoder = g_mps_state.global_encoder;
    use_encoder(my_encoder);
    pthread_mutex_unlock(&fix_mutex);
    return NULL;
}

// Main verification harness
int main() {
    // Initialize global encoder
    int dummy_encoder;
    g_mps_state.global_encoder = &dummy_encoder;
    atomic_init(&g_mps_state.encoder_in_use, false);

    // Spawn two threads - CBMC will explore all interleavings
    pthread_t t1, t2;

    // TEST BUGGY VERSION: Should find race
    pthread_create(&t1, NULL, mps_buggy_encode, NULL);
    pthread_create(&t2, NULL, mps_buggy_encode, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    // CBMC will find assertion violation proving the race exists
    return 0;
}
```

---

## Phase 3: Upstream Contributions (10-15 commits)

### 3.1 Apple MPS Bug Report with Suggested Fix

**NOTE: Apple MPS is closed-source.** We cannot submit actual code patches.
Instead, we submit a detailed bug report via Apple Feedback Assistant with:
- Reproduction steps
- Root cause analysis (from binary research)
- Suggested fix approach (conceptual code showing how Apple could fix it)

```objc
// Proposed patch for MetalPerformanceShaders
// File: MPSNDArrayMatrixMultiplication.m (conceptual)

// BEFORE (buggy):
@implementation MPSNDArrayMatrixMultiplication {
    // Uses global encoder cache (discovered at offset 0x????)
}

- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)cmdBuf ... {
    // Access global state without synchronization
    id<MTLComputeCommandEncoder> encoder = [self _getGlobalEncoder];  // BUG
    [self _bindResources:encoder];
    [encoder dispatchThreadgroups:...];
}

// AFTER (fixed):
@implementation MPSNDArrayMatrixMultiplication {
    id<MTLComputeCommandEncoder> _instanceEncoder;  // Per-instance!
}

- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)cmdBuf ... {
    // Create per-invocation encoder - thread safe
    id<MTLComputeCommandEncoder> encoder =
        [cmdBuf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [self _bindResources:encoder];
    [encoder dispatchThreadgroups:...];
    [encoder endEncoding];
}

@end
```

### 3.2 Submission Package to Apple

```
apple_submission/
├── README.md                    # Overview
├── BUG_REPORT.md               # Detailed bug report
├── TECHNICAL_ANALYSIS.md       # Binary analysis findings
├── PROPOSED_FIX.md             # Conceptual fix approach (we can't patch MPS)
├── FORMAL_VERIFICATION/
│   ├── MPSThreadUnsafe.tla     # TLA+ model of bug
│   ├── MPSFixed.tla            # TLA+ model of fix
│   ├── MPSProof.lean           # Lean proof of fix correctness
│   └── cbmc_results/           # CBMC verification output
├── REPRODUCTION/
│   ├── crash_repro.mm          # Minimal crash reproduction
│   ├── race_detector.mm        # TSan-based race detector
│   └── benchmark.py            # Performance impact (shows ~13% vs baseline at 8T)
└── COMPARISON/
    ├── mlx_analysis.md         # MLX has same issue (also crashes at 2+ threads)
    └── metal_agx_limitation.md # Documentation of AGXG16XFamilyCommandBuffer issue
```

### 3.3 MLX Contribution (Updated Post-Phase 0)

**Note:** Phase 0 proved Steel won't bypass the Metal/AGX limitation. MLX itself crashes with 2+ threads (same root cause). Our contribution shifts to:

1. **Share our findings** - Report to MLX team that the issue is in `AGXG16XFamilyCommandBuffer`, not their code
2. **Mutex workaround** - MLX PR #2104 is adding mutexes (same as our MPS patches)
3. **Formal verification specs** - For MLX's mutex-based workaround, not Steel threading

```
mlx_contribution/
├── specs/
│   ├── MLXStreamContext.tla    # TLA+ spec for MLX mutex pattern
│   ├── MLXThreadSafe.lean      # Lean proof of mutex-based safety
│   └── cbmc_harnesses/         # CBMC verification of MLX workaround
├── docs/
│   └── METAL_LIMITATION.md     # Documentation of Metal/AGX threading ceiling
└── tests/
    └── parallel_stress/        # Multi-threaded stress tests (expect ~30% efficiency at 8T)
```

### 3.4 PyTorch Contribution (Updated Post-Phase 0)

**Note:** Steel kernel integration is NOT part of this contribution since Phase 0 proved it won't help with threading. Instead, we contribute our mutex-based thread-safety patches.

```
pytorch_contribution/
├── patches/
│   └── cumulative-v2.9.1-to-mps-stream-pool.patch  # Our complete patch (201 fixes)
├── aten/src/ATen/mps/
│   ├── MPSStream.mm            # Thread-local stream binding
│   ├── MPSAllocator.mm         # Race-safe allocator
│   └── MPSEvent.mm             # Thread-safe events
├── test/
│   └── test_mps_parallel.py    # Parallel safety tests (8 threads, no crashes)
└── docs/
    └── MPS_THREADING.md        # Documents Metal/AGX efficiency ceiling at 8T
```

**Why no Steel?** Phase 0 benchmarks proved the efficiency ceiling is in Apple's Metal command encoder (`AGXG16XFamilyCommandBuffer`), which Steel also uses. Steel would only help if MPS kernels were the bottleneck - they're not.

---

## Phase 4: Publication (3-5 commits)

### 4.1 Research Paper

```
Title: Formal Verification of GPU Framework Thread-Safety:
       A Case Study of Apple MetalPerformanceShaders

Authors: [Team]

Abstract:
We present a comprehensive analysis of thread-safety issues in Apple's
MetalPerformanceShaders (MPS) framework using state-of-the-art formal
verification techniques. Through binary reverse engineering combined
with TLA+ model checking and Lean theorem proving, we identify the
root cause of race conditions in Apple's closed-source Metal/AGX driver
(AGXG16XFamilyCommandBuffer) that affect ALL Metal-based ML frameworks
including PyTorch MPS, Apple MLX, and TensorFlow-metal. We document
a mutex-based workaround achieving thread-safe inference that plateaus
at ~3,800 ops/s regardless of thread count (GPU command queue bottleneck), and provide a detailed bug report to
Apple via Feedback Assistant. Our patches for PyTorch MPS demonstrate
the first thread-safe parallel inference on Apple Silicon...

1. Introduction
2. Background: Metal, MPS, and ML Frameworks
3. Methodology: Binary Analysis + Formal Verification
4. Findings: Metal/AGX Command Encoder Race Condition
5. Formal Models: TLA+ and Lean Specifications
6. Verified Workaround: Mutex-Based Thread Safety
7. Upstream Impact: PyTorch Patches + Apple Bug Report
8. Evaluation: ~13% Baseline Efficiency Analysis (threading overhead)
9. Related Work
10. Conclusion

Venue: OSDI, SOSP, USENIX Security, or S&P
```

### 4.2 Blog Post Series

1. "Reverse Engineering Apple's MPS: Finding the Metal/AGX Thread-Safety Bug"
2. "Formal Verification of GPU Frameworks with TLA+ and Lean"
3. "Thread-Safe PyTorch MPS: Patches and Apple Bug Report"

---

## Timeline Integration

### Prerequisite: Phase 0 (COMPLETE)
MLX benchmark validated Metal/AGX is the bottleneck. Steel integration NOT proceeding.

### Remaining Research (Updated Post-Phase 0)

| Phase | Activities | Notes |
|-------|------------|-------|
| 1 | Binary Analysis | Ghidra extract, find globals, document race |
| 2 | Formal Verification | TLA+ model, Lean proofs for mutex approach |
| 3a | Apple Feedback | Submit bug report via Feedback Assistant |
| 3b | PyTorch PR | Submit thread-safety patches (mutex-based, no Steel) |
| 3c | MLX Collaboration | Share Metal/AGX findings with MLX team |
| 4 | Publication | Blog posts, possible paper |

**What's NOT happening:**
- ~~Steel kernel integration~~ (Phase 0 invalidated)
- ~~MPS source patching~~ (closed source)
- ~~50% efficiency at 8 threads~~ (Metal/AGX ceiling is ~30%)

---

## Success Criteria (Updated Post-Phase 0)

| Criterion | Measure | Status |
|-----------|---------|--------|
| Phase 0 validation | MLX benchmark confirms Metal/AGX bottleneck | DONE (N=1055) |
| Bug identified | Exact offset, disassembly documented | PENDING |
| Formally verified | TLA+ model checked, Lean proof compiles | PENDING |
| Apple notified | Bug filed via Feedback Assistant with full analysis | DONE (N=1056) |
| PyTorch patch submitted | PR with mutex-based thread-safety (NO Steel) | PENDING |
| MLX collaboration | Share findings, possible formal spec contribution | PENDING |
| Publication | Blog posts live | PENDING |

**What was removed:**
- ~~"PyTorch patch accepted: PR merged with Steel kernels"~~ - Steel won't help
- ~~"Apple acknowledges"~~ - Aspirational, not a success criterion we control

**Note on Apple:** MPS is closed-source. We cannot submit PRs. Best case is Apple acknowledges the bug and fixes it in a future macOS release. Our bug report via Feedback Assistant (filed N=1056) includes detailed analysis.

---

## Impact

### Immediate
- PyTorch MPS threading plateaus at ~3,800 ops/s regardless of thread count (GPU command queue bottleneck)
- Per-thread efficiency decreases as threads increase (~27% at 8T, ~13% at 16T vs single-thread baseline)
- **Steel integration will NOT help** - The bottleneck is GPU command queue saturation, not a Metal driver bug
- MLX crashes with 2+ threads; our MPS patches are ahead in thread safety
- Batching achieves ~373x higher throughput than threading at batch 256

### Long-term
- Formal verification methodology for GPU frameworks
- Template for future GPU thread-safety analysis
- Academic contribution to systems verification
- Apple Feedback package may lead to Metal framework fix in future macOS
