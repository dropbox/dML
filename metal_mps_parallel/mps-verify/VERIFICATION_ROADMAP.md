# MPS Formal Verification Roadmap

**Status:** Active Development
**Manager:** AI Manager
**Start:** N=982
**Goal:** Build a production-quality multi-language verification platform while gaining deep experience with formal verification tools and techniques.

---

## Executive Summary

This project has two parallel objectives:

1. **Practical:** Formally verify the MPS parallel inference implementation
2. **Educational:** Build deep expertise in formal verification tools (TLA+, Lean 4, Iris/Coq, CBMC)

The platform architecture uses **Lean 4 as the integration hub** because:
- Excellent metaprogramming for building DSLs
- Can embed other specification languages
- Generates executable code
- Growing ecosystem and community

---

## Phase 1: Lean 4 Foundation ✓ COMPLETE (N=982)

**Objective:** Establish core Lean 4 infrastructure for verification.

**Completed:**
- [x] Lake project structure
- [x] Core.Types - ThreadId, StreamId, BufferId, verification status types
- [x] Core.MemoryModel - C++11 memory ordering formalization
- [x] Core.Concurrency - Atomic, Mutex, OnceFlag, TLS models
- [x] CLI skeleton (mpsverify command)
- [x] Incremental verification cache structure

**Key Learning:** Lean 4 namespaces require explicit paths; `open` doesn't work across module boundaries the way you might expect from other languages.

---

## Phase 2: TLA+ Model Checking ✓ COMPLETE (N=989)

**Objective:** Model critical concurrent protocols and verify safety/liveness properties using TLA+ and the TLC model checker.

### Why TLA+?

TLA+ excels at finding bugs in concurrent protocol *design* before implementation. It's not about verifying code directly—it's about verifying that your algorithm is correct. Many serious concurrency bugs are design flaws, not implementation typos.

**Key insight:** TLA+ finds bugs that testing cannot. A test runs one execution; TLC explores *all* possible interleavings.

### 2.1 Stream Pool Specification ✓ COMPLETE

**File:** `specs/MPSStreamPool.tla`

Models:
- Pool lifecycle (create, destroy)
- TLS stream binding with round-robin selection
- Fork safety (invalidates all TLS)
- Concurrent access by multiple threads

Properties verified:
- `NoUseAfterFree` - Cannot access stream after pool destruction
- `TLSBindingValid` - TLS always points to valid stream index
- `ForkInvalidatesTLS` - Fork clears all thread bindings
- `WorkerStreamNotDefault` - Worker threads never get stream 0

### 2.2 Allocator Specification ✓ COMPLETE (N=986)

**File:** `specs/MPSAllocator.tla`

**Why this matters:** The allocator has the most complex synchronization in the codebase—double-check locking with ABA detection, nested mutex acquisition, and TLS cache interaction.

**What to model:**

```
State Variables:
- m_mutex (global allocator lock)
- pool_mutex[pool_id] (per-size-class locks)
- buffer_blocks: Map<ptr, BufferBlock>
- BufferBlock.in_use: atomic<bool>
- BufferBlock.use_count: atomic<uint32>  // ABA generation counter
- tls_cache[thread]: List<BufferBlock*>
- allocator_alive: atomic<bool>
- pending_handlers: atomic<int>

Key Operations to Model:
1. allocate() - Find free buffer or create new
2. free() - Return to pool or TLS cache
3. getSharedBufferPtr() - The ABA double-check pattern
4. recordStream() - Cross-stream buffer tracking
5. emptyCache() - TLS cache flush with shutdown race
```

**Properties to verify:**

```tla
(* ABA Detection Correctness *)
ABADetectionSound ==
    \* If we re-acquire locks and use_count changed, we abort
    \A t \in Threads:
        (acquired_use_count[t] # current_use_count[ptr]) =>
        ~WillUseBuffer[t]

(* No Double Free *)
NoDoubleFree ==
    \A b \in BufferBlocks:
        (b.in_use = FALSE) => ~(FreeOperation(b) \in EnabledActions)

(* TLS Cache Flush Safety *)
TLSFlushSafe ==
    \* If allocator is shutting down, TLS flush completes before destruction
    (allocator_alive = FALSE) =>
        \A t \in Threads: tls_cache[t] = {}

(* Completion Handler Counting *)
HandlerCountingCorrect ==
    (pending_handlers = 0) => NoOutstandingCallbacks
```

**Worker Directive for 2.2:**
```
WORKER TASK: Create specs/MPSAllocator.tla

1. Read the actual implementation first:
   - pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm
   - pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.h
   Focus on: getSharedBufferPtr(), get_free_buffer(), free_buffer()

2. Model the ABA double-check pattern precisely:
   - First check outside lock
   - Acquire lock
   - Re-check with use_count comparison
   - This pattern appears in 6 functions (see issue 32.267 in WORKER_DIRECTIVE.md)

3. Start with CONSTANTS NumThreads=3, NumPools=2, NumBuffers=4
   Keep state space tractable for TLC

4. Run TLC and fix any counterexamples found
   Command: tlc MPSAllocator.tla -config MPSAllocator.cfg

5. Document what properties pass/fail in the spec comments
```

### 2.3 Event Pool Specification ✓ COMPLETE (N=988)

**File:** `specs/MPSEvent.tla`

**Why this matters:** The event system has subtle lifetime issues—callbacks can fire after the MPSEvent object is destroyed. The fix uses shared_ptr to ensure callback state survives.

**Properties verified:**
- `CallbackNeverAccessesDeadState` - Callbacks check alive flag before work
- `CallbackStateRefCountPositive` - shared_ptr keeps state alive
- `PoolInUseDisjoint` - No double-acquire of events
- `MutexExclusivity` - Proper event mutex usage

**What to model:**

```
State Variables:
- events[event_id]: EventState
- EventState.callback_state: shared_ptr<CallbackState>
- CallbackState.alive: atomic<bool>
- CallbackState.sync_completed: bool (under mutex)
- event_pool: List<EventId>  // Reusable event objects

Key Operations:
1. notify() - Schedule callback on GPU completion
2. synchronize() - Wait with timeout
3. ~MPSEvent() - Destructor must handle in-flight callback
4. reset() - Reuse event from pool (must not cross-talk)
```

**Properties to verify:**

```tla
(* Callback Safety *)
CallbackNeverAccessesDeadState ==
    \A e \in Events:
        CallbackRunning[e] => callback_state[e].ref_count > 0

(* No Use-After-Free in Destructor *)
DestructorWaitsForCallback ==
    \A e \in Events:
        InDestructor[e] =>
            (callback_state[e].alive = FALSE \/ WaitingOnCallback[e])

(* Pool Reuse Safety *)
NoPoolCrosstalk ==
    \A e \in Events:
        (Reused[e] /\ NewCallback[e]) =>
        OldCallbackCompleted[e]
```

**Worker Directive for 2.3:**
```
WORKER TASK: Create specs/MPSEvent.tla

1. Read the implementation:
   - pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm
   - Focus on: notify(), synchronize(), destructor, MPSEventPool

2. The key insight is shared_ptr prevents UAF:
   - Callback captures shared_ptr<CallbackState>
   - Destructor sets alive=false, then waits
   - Callback checks alive before accessing MPSEvent members

3. Model the race between destructor timeout and callback execution

4. Verify pool reuse doesn't cause one callback to see another's state
```

### 2.4 TLC Integration in Lean ✓ COMPLETE (N=989)

**Files:**
- `MPSVerify/Bridges/TLAPlus.lean` - TLC output parser
- `MPSVerify/Bridges/TLCRunner.lean` - TLC process executor
- `MPSVerify/Bridges.lean` - Module re-export

**Objective:** Automate TLC execution and parse results back into Lean.

**Implemented:**
- TLCResult structure (success, states, depth, violations, time)
- containsSubstr using zipIdx (Lean 4 compatible)
- TLC availability check (command or jar)
- CLI command: `mpsverify tla --spec=NAME` or `--all`
- knownSpecs for all three TLA+ specifications

```lean
-- Run TLC and get structured results
def runTLC (spec : FilePath) (config : FilePath) : IO TLCResult := do
  let output ← IO.Process.output {
    cmd := "tlc"
    args := #[spec.toString, "-config", config.toString, "-deadlock"]
  }
  parseTLCOutput output.stdout

structure TLCResult where
  statesGenerated : Nat
  distinctStates : Nat
  violations : List Violation
  time : Float

-- Integrate with verification report
def verifyTLA (target : VerificationTarget) : IO VerificationStatus := do
  let result ← runTLC target.spec target.config
  if result.violations.isEmpty then
    pure .verified
  else
    pure (.failed s!"TLC found {result.violations.length} violations")
```

**Worker Directive for 2.4:**
```
WORKER TASK: Implement TLC bridge in Lean

1. Create TLAPlus.lean with TLC output parser
   - Parse "X states generated, Y distinct states"
   - Parse "Error: Invariant X is violated"
   - Parse counterexample traces

2. Create TLCRunner.lean with process execution
   - Handle TLC not installed gracefully
   - Timeout support
   - Capture stderr for error messages

3. Integrate into CLI: `mpsverify tla --spec=StreamPool`

4. Test with the existing MPSStreamPool.tla spec
```

---

## Phase 3: CBMC Bounded Model Checking ✓ COMPLETE (N=998)

**Objective:** Verify actual C++ code (not just models) using bounded model checking.

### Phase 3 Progress

**Completed (N=998):**
- [x] Directory structure: `verification/cbmc/{stubs,models,harnesses}/`
- [x] Metal API stubs: `stubs/metal_stubs.h` - MTLDevice, MTLBuffer, MTLHeap, MTLCommandQueue stubs
- [x] BufferBlock model: `models/buffer_block.h` - Atomic in_use, use_count, pool model
- [x] Alloc/free harness: `harnesses/alloc_free_harness.c` - Basic allocation safety
- [x] ABA detection harness: `harnesses/aba_detection_harness.c` - Double-check pattern verification
- [x] TLS cache harness: `harnesses/tls_cache_harness.c` - Thread-local cache safety
- [x] Stream pool harness: `harnesses/stream_pool_harness.c` - Pool lifecycle, TLS binding, fork safety
- [x] CBMC installed (v6.8.0) and all 10 harnesses verified (3,856 checks)

**CBMC Verification Results (N=998):**
| Harness | Result | Assertions |
|---------|--------|------------|
| aba_detection_harness.c | ✓ PASS | 0/384 failed - ABA detection, mutex safety |
| alloc_free_harness.c | ✓ PASS | 0/239 failed - no double-free, no UAF, bounds safe |
| tls_cache_harness.c | ✓ PASS | 0/318 failed - cache bounds, no double-cache, shutdown safe |
| stream_pool_harness.c | ✓ PASS | 0/249 failed - pool lifecycle, TLS binding, fork safety, TOCTOU |

**Completed (N=998):**
- [x] Lean bridge: CBMC.lean (output parser), CBMCRunner.lean (process execution)
- [x] Updated Bridges.lean to export CBMC modules

### Why CBMC?

TLA+ verifies the *design*. CBMC verifies the *implementation*. CBMC unrolls loops and converts C/C++ to SAT/SMT problems, checking for:
- Buffer overflows
- Null pointer dereferences
- Assertion violations
- Memory leaks
- Use-after-free (limited)

**Limitation:** CBMC requires bounding loops. It proves "no bug exists within N iterations" not "no bug exists ever." But N=10 catches most bugs.

### 3.1 CBMC Harness Design

**Challenge:** MPS code uses Objective-C++ and Metal APIs that CBMC can't handle directly. We need to create *verification harnesses* that stub out Apple frameworks.

**Approach:**
```cpp
// verification/cbmc/stubs/metal_stubs.h
// Stub out Metal types for CBMC verification

struct MTLDevice_stub {
    bool is_valid;
};

struct MTLCommandQueue_stub {
    MTLDevice_stub* device;
    int queue_id;
};

#define id MTLDevice_stub*
#define MTLDevice MTLDevice_stub
#define MTLCommandQueue MTLCommandQueue_stub
```

**Harness structure:**
```cpp
// verification/cbmc/harnesses/allocator_harness.cpp

#include "metal_stubs.h"
#include "allocator_model.h"

void main() {
    // Non-deterministic inputs
    int num_threads = nondet_int();
    __CPROVER_assume(num_threads > 0 && num_threads <= 8);

    size_t alloc_size = nondet_size_t();
    __CPROVER_assume(alloc_size > 0 && alloc_size <= 1024*1024);

    // Create allocator
    MPSHeapAllocatorModel allocator;

    // Simulate concurrent operations
    for (int t = 0; t < num_threads; t++) {
        void* ptr = allocator.allocate(alloc_size);
        __CPROVER_assert(ptr != nullptr || allocator.is_oom(),
                        "Allocation must succeed or report OOM");

        if (ptr) {
            allocator.free(ptr);
        }
    }

    // Check no leaks
    __CPROVER_assert(allocator.allocated_count() == 0,
                    "All allocations must be freed");
}
```

### 3.2 Verification Targets

**Priority 1: Memory Safety**
- `getSharedBufferPtr()` - Pointer validity
- `get_free_buffer()` - Buffer bounds
- `TLSBlockCache::get()` - Cache indexing

**Priority 2: Assertion Checking**
- Verify all `TORCH_INTERNAL_ASSERT` conditions
- Check `MPS_PRECONDITION` / `MPS_POSTCONDITION` contracts

**Worker Directive for Phase 3:**
```
WORKER TASK: Set up CBMC infrastructure

1. Create verification/cbmc/ directory structure:
   - stubs/ - Metal API stubs
   - harnesses/ - Verification harness files
   - models/ - Simplified C++ models of MPS code

2. Start with a minimal model of BufferBlock:
   struct BufferBlock {
       void* ptr;
       size_t size;
       bool in_use;
       uint32_t use_count;
   };

3. Write harness for basic alloc/free cycle

4. Run: cbmc harness.cpp --unwind 10 --pointer-check --bounds-check

5. Document findings - CBMC will likely find edge cases we missed
```

---

## Phase 4: Static Analysis Integration ✓ COMPLETE (N=999)

**Objective:** Add lightweight continuous verification via static analysis tools.

### Phase 4 Progress

**Completed (N=998):**
- [x] Created `verification/static/clang_annotations.h` - Full set of TSA macros
- [x] Created `verification/static/run_analysis.sh` - Analysis runner script
- [x] Created `MPSVerify/Bridges/StaticAnalysis.lean` - Lean bridge for output parsing

**Completed (N=999):**
- [x] Created `pytorch-mps-fork/aten/src/ATen/mps/MPSThreadSafety.h` - TSA macros for MPS code
- [x] Applied TSA annotations to `MPSAllocator.h`:
  - `m_mutex` marked as CAPABILITY("allocator_mutex")
  - `pool_mutex` marked as CAPABILITY("pool_mutex") in BufferPool
  - Data members marked with GUARDED_BY
  - Public API marked with EXCLUDES, private methods with REQUIRES
- [x] Applied TSA annotations to `MPSStream.h`:
  - `_streamMutex` marked as CAPABILITY("stream_mutex")
  - `stream_creation_mutex_` marked as CAPABILITY("pool_creation_mutex")
  - Command buffer/encoder state marked with GUARDED_BY
  - Public methods marked with EXCLUDES, private commit methods with REQUIRES
- [x] Applied TSA annotations to `MPSEvent.h`:
  - `m_mutex` marked as CAPABILITY("event_mutex")
  - `sync_mutex` in CallbackState marked as CAPABILITY
  - `m_mutex` in MPSEventPool marked as CAPABILITY("event_pool_mutex")
  - All protected data marked with GUARDED_BY
  - Conditional locking methods marked with NO_THREAD_SAFETY_ANALYSIS

**Note:** Full static analysis requires a complete PyTorch build environment with
`compile_commands.json`. The annotation headers parse correctly in isolation.

### 4.1 Clang Thread Safety Annotations

**Why:** Zero runtime cost, catches lock ordering bugs at compile time.

**What to add:**
```cpp
// MPSAllocator.h
class MPSHeapAllocatorImpl {
private:
    std::recursive_mutex m_mutex CAPABILITY("allocator_mutex");

    ska::flat_hash_map<const void*, BufferBlock*> m_allocated_buffers
        GUARDED_BY(m_mutex);

public:
    void* allocate(size_t size) EXCLUDES(m_mutex);
    void free(void* ptr) EXCLUDES(m_mutex);

private:
    bool get_free_buffer_internal(/*...*/) REQUIRES(m_mutex);
};
```

**Annotations to use:**
- `CAPABILITY("name")` - Declares a mutex
- `GUARDED_BY(mutex)` - Data protected by mutex
- `REQUIRES(mutex)` - Function requires lock held
- `EXCLUDES(mutex)` - Function must not hold lock
- `ACQUIRE(mutex)` - Function acquires lock
- `RELEASE(mutex)` - Function releases lock

### 4.2 Facebook Infer Integration

**Why:** Finds concurrency bugs (data races, deadlocks) through abstract interpretation.

**Setup:**
```bash
# Create compile_commands.json for the MPS files
cd pytorch-mps-fork
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

# Run Infer's concurrency checkers
infer run --racerd --starvation \
    --compilation-database compile_commands.json \
    --select-analyzers racerd,starvation
```

**Worker Directive for Phase 4:**
```
WORKER TASK: Add static analysis

1. Create verification/static/clang_annotations.h with annotation macros

2. Add annotations to these headers (don't modify .mm files yet):
   - MPSAllocator.h - Mark m_mutex and pool_mutex relationships
   - MPSStream.h - Mark _streamMutex usage
   - MPSEvent.h - Mark event pool mutex

3. Create verification/static/run_analysis.sh that:
   - Runs clang with -Wthread-safety
   - Runs infer racerd
   - Aggregates results

4. Document any real bugs found (unlikely but possible)
```

---

## Phase 5: Iris/Coq Deep Verification

**Objective:** Achieve machine-checked proofs of memory safety and race freedom using separation logic.

### Why Iris?

Iris is the state-of-the-art for concurrent program verification. Unlike testing or model checking, Iris proofs are *mathematical certainty*—if the proof checks, the property holds for all inputs, forever.

**Trade-off:** Iris proofs are labor-intensive. Reserve for the most critical code.

### 5.1 Target Functions

Focus on the 3 most complex/critical functions:

1. **`getSharedBufferPtr()`** - ABA double-check locking
   - Prove: No use-after-free even with concurrent free/realloc

2. **`getCurrentStream()`** - TLS binding with pool destruction race
   - Prove: Never returns dangling pointer

3. **`~MPSEvent()`** - Destructor with in-flight callback
   - Prove: Callback never accesses freed memory

### 5.2 Proof Strategy

Use **RefinedC** - a type system for C built on Iris that automates much of the proof:

```c
// Annotated C for RefinedC
[[rc::args("&own<buffer_block>")]]
[[rc::returns("void")]]
[[rc::requires("{buffer.in_use = true}")]]
[[rc::ensures("{buffer.in_use = false}")]]
void free_buffer(BufferBlock* buffer) {
    buffer->in_use.store(false, std::memory_order_release);
}
```

**Worker Directive for Phase 5:**
```
WORKER TASK: Set up Coq/Iris infrastructure

This is advanced work. Prerequisites:
- Coq installed (opam install coq)
- Iris installed (opam install coq-iris)
- RefinedC (clone from github.com/plv-mpi-sws/refinedc)

1. Create verification/coq/ directory
2. Start with a tiny example - verify a simple atomic counter
3. Build up to BufferBlock.in_use atomic flag semantics
4. Document the proof structure for future reference

Expected effort: 10-15 commits for one verified function
```

---

## Phase 6: Unified Platform ✓ COMPLETE (N=1000)

**Objective:** Integrate all verification tools into a single coherent platform.

### Phase 6 Progress

**Completed (N=1000):**
- [x] CLI version bump to 0.4.0
- [x] `check --all` command runs TLA+, CBMC, and static analysis
- [x] `check --tla`, `--cbmc`, `--static` for selective verification
- [x] `report` command generates unified verification report
- [x] Markdown report format with summary tables
- [x] HTML report format with basic styling
- [x] Helper functions for result aggregation

### 6.1 Architecture

```
mps-verify check --all
    │
    ├─→ TLA+ Model Checking (specs/*.tla)
    │   └─→ TLC runner, result parser
    │
    ├─→ CBMC Bounded Verification (verification/cbmc/)
    │   └─→ Harness compilation, CBMC runner
    │
    ├─→ Static Analysis (verification/static/)
    │   └─→ Clang TSA, Infer
    │
    ├─→ Coq/Iris Proofs (verification/coq/)
    │   └─→ Proof status checker
    │
    └─→ Unified Report Generation
        └─→ HTML/Markdown verification report
```

### 6.2 Incremental Verification

The platform tracks dependencies:
```json
{
  "MPSStream.mm": {
    "tla_specs": ["MPSStreamPool.tla"],
    "cbmc_harnesses": ["stream_harness.cpp"],
    "coq_proofs": ["StreamPool.v"],
    "last_verified": "2025-12-16T10:00:00Z",
    "hash": "abc123..."
  }
}
```

On `mps-verify check`:
1. Hash all source files
2. Compare to cached hashes
3. Only re-run verification for changed files
4. Aggregate results into report

### 6.3 Verification Report

```markdown
# MPS Verification Report
Generated: 2025-12-16 10:30:00

## Summary
- TLA+ Model Checking: 3/3 specs pass
- CBMC Verification: 5/5 harnesses pass
- Static Analysis: 0 warnings
- Coq Proofs: 1/3 complete

## Details

### TLA+ Results
| Spec | States | Time | Status |
|------|--------|------|--------|
| MPSStreamPool | 45,231 | 2.3s | ✓ PASS |
| MPSAllocator | 128,492 | 8.1s | ✓ PASS |
| MPSEvent | 23,847 | 1.2s | ✓ PASS |

### Properties Verified
- [x] Deadlock freedom (TLA+)
- [x] No use-after-free (TLA+, CBMC)
- [x] ABA detection correctness (TLA+)
- [x] Memory bounds safety (CBMC)
- [ ] Full memory safety proof (Coq) - in progress
```

---

## Phase 7: Operation-Level Mutex Analysis (NEW - N=1040)

**Objective:** Formally model and verify the operation-level serialization that limits parallel scaling.

### 7.1 The Problem

Benchmarks show 8-thread efficiency at only 30-35%, far below the 50%+ target. The allocator sharding (N=1039) provided only ~1-2% improvement. The root cause is **operation-level global mutexes** in Apple's MPS operations:

```cpp
// Global mutexes that serialize ALL threads doing the same operation:
static std::mutex s_linear_nograph_mutex;     // Linear.mm:19
static std::mutex s_layer_norm_mutex;         // Normalization.mm:31
static std::mutex s_bmm_tiled_mutex;          // LinearAlgebra.mm:49
static std::mutex s_lu_decomposition_mutex;   // LinearAlgebra.mm:50
static std::mutex s_lu_solve_mutex;           // LinearAlgebra.mm:51
static std::mutex s_solve_triangular_mutex;   // LinearAlgebra.mm:52
static std::mutex s_ndarray_identity_mutex;   // OperationUtils.mm:484
```

The comment explains why:
```cpp
// Apple's MPS framework has internal shared state that makes concurrent encoding
// of MPSNDArrayMatrixMultiplication kernels unsafe, even with per-thread instances.
```

### 7.2 The Two-Path Architecture

The MPS backend has two execution paths:

| Path | API | Thread-Safe | Overhead |
|------|-----|-------------|----------|
| No-Graph | MPSNDArrayMatrixMultiplication | NO (requires mutex) | Low per-op |
| Graph | MPSGraph | YES (no mutex) | Graph compilation |

Path selection logic (Linear.mm:140-160):
```cpp
const bool parallel_streams_active =
    MPSStreamPool::instance().getActiveStreamCount() > 1;
const bool force_graph_path = force_graph_path_env || parallel_streams_active;

if (!force_graph_path && is_contiguous) {
    _mps_linear_nograph(...);  // Uses mutex
} else {
    // MPSGraph path - thread-safe
}
```

### 7.3 Experimental Results (N=1040)

| Model | Default | MPS_FORCE_GRAPH_PATH=1 | Delta |
|-------|---------|------------------------|-------|
| nn.Linear 8-thread | 30.6% | **35.0%** | +4.4% |
| TransformerEncoderLayer 8-thread | 21.7% | 20.8% | -0.9% |

**Key Finding:** Graph path helps simple ops (nn.Linear) but hurts complex ops (Transformer) due to compilation overhead.

### 7.4 TLA+ Extension: Operation Serialization Model

**File:** `specs/MPSOperationMutex.tla` (NEW)

```tla+
VARIABLES
    op_mutex_holder,        \* Thread holding operation mutex (0 = unlocked)
    path_selection,         \* Thread -> {"graph", "nograph", "none"}
    g_worker_stream_used,   \* Global flag for worker stream usage
    graph_compilation_time, \* Time spent compiling graphs

\* Model the path selection race condition
SelectPath(t) ==
    /\ pc[t] = "select_path"
    /\ IF g_worker_stream_used THEN
           /\ path_selection' = [path_selection EXCEPT ![t] = "graph"]
           /\ graph_compilation_time' = graph_compilation_time + COMPILATION_COST
       ELSE
           path_selection' = [path_selection EXCEPT ![t] = "nograph"]
    /\ pc' = [pc EXCEPT ![t] = "execute_op"]

\* Scalability property: Graph path allows parallelism
GraphPathParallel ==
    g_worker_stream_used =>
        Cardinality({t \in Threads : pc[t] = "execute_op"}) >= 2

\* No-graph path serializes
NoGraphSerializes ==
    ~g_worker_stream_used =>
        Cardinality({t \in Threads : pc[t] = "encode_kernel"}) <= 1
```

### 7.5 Hypotheses to Test

| # | Hypothesis | Test Method | Status |
|---|------------|-------------|--------|
| 1 | Graph path scales better | MPS_FORCE_GRAPH_PATH=1 | ✓ Confirmed (+4.4% for Linear) |
| 2 | Graph overhead hurts complex ops | Benchmark Transformer | ✓ Confirmed (-0.9%) |
| 3 | GPU saturation is ceiling | Profile GPU utilization | PENDING |
| 4 | `getActiveStreamCount()` race | Add instrumentation | PENDING |

### 7.6 Worker Directive for Phase 7

```
WORKER TASK: Operation Mutex Formal Analysis

1. Create specs/MPSOperationMutex.tla
   - Model the two-path selection (graph vs no-graph)
   - Model the g_worker_stream_used flag race
   - Model graph compilation overhead as a cost

2. Verify with TLC:
   - NoGraphSerializes: No-graph path allows only 1 thread in critical section
   - GraphPathParallel: Graph path allows multiple threads
   - PathSelectionSound: Path selection uses correct flag value

3. Instrument pytorch-mps-fork to measure:
   - How often each path is taken
   - Actual time spent in mutex vs graph compilation
   - GPU utilization at 8 threads

4. Determine optimal strategy:
   - Always graph path? (simpler, consistent)
   - Adaptive based on workload size?
   - Shard the no-graph mutex?

Expected effort: 4-6 commits
```

### 7.7 Potential Solutions

| Solution | Pros | Cons | Effort |
|----------|------|------|--------|
| Force graph path | Simple, thread-safe | Compilation overhead | Low |
| Shard no-graph mutex | Better parallelism | Complex, may crash | Medium |
| Remove mutex (test Apple fix) | Best performance | May crash | High risk |
| Hybrid: size threshold | Optimal for both | Complex logic | Medium |

### 7.8 Apple Metal Limitation

The fundamental issue is that Apple's `MPSNDArrayMatrixMultiplication` has internal shared state. This is **not something we can fix** without Apple's cooperation. Our options are:

1. **Use MPSGraph** - Apple's newer API is thread-safe
2. **Accept serialization** - For operations where graph overhead > mutex overhead
3. **Report to Apple** - File radar for thread-safety improvement

---

## Phase 8: Paragon Verification Gate (NEW - MANAGER 2025-12-17)

**Objective:** Turn verification into a reliable, offline, enforceable engineering gate with traceability and “scalability proof obligations.”

**Primary docs:**
- `FORMAL_VERIFICATION_PARAGON_DESIGN.md` (complete design report: contract, properties, artifacts, assumptions)
- `WORKER_VERIFICATION_PARAGON_CHECKLIST.md` (worker execution checklist with evidence requirements)

### 8.1 Deliverables (must become true)

1. **TLA+ must run offline by default**
   - `mpsverify tla --all` must find and use the vendored TLC jar (`mps-verify/tools/tla2tools.jar`) when `tlc` is not installed.

2. **Lean must have zero `sorry` in gated modules**
   - Any remaining conjectures must be isolated from the verification gate.

3. **Static analysis must be real and compile-db driven**
   - Clang TSA must replay `pytorch-mps-fork/build/compile_commands.json` and be gated.

4. **Unified artifact reporting**
   - `mpsverify check --all` must produce durable logs + machine-readable summary per run.

5. **Traceability + property catalog**
   - A maintained map of code anchors → specs/harnesses/theorems → named properties → assumptions → evidence artifacts.

6. **Aspirational/scaling properties**
   - Add at least one “should-pass-when-optimized” property (e.g., avoidable global serialization in parallel mode) and track it explicitly as a goal.

### 8.2 Next coverage expansions (after the gate is stable)

Once `mpsverify check --all` is reliable, expand formal coverage to the highest-ROI concurrency protocols (do not boil the ocean). Use `FORMAL_VERIFICATION_PARAGON_DESIGN.md` Appendix B as the canonical inventory and start with B1:

1. **`recordStream()` cross-stream lifetime protocol** (allocator + events)
2. **Batch queue liveness (no stuck futures)**
3. **Global encoding-lock deadlock freedom + “no wait under global lock” aspirational goal**
4. **Stream slot allocator/backpressure correctness**
5. **Dispatch/TLS hazard and reentrancy structural enforcement**

---

## Phase 9: MPS Binary Research + Dual Upstream Patches (NEW - MANAGER 2025-12-17)

**Prerequisite**: Complete Phase 8 (Paragon Verification Gate) first.

**Goal**: Use SOTA formal verification to analyze Apple MPS internals, then submit verified patches to BOTH Apple (MPS/MLX) AND PyTorch.

**Reference**: See `MPS_RESEARCH_PLAN.md` for complete details.

### 9.1 Overview

```
MPS Binary Research → Formal Verification → Dual Upstream Impact
     (Ghidra)            (TLA+/Lean/CBMC)    (Apple + PyTorch)
```

This phase applies our formal verification infrastructure to:
1. **Reverse engineer** Apple MPS to find the exact race condition
2. **Formally model** both the bug (TLA+) and the fix (Lean proofs)
3. **Submit patches** to Apple MPS AND PyTorch with formal proofs

### 9.2 Why Research Helps Everything

| Without Research | With Research |
|------------------|---------------|
| Steel design: Guessing | Steel design: Know EXACT patterns to avoid |
| Apple report: "It crashes" | Apple report: "Bug at offset 0x1234, here's fix" |
| Formal proofs: Assumptions | Formal proofs: Model ACTUAL internals |

### 9.3 Activities

**Binary Analysis (5-7 commits)**:
- Extract MPS from dyld shared cache
- Ghidra analysis of `MPSNDArrayMatrixMultiplication`
- Identify global state variables and race condition path
- Dynamic analysis with Frida to trace execution

**Formal Verification (8-12 commits)**:
- TLA+ model of MPS bug (prove race exists)
- TLA+ model of fix (prove fix is safe)
- Lean proofs of fix correctness
- CBMC harnesses validating bug and fix

**Upstream Patches (10-15 commits)**:
- Apple MPS patch with formal verification artifacts
- Apple MLX contribution (formal specs for Steel)
- PyTorch Steel integration with verified kernels
- Publication (paper/blog documenting methodology)

### 9.4 Deliverables

| Deliverable | Target | Evidence |
|-------------|--------|----------|
| MPS Bug Report | Apple Feedback Assistant | Offsets, disassembly, repro |
| MPS Patch | Apple (conceptual fix) | Formal proof of correctness |
| MLX Formal Specs | github.com/ml-explore/mlx PR | TLA+, Lean specs for Steel |
| PyTorch Steel | github.com/pytorch/pytorch PR | Verified thread-safe kernels |
| Research Paper | OSDI/SOSP/USENIX | Full methodology + results |

### 9.5 Legal Framework

Research is protected under:
- DMCA §1201(f) - Interoperability exception
- DMCA §1201(j) - Security research exemption
- First Amendment - Publication rights

See `reports/main/mps_research_legal_framework_N1042_2025-12-17.md`.

### 9.6 Success Criteria

| Criterion | Measure |
|-----------|---------|
| Bug identified | Exact offset and disassembly documented |
| Formally verified | TLA+ checked, Lean compiles, CBMC passes |
| Apple acknowledges | Bug report accepted or patch considered |
| PyTorch merged | Steel kernels in pytorch/pytorch |
| Publication | Paper submitted or blog posts live |

### 9.7 Expected Impact

- **PyTorch MPS**: Threading safe but plateaus at ~3,800 ops/s (GPU command queue bottleneck); batching is ~373x more efficient
- **Apple MPS**: Thread-safety fix benefits ALL Apple ML frameworks
- **MLX**: Formal verification specs added to Apple's ML framework
- **Research**: Novel methodology for GPU framework verification

---

## Learning Objectives by Phase

| Phase | Tool | Key Learning |
|-------|------|--------------|
| 1 | Lean 4 | Dependent types, tactic proofs, metaprogramming |
| 2 | TLA+ | Temporal logic, state machines, model checking |
| 3 | CBMC | Bounded model checking, SAT/SMT solving |
| 4 | Clang/Infer | Static analysis, abstract interpretation |
| 5 | Iris/Coq | Separation logic, concurrent reasoning |
| 6 | Integration | Tool orchestration, incremental verification |
| 7 | TLA+ (ext) | Operation-level serialization, performance modeling |
| 8 | Engineering | Verification as a gate, traceability, artifact discipline |
| 9 | Research | Binary RE, formal bug modeling, upstream contribution |

---

## Worker Checklist

For each phase, workers should:

1. **Read the source code first** - Understand what you're verifying
2. **Start small** - Verify a subset before scaling up
3. **Document counterexamples** - If TLC finds a bug, that's valuable!
4. **Commit incrementally** - One spec or proof per commit
5. **Update this roadmap** - Mark completed items, add lessons learned

---

## Timeline Estimates

| Phase | Estimated Commits | Status |
|-------|-------------------|--------|
| 1. Lean Foundation | 10-12 | ✓ Complete |
| 2. TLA+ Integration | 8-10 | ✓ Complete |
| 3. CBMC | 6-8 | ✓ Complete (N=1289) - 10 harnesses, 3,856 checks |
| 4. Static Analysis | 5-6 | ✓ Complete (N=999) - TSA annotations applied |
| 5. Iris/Coq | 12-15 | 🔄 In Progress (N=1293) - 6 modules compiling |
| 6. Integration | 10-12 | ✓ Complete (N=1000) - Unified CLI |
| 7. Op-Level Mutex | 4-6 | ✓ Complete (N=1042) - Root cause: Apple MPS limitation |
| 8. Paragon Gate | 4-8 | **NEXT** - make verification enforceable |
| 9. MPS Research | 23-34 | PLANNED - binary analysis + dual upstream patches |

**Total: ~86-113 commits**

**Overall Status:** Phases 1-4, 6-7 complete. Phase 7 identified root cause: Apple MPS has internal global state causing races (Apple's own MLX avoids MPS entirely). Phase 8 (Paragon Gate) is the immediate priority to make verification enforceable. Phase 9 (MPS Research) uses our formal verification to analyze MPS binary and submit patches to both Apple and PyTorch. Phase 5 is optional/blocked on Coq installation.

**Critical Path:**
1. Complete Phase 8 (Paragon Gate) - enforce verification quality
2. Execute Phase 9 (MPS Research) - use verification to fix root cause
3. Submit upstream patches with formal proofs

---

## References

### TLA+
- [Lamport's TLA+ Home](https://lamport.azurewebsites.net/tla/tla.html)
- [Learn TLA+ (Hillel Wayne)](https://learntla.com/)
- [TLA+ Video Course](https://lamport.azurewebsites.net/video/videos.html)

### Lean 4
- [Lean 4 Documentation](https://lean-lang.org/lean4/doc/)
- [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/)
- [Metaprogramming in Lean 4](https://leanprover-community.github.io/lean4-metaprogramming-book/)

### Iris/Coq
- [Iris Project](https://iris-project.org/)
- [RefinedC](https://plv.mpi-sws.org/refinedc/)
- [Software Foundations](https://softwarefoundations.cis.upenn.edu/)

### CBMC
- [CBMC Documentation](https://www.cprover.org/cbmc/)
- [CBMC Tutorial](https://www.cprover.org/cprover-manual/)

---

*Last updated: 2025-12-17 ([MANAGER] Paragon verification gate extension)*
