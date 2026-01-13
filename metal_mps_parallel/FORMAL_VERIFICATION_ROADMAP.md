# Formal Verification Roadmap for MPS Parallel Inference

**Created:** 2025-12-18 (Ralph Loop Iteration 1)
**Purpose:** Comprehensive roadmap for applying formal verification to PyTorch MPS parallelization on Apple Silicon

---

## Executive Summary

This roadmap catalogs all formal verification techniques applicable to the MPS parallel inference project, including:
- Currently implemented verification
- New tools discovered via research
- Non-trivial applications for proving correctness
- Worker directives for each item

**Important cross-references (avoid duplication):**
- For the *verification gate contract*, traceability requirements, and the curated “where formal methods help” inventory, use `FORMAL_VERIFICATION_PARAGON_DESIGN.md` (especially Appendix B) and `WORKER_VERIFICATION_PARAGON_CHECKLIST.md`.
- For the *in-repo enforcement roadmap* of the verification gate, use `mps-verify/VERIFICATION_ROADMAP.md` (Phase 8).

---

## Part I: Current Verification Infrastructure

### ✅ COMPLETED - TLA+ Model Checking

| Spec | States Checked | Properties | Status |
|------|----------------|------------|--------|
| MPSStreamPool.tla | 7,981 | Mutual exclusion, pool integrity, deadlock freedom | PASS |
| MPSAllocator.tla | 2,821,612 | ABA detection, buffer lifecycle | PASS |
| MPSEvent.tla | 11,914,912 | Callback survival, event lifecycle | PASS |

**Total:** 14,744,505 states verified with 0 errors.

### ✅ COMPLETED - Lean 4 Theorem Proving

| Module | Theorems | Status |
|--------|----------|--------|
| ABA.lean | `cas_success_means_match`, `generation_detects_modifications`, `aba_safety_holds` | PROVEN |
| DCL.lean | Double-check locking tactics | PROVEN |
| MemoryModel.lean | C++ memory ordering model, `seq_cst_race_free`, `all_atomic_race_free` | PROVEN (N=1343) |
| Conjectures.lean | `seq_cst_race_free_conjecture` (references MemoryModel proof) | PROVEN (N=1343) |
| RaceCondition.lean | `single_thread_race_free_v2`, `release_acquire_happens_before`, `mutex_protected_race_free` | PROVEN (N=1344) |

**Sorry count:** 0 (all theorems proven, placeholder macros removed N=1344)

### ✅ COMPLETED - CBMC Bounded Model Checking

| Harness | Property | Bound | Checks | Status |
|---------|----------|-------|--------|--------|
| aba_detection_harness.c | ABA counter correctness | 15 | 384 | PASS |
| alloc_free_harness.c | Buffer lifecycle | 10 | 239 | PASS |
| stream_pool_harness.c | Pool integrity | 10 | 249 | PASS |
| tls_cache_harness.c | TLS safety | 15 | 318 | PASS |
| event_pool_harness.c | Callback lifetime safety (N=1285) | 10 | 179 | PASS |
| batch_queue_harness.c | Producer/consumer correctness (N=1286) | 15 | 380 | PASS |
| graph_cache_harness.c | Cache coherency, eviction (N=1287) | 10 | 586 | PASS |
| command_buffer_harness.c | Command buffer lifecycle (N=1288) | 10 | 696 | PASS |
| tls_binding_harness.c | TLS stream binding safety (N=1289) | 10 | 354 | PASS |
| fork_safety_harness.c | Fork handler correctness (N=1289) | 10 | 471 | PASS |

**Total:** 10 harnesses, 3,856 checks, 0 failures

### ⚠️ ANALYZED - Clang TSA

- **Original:** 92 warnings
- **Fixed:** 55 (actual lock violations)
- **Remaining:** 37 (TSA template limitations)
- **Status:** Documented in `reports/main/tsa_analysis_N1283_2025-12-18.md`
- **Root cause:** `mps_lock_guard` template prevents TSA from tracing negative capabilities

### ✅ COMPLETE - Iris/Coq Separation Logic (N=1298)

- **Status:** Phase 5 Complete - All core proofs done
- **Goal:** Separation logic proofs for MPS concurrency primitives
- **Location:** `verification/iris/`
- **Tools:** Coq 9.1.0 (Rocq Prover), coq-iris, coq-iris-heap-lang

**Completed:**
- 6 Iris modules created and compiling (prelude, mutex, aba, tls, callback, stream_pool)
- mutex_token_exclusive proved (mutual exclusion)
- gen_agree proved (generation counter agreement)
- gen_update proved (generation counter atomic update) - N=1292
- aba_detection_sound proved (ABA safety)
- is_mutex refactored to standard Iris spin lock pattern - N=1293
  - Token now in FREE disjunct: (l ↦ #false ∗ R ∗ mutex_token γ) ∨ (l ↦ #true)
  - newlock_spec added for proper lock initialization
- **ALL mutex Hoare triples now PROVEN** - N=1296, N=1297
  - newlock_spec (allocation) - PROVEN N=1296
  - acquire_spec (Löb induction for spin loop) - PROVEN N=1297
  - release_spec (lock release) - PROVEN N=1296
- **TLS and Callback proofs STRENGTHENED** - N=1298
  - stream_slot_exclusive - Stream slots have exclusive ownership
  - tls_unique_slot - TLS bindings unique per ghost name
  - tls_alloc - Fresh TLS bindings can be allocated
  - callback_token_exclusive - Callback tokens are exclusive
  - callback_schedule - Callbacks can be scheduled with token creation
  - stream_sharing_impossible - No two threads share a stream

---

## Part II: NEW Verification Opportunities

### 2.1 Apalache - Symbolic TLA+ Model Checking ✅ COMPLETE (N=1364)

**Tool:** https://github.com/informalsystems/apalache

**What it does:** Symbolic model checking for TLA+ using SMT solvers (Z3). Unlike TLC which enumerates states, Apalache uses constraint solving.

**Why it matters:** Can verify inductive invariants for unbounded parameters and handle larger state spaces.

**Application to MPS:**
```
[x] All 10 TLA+ specs verified with Apalache symbolic checking (N=1364)
```

**Results (N=1364 - First Full Run):**
| Spec | Apalache Status | Bounds |
|------|-----------------|--------|
| MPSAllocator.tla | PASS | 3 buffers, 2 streams, 2 threads |
| MPSBatchQueue.tla | PASS | 2 user threads, 1 worker, 6 ops |
| MPSCommandBuffer.tla | PASS | 2 threads, 3 buffers |
| MPSEvent.tla | PASS | 2 events, 2 streams, 2 threads |
| MPSForkHandler.tla | PASS | 2 threads, 4 ops |
| MPSFullSystem.tla | PASS | 2 threads, 2 streams |
| MPSGraphCache.tla | PASS | 2 threads, 3 entries |
| MPSKernelCache.tla | PASS | 2 threads, 3 entries |
| MPSStreamPool.tla | PASS | 2 threads, 2 streams |
| MPSTLSBinding.tla | PASS | 2 threads, 2 streams |

**Files Added:**
- 10 `*_Apalache.cfg` configs (one per spec)
- `run_apalache.sh` - Automation script for all specs
- `apalache_results.json` - JSON results from verification

**Worker Directive:**
```bash
# Run all Apalache verifications
./run_apalache.sh  # Produces apalache_results.json

# Or run single spec
cd specs
export JAVA_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
~/tools/apalache/bin/apalache-mc check --config=MPSStreamPool_Apalache.cfg MPSStreamPool.tla
```

**Scalability Note (N=1364):**
Extended bounds (4+ streams, 3+ threads) cause Z3 to report UNKNOWN due to symbolic reasoning limits. Small bounds (2-3) are optimal for Apalache. TLC handles larger enumeration bounds.

**Completed:** Multiple commits (N=1302 configs, N=1346 type annotations, N=1360-1363 Apalache compatibility, N=1364 first full run)

---

### 2.2 Frama-C / ACSL - C Specification Language

**Tool:** https://frama-c.com/

**What it does:** Static analysis and formal verification of C programs using ACSL (ANSI/ISO C Specification Language) contracts.

**Why it matters:**
- Eva plugin proves absence of runtime errors
- WP plugin proves functional correctness
- E-ACSL generates runtime checks

**Application to MPS:**
```
[ ] Add ACSL contracts to MPSAllocator getSharedBufferPtr()
[ ] Add ACSL contracts to MPSStream getCurrentStream()
[ ] Use Eva to prove absence of buffer overflows
[ ] Use WP to prove allocation invariants
```

**Example ACSL for getSharedBufferPtr:**
```c
/*@
  requires \valid(ptr);
  requires use_count > 0;
  ensures \result != \null ==> \valid(\result);
  ensures \result != \null ==> *use_count_out == \old(use_count);
  assigns *use_count_out;
*/
BufferBlock* getSharedBufferPtr(const void* ptr, uint64_t* use_count_out);
```

**Worker Directive:**
```bash
# Install Frama-C
opam install frama-c

# Run Eva analysis
frama-c -eva aten/src/ATen/mps/MPSAllocator.mm
```

**Estimated:** 8-12 commits

---

### 2.3 VeriFast - Separation Logic for C

**Tool:** https://github.com/verifast/verifast

**What it does:** Modular verification using separation logic with predictable verification time.

**Why it matters:** Can verify:
- Memory safety and thread safety
- Lock ordering invariants
- Resource ownership transfer

**Application to MPS:**
```
[ ] Annotate MPSEvent with ownership predicates
[ ] Verify callback lifetime (no use-after-free)
[ ] Verify lock hierarchy compliance
[ ] Verify buffer ownership transfer in allocator
```

**Example VeriFast annotation:**
```c
//@ predicate event_inv(MPSEvent* e) = e->m_mutex |-> ?m &*& mutex(m);

void recordLocked(MPSEvent* e)
//@ requires event_inv(e);
//@ ensures event_inv(e);
{
    // ...
}
```

**Worker Directive:**
```bash
# Install VeriFast
brew install verifast

# Verify annotated file
verifast -c MPSEvent_verified.c
```

**Estimated:** 10-15 commits

---

### 2.4 CPAchecker - Configurable Program Analysis

**Tool:** https://github.com/sosy-lab/cpachecker

**What it does:** Configurable static analysis with multiple abstract interpretation approaches.

**Why it matters:**
- Automatic invariant discovery
- Predicate abstraction
- k-induction for loop verification

**Application to MPS:**
```
[ ] Run CPAchecker on MPSStream.mm with predicate analysis
[ ] Verify loop invariants in buffer pool iteration
[ ] Check assertions automatically
[ ] Generate counterexamples for any violations
```

**Worker Directive:**
```bash
# Run CPAchecker
java -jar cpachecker.jar -config config/predicateAnalysis.properties \
  -spec config/specification/default.spc \
  aten/src/ATen/mps/MPSStream.mm
```

**Estimated:** 5-8 commits

---

### 2.5 SMACK - LLVM-based Verification

**Tool:** https://github.com/smackers/smack

**What it does:** Translates LLVM IR to Boogie verification language, enabling verification via Corral.

**Why it matters:** Works at LLVM IR level, so can verify optimized code.

**Application to MPS:**
```
[ ] Compile MPS code with Clang to LLVM IR
[ ] Translate to Boogie
[ ] Verify assertions on optimized code
[ ] Check for undefined behavior in atomic operations
```

**Worker Directive:**
```bash
# Compile to LLVM IR
clang -emit-llvm -c MPSStream.mm -o MPSStream.bc

# Verify with SMACK
smack --verifier=corral MPSStream.bc
```

**Estimated:** 6-10 commits

---

### 2.6 Temporal Verifier - First-Order Temporal Logic

**Tool:** https://github.com/vmware-research/temporal-verifier

**What it does:** Verifies temporal properties using SMT solvers with automatic invariant inference.

**Why it matters:** Can prove liveness properties (progress, termination) not just safety.

**Application to MPS:**
```
[ ] Model batch queue as transition system
[ ] Prove liveness: "Every submitted request eventually completes"
[ ] Prove progress: "If pool has free slots, threads eventually acquire"
[ ] Verify fairness properties
```

**Example temporal property:**
```
# Liveness: submitted requests complete
always (request_submitted -> eventually request_completed)

# Progress: free slots lead to acquisition
always (free_slots > 0 && thread_waiting -> eventually thread_has_stream)
```

**Worker Directive:**
```bash
# Model batch queue
cd verification/temporal
temporal-verifier verify batch_queue.fly
```

**Estimated:** 5-8 commits

---

### 2.7 GPUVerify - GPU Kernel Verification

**Tool:** https://github.com/mc-imperial/gpuverify

**What it does:** Static analysis for race and divergence freedom in GPU kernels.

**Why it matters:** Can verify that Metal shader code (if translated to OpenCL/CUDA) is race-free.

**Application to MPS:**
```
[ ] Translate critical Metal shaders to OpenCL for verification
[ ] Verify LayerNorm kernel is race-free
[ ] Verify SDPA kernel divergence freedom
[ ] Verify reduction operations
```

**Note:** GPUVerify doesn't directly support Metal, but can verify functionally equivalent OpenCL code.

**Estimated:** 8-12 commits

---

### 2.8 Extended CBMC Verification

**Current:** 4 harnesses
**Goal:** Comprehensive bounded verification

**New harnesses to create:**
```
[x] batch_queue_harness.c - Verify producer/consumer correctness (N=1286)
[x] event_pool_harness.c - Verify event allocation/deallocation (N=1285)
[x] graph_cache_harness.c - Verify cache coherency (N=1287)
[x] command_buffer_harness.c - Verify buffer lifecycle (N=1288)
[x] tls_binding_harness.c - Verify thread-local binding safety (N=1289)
[x] fork_safety_harness.c - Verify fork handler correctness (N=1289)
```

**ALL CBMC HARNESSES COMPLETE** - 10 harnesses, 3,856 checks, 0 failures

**Properties to verify:**
```
[ ] Memory bounds (--pointer-check --bounds-check)
[ ] Null pointer dereference (--pointer-check)
[ ] Memory leak detection (--memory-leak-check)
[ ] Concurrency: mutex invariants
[ ] Unsigned overflow (--unsigned-overflow-check)
```

**Worker Directive:**
```bash
# Create new harness
cat > verification/cbmc/harnesses/batch_queue_harness.c << 'EOF'
// CBMC harness for BatchQueue verification
#include "../models/batch_queue.h"

void main() {
    BatchQueue* q = batch_queue_create(8);
    __CPROVER_assume(q != NULL);

    // Non-deterministic operations
    for (int i = 0; i < 10; i++) {
        int op = nondet_int();
        if (op == 0) {
            batch_queue_submit(q, nondet_ptr());
        } else {
            batch_queue_process(q);
        }
    }

    // Verify invariants
    assert(batch_queue_size(q) >= 0);
    assert(batch_queue_size(q) <= 8);
}
EOF

# Run verification
cbmc verification/cbmc/harnesses/batch_queue_harness.c \
  --unwind 15 --pointer-check --bounds-check
```

**Estimated:** 8-12 commits

---

### 2.9 Lean 4 Extended Proofs

**Current:** ABA detection, DCL tactics
**Goal:** Complete memory model and race freedom proofs

**New theorems to prove:**
```
[ ] seq_cst_race_free - Complete the conjecture proof
[ ] double_check_locking_correctness - Full DCL safety proof
[ ] tls_binding_safety - Thread-local storage correctness
[ ] callback_lifetime_safety - Callbacks don't outlive objects
[ ] pool_exhaustion_safety - Graceful degradation on pool exhaustion
[ ] fork_handler_correctness - Fork safety verification
```

**New modules to create:**
```
[ ] MPSVerify/Proofs/CallbackLifetime.lean
[ ] MPSVerify/Proofs/TLSBindingSafety.lean
[ ] MPSVerify/Proofs/ForkSafety.lean
[ ] MPSVerify/Proofs/PoolExhaustion.lean
```

**Worker Directive:**
```lean
-- Complete seq_cst race freedom proof
theorem seq_cst_race_free (events : List MemoryEvent) :
    (∀ e ∈ events, e.op.order = .seq_cst) →
    isRaceFree events := by
  intro h_seq_cst
  -- Model seq_cst total order
  have total_order := seq_cst_implies_total_order events h_seq_cst
  -- Show conflicting accesses are ordered
  have ordered := total_order_implies_ordered events total_order
  -- Derive race freedom
  exact ordered_implies_race_free events ordered
```

**Estimated:** 10-15 commits

---

### 2.10 TLA+ Extended Specifications

**Current:** 6 core specs (StreamPool, Allocator, Event, BatchQueue, GraphCache, CommandBuffer)
**Goal:** Complete system model

**New specifications:**
```
[x] MPSBatchQueue.tla - Batch queue correctness (24,419 states, PASS)
[x] MPSGraphCache.tla - Graph cache coherency (776,185 states, PASS)
[x] MPSKernelCache.tla - Kernel cache thread safety (107.9M states, PASS) - N=1357
[x] MPSCommandBuffer.tla - Command buffer lifecycle (3,559 states, PASS)
[x] MPSForkHandler.tla - Fork handler correctness (8,606 states, PASS) - N=1356
[x] MPSFullSystem.tla - Composed system model (8.0M states, PASS) - N=1358
```

**Properties to verify:**
```
[ ] System-wide deadlock freedom
[ ] Starvation freedom under fairness
[ ] Resource exhaustion handling
[ ] Error recovery correctness
```

**Worker Directive:**
```tla
--------------------------- MODULE MPSBatchQueue ---------------------------
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    MaxQueueSize,
    NumProducers,
    NumWorkers

VARIABLES
    queue,
    workers_active,
    pending_count

TypeOK ==
    /\ queue \in Seq(Request)
    /\ Len(queue) <= MaxQueueSize
    /\ workers_active \in 0..NumWorkers
    /\ pending_count \in Nat

Init ==
    /\ queue = <<>>
    /\ workers_active = 0
    /\ pending_count = 0

Submit(producer) ==
    /\ Len(queue) < MaxQueueSize
    /\ queue' = Append(queue, NewRequest(producer))
    /\ pending_count' = pending_count + 1
    /\ UNCHANGED workers_active

Process(worker) ==
    /\ Len(queue) > 0
    /\ queue' = Tail(queue)
    /\ pending_count' = pending_count - 1
    /\ UNCHANGED workers_active

\* Liveness: All submitted requests eventually complete
Progress ==
    \A r \in Range(queue) : r.submitted ~> r.completed

=============================================================================
```

**Estimated:** 8-12 commits

---

## Part III: Prioritized Roadmap

### Phase A: Quick Wins (1-2 weeks)

| Item | Tool | Commits | Priority |
|------|------|---------|----------|
| Fix 92 TSA warnings | Clang TSA | 5-8 | **CRITICAL** |
| Write formal verification blog | N/A | 1-2 | HIGH |
| Fix mpsverify tool issues | Lean 4 | 2-3 | HIGH |
| Install Apalache, run on existing specs | Apalache | 2-3 | MEDIUM |

### Phase B: Deep Verification (2-4 weeks)

| Item | Tool | Commits | Priority |
|------|------|---------|----------|
| Iris/Coq separation logic proofs | Iris | 10-15 | **MANDATORY** |
| Extended CBMC harnesses (6 new) | CBMC | 8-12 | HIGH |
| VeriFast annotations for MPSEvent | VeriFast | 5-8 | HIGH |
| Frama-C ACSL contracts | Frama-C | 8-12 | MEDIUM |

### Phase C: Complete Verification (4-8 weeks)

| Item | Tool | Commits | Priority |
|------|------|---------|----------|
| Complete Lean 4 proofs | Lean 4 | 10-15 | HIGH |
| Extended TLA+ specs (6 new) | TLA+ | 8-12 | HIGH |
| Temporal liveness proofs | Temporal Verifier | 5-8 | MEDIUM |
| CPAchecker analysis | CPAchecker | 5-8 | MEDIUM |
| SMACK LLVM verification | SMACK | 6-10 | LOW |
| GPUVerify kernel analysis | GPUVerify | 8-12 | LOW |

---

## Part IV: Properties to Verify

### Memory Safety
- [ ] No buffer overflows in allocator
- [ ] No use-after-free in event pool
- [ ] No null pointer dereferences
- [ ] No memory leaks

### Thread Safety
- [ ] No data races in stream pool
- [ ] Mutual exclusion in critical sections
- [ ] Lock ordering consistency
- [ ] TLS binding safety

### Protocol Correctness
- [ ] Deadlock freedom
- [ ] Starvation freedom (under fairness)
- [ ] Progress guarantees
- [ ] ABA detection completeness

### Functional Correctness
- [ ] Buffer allocation returns valid memory
- [ ] Stream assignment is deterministic per thread
- [ ] Event signaling is monotonic
- [ ] Batch queue FIFO ordering

### Liveness
- [ ] Submitted requests eventually complete
- [ ] Idle threads eventually acquire streams
- [ ] Released buffers eventually reusable

---

## Part V: Worker Directives Summary

### Immediate (N=1282+)

1. **Fix all 92 TSA warnings** (Task 4 in WORKER_DIRECTIVE.md)
2. **Write formal verification blog** (Task 1)
3. **Fix mpsverify tool issues** (Task 2)

### Next Sprint

4. **Install and run Apalache** on existing TLA+ specs
5. **Create 6 new CBMC harnesses**
6. **Begin Iris/Coq setup** (Phase 5)

### Future

7. Add Frama-C ACSL contracts to critical functions
8. Add VeriFast separation logic annotations
9. Complete Lean 4 proof of `seq_cst_race_free_conjecture`
10. Create TLA+ specs for batch queue and caches

---

### 2.11 Herdtools7 - Memory Model Testing

**Tool:** https://github.com/herd/herdtools7

**What it does:** Tool suite to test weak memory models. Includes:
- herd7: Generic simulator for weak memory models
- litmus7: Runs litmus tests on actual hardware
- diy7: Generates litmus tests from specifications

**Why it matters:** Can verify that our atomic operations behave correctly on ARM (Apple Silicon).

**Application to MPS:**
```
[ ] Write litmus tests for ABA counter operations
[ ] Verify atomic bitmask operations in stream pool
[ ] Test memory ordering of pool_alive flags
[ ] Generate stress tests from specifications
```

**Estimated:** 5-8 commits

---

### 2.12 GenMC - Stateless Model Checker

**Tool:** https://github.com/MPI-SWS/genmc

**What it does:** Stateless model checker for C programs at LLVM IR level.

**Why it matters:** Systematically explores thread interleavings to find concurrency bugs.

**Application to MPS:**
```
[ ] Check stream pool allocation/release interleavings
[ ] Verify batch queue producer/consumer races
[ ] Find races in event callback handling
```

**Worker Directive:**
```bash
# Compile to LLVM IR
clang -emit-llvm -c test_stream_pool.c -o test_stream_pool.bc

# Run GenMC
genmc test_stream_pool.bc
```

**Estimated:** 6-10 commits

---

### 2.13 P Language - State Machine Verification

**Tool:** https://github.com/p-org/P

**What it does:** State machine-based programming for formally modeling distributed systems. Used by AWS to verify S3, Windows USB driver stack.

**Why it matters:** Explicitly models communicating state machines with formal analysis backend.

**Application to MPS:**
```
[ ] Model batch queue as P state machine
[ ] Model stream pool lifecycle
[ ] Analyze interleaving of stream operations
[ ] Generate test cases from P models
```

**Estimated:** 8-12 commits

---

### 2.14 Dafny - Verification-Ready Language

**Tool:** https://github.com/dafny-lang/dafny

**What it does:** Verification-ready language with real-time verification. Can compile to C#, Go, Python, Java, JavaScript.

**Why it matters:** Could write reference implementation with proofs, then port to C++.

**Application to MPS:**
```
[ ] Write verified reference implementation of ABA counter in Dafny
[ ] Prove lock-free correctness properties
[ ] Use as specification for C++ implementation
```

**Estimated:** 5-8 commits

---

### 2.15 TLA+ Pattern Library

**Source:** https://github.com/tlaplus/Examples

**What it offers:** Collection of TLA+ specifications including:
- Mutual exclusion algorithms
- Lock-free data structures
- Distributed consensus protocols
- Resource allocation patterns

**Application to MPS:**
```
[ ] Adapt Peterson Lock pattern for stream mutex modeling
[ ] Use Readers-Writers pattern for cache access
[ ] Apply barrier synchronization for batch processing
[ ] Study lock-free set for pool implementation
```

**Worker Directive:**
```bash
# Clone TLA+ examples
git clone https://github.com/tlaplus/Examples tla-patterns

# Study relevant patterns
ls tla-patterns/specifications/
```

**Estimated:** 3-5 commits

---

### 2.16 Dat3M/Dartagnan - Memory Model Verification (SVCOMP Gold)

**Tool:** https://github.com/hernanponcedeleon/Dat3M

**What it does:** State reachability verification under weak memory models. Won SVCOMP Gold 2023, 2024. Supports ARM8, TSO, Power, RISC-V, PTX, Vulkan, OpenCL.

**Why it matters:**
- Explicit ARM8 memory model support - ideal for Apple Silicon
- Can verify litmus tests and full C programs
- Uses SMT solving for verification

**Application to MPS:**
```
[ ] Verify stream pool atomics under ARM8 memory model
[ ] Verify allocator ABA detection under ARM8
[ ] Check memory ordering of pool_alive_ flags
[ ] Verify atomic bitmask operations correctness
```

**Worker Directive:**
```bash
# Run Dartagnan on litmus test
java -jar dartagnan.jar --target=arm8 test_stream_pool.litmus

# Verify C program
java -jar dartagnan.jar --target=arm8 test_atomics.c
```

**Estimated:** 6-10 commits

---

### 2.17 RCMC - RC11 Memory Model Verification

**Tool:** https://plv.mpi-sws.org/rcmc/

**What it does:** Stateless model checking for RC11 (repaired C++11) memory model. Works directly on execution graphs, avoids redundant interleaving exploration.

**Why it matters:**
- Specifically designed for C++11 atomics
- More scalable than traditional model checking
- Can find memory model bugs missed by TSan

**Application to MPS:**
```
[ ] Verify atomic operations in stream pool
[ ] Check release-acquire synchronization in allocator
[ ] Verify memory ordering correctness in event signaling
```

**Estimated:** 5-8 commits

---

### 2.18 Verus - Verified Rust for Concurrent Systems

**Tool:** https://github.com/verus-lang/verus

**What it does:** Static verification of Rust code correctness. Supports concurrent code verification including ownership, borrowing, and thread safety proofs.

**Why it matters:**
- Could write verified reference implementation in Rust
- Proofs verified by Z3 SMT solver
- Can verify concurrent and lock-free code

**Application to MPS:**
```
[ ] Write verified ABA counter implementation in Verus
[ ] Verify lock-free pool data structure
[ ] Create verified reference for batch queue
[ ] Use as correctness specification for C++ port
```

**Estimated:** 8-12 commits

---

### 2.19 ESBMC - SMT-based Bounded Model Checker

**Tool:** https://github.com/esbmc/esbmc

**What it does:** Comprehensive bounded model checker supporting C/C++, CUDA, and more. Detects: buffer overflows, null dereference, memory leaks, deadlocks, data races, atomicity violations.

**Why it matters:**
- Explores thread interleavings systematically
- Checks lock acquisition ordering
- Can verify actual MPS C++ code directly

**Application to MPS:**
```
[ ] Check stream pool for deadlocks
[ ] Verify allocator for data races
[ ] Check event pool atomicity
[ ] Verify lock ordering in all mutex-protected code
```

**Worker Directive:**
```bash
# Run ESBMC with concurrency checking
esbmc --no-bounds-check --memory-leak-check --deadlock-check \
      --context-bound 2 MPSStream.mm
```

**Estimated:** 6-10 commits

---

### 2.20 K-C (c-semantics) - Formal C Semantics

**Tool:** https://github.com/kframework/c-semantics

**What it does:** Executable formal semantics of C using the K framework. Precisely defines C behavior including undefined behavior detection.

**Why it matters:**
- Can detect subtle undefined behavior
- Based on formal K framework semantics
- Generates deterministic and non-deterministic versions

**Application to MPS:**
```
[ ] Check MPS code for undefined behavior
[ ] Verify expression evaluation order in atomic sequences
[ ] Detect UB in pointer arithmetic
```

**Worker Directive:**
```bash
# Run kcc on MPS code
kcc MPSAllocator.mm -o test_alloc
./test_alloc  # Detects UB during execution
```

**Estimated:** 4-6 commits

---

### 2.21 GPU Litmus - GPU Memory Model Testing

**Tool:** https://fastpl.doc.ic.ac.uk/gpu-litmus/

**What it does:** Methodology for testing concurrent behavior on GPUs. Generates litmus tests that probe memory model behavior across GPU architectures.

**Why it matters:**
- Can verify Metal's memory model empirically
- Tests actual hardware behavior, not just spec
- Reveals weak behaviors under concurrent execution

**Application to MPS:**
```
[ ] Adapt GPU litmus methodology for Metal
[ ] Test memory ordering between CPU and GPU
[ ] Verify Metal's threadgroup synchronization
[ ] Create litmus tests for MPS kernel operations
```

**Note:** Would need to adapt methodology for Metal, as original targets CUDA/OpenCL.

**Estimated:** 10-15 commits

---

### 2.22 SPIN Model Checker - Promela Verification

**Tool:** https://spinroot.com/

**What it does:** Classic model checker for concurrent systems using Promela modeling language. Won ACM System Software Award 2002. Has been verifying concurrent systems since 1980.

**Why it matters:**
- Mature and battle-tested (40+ years)
- Well-documented patterns for mutex, semaphores, barriers
- Can generate C code from verified models

**Application to MPS:**
```
[ ] Model stream pool protocol in Promela
[ ] Verify mutex acquisition patterns
[ ] Generate test harnesses from SPIN counterexamples
[ ] Compare with TLA+ model checking results
```

**Worker Directive:**
```promela
/* Stream pool mutual exclusion in Promela */
mtype = { FREE, ACQUIRED, RELEASING };
mtype slot_state[32] = FREE;
bool slot_owner[32] = false;

proctype Thread(byte id) {
    byte slot;
    do
    :: atomic {
         slot_state[slot] == FREE ->
         slot_state[slot] = ACQUIRED;
         slot_owner[slot] = id
       }
    :: slot_state[slot] == ACQUIRED && slot_owner[slot] == id ->
       slot_state[slot] = RELEASING;
       slot_state[slot] = FREE
    od
}
```

**Estimated:** 5-8 commits

---

### 2.23 Loom - C11 Memory Model Testing for Rust

**Tool:** https://github.com/tokio-rs/loom

**What it does:** Exhaustive permutation testing under C11 memory model. Runs tests many times with different thread interleavings.

**Why it matters:**
- Can verify C11 atomic behaviors systematically
- State reduction avoids combinatorial explosion
- Verified implementations can guide C++ port

**Application to MPS:**
```
[ ] Write verified Rust reference for ABA counter using Loom
[ ] Verify atomic bitmask operations
[ ] Test release-acquire synchronization patterns
[ ] Use as specification for C++ atomics
```

**Estimated:** 6-10 commits

---

### 2.24 Shuttle - AWS Randomized Concurrency Testing

**Tool:** https://github.com/awslabs/shuttle

**What it does:** Randomized thread scheduling to find concurrency bugs. Controls scheduling and enables deterministic reproduction.

**Why it matters:**
- Scales to larger test cases than exhaustive testing
- Over 99.9999% probability of finding bugs in examples
- Deterministically reproduces failures

**Application to MPS:**
```
[ ] Write Rust reference implementation with Shuttle
[ ] Test pool allocation under random scheduling
[ ] Verify batch queue producer/consumer
[ ] Document any bugs found for C++ implementation
```

**Estimated:** 5-8 commits

---

### 2.25 Relacy - C++ Synchronization Algorithm Verifier

**Tool:** https://github.com/dvyukov/relacy

**What it does:** Race detector specifically for synchronization algorithms in C++. Verifies algorithms under relaxed memory models.

**Why it matters:**
- Directly applicable to C++ code
- Specifically designed for synchronization verification
- Can verify custom atomic patterns

**Application to MPS:**
```
[ ] Verify ABA counter algorithm
[ ] Test atomic bitmask operations in stream pool
[ ] Verify double-check locking patterns
[ ] Check release-acquire sequences in allocator
```

**Worker Directive:**
```cpp
// Relacy test for ABA counter
#include <relacy/relacy_std.hpp>

struct aba_counter_test : rl::test_suite<aba_counter_test, 2> {
    std::atomic<uint64_t> counter;

    void thread(unsigned id) {
        uint64_t old_val = counter.load($);
        // ... simulate ABA scenario
        bool success = counter.compare_exchange_strong(old_val, new_val, $);
    }
};
```

**Estimated:** 6-10 commits

---

### 2.26 Kani - AWS Bit-Precise Model Checker for Rust

**Tool:** https://github.com/model-checking/kani

**What it does:** Bit-precise model checker for Rust using symbolic execution. Automatically checks for undefined behavior and custom assertions.

**Why it matters:**
- Proves program properties across all inputs
- Can verify unsafe Rust code blocks
- Bit-level precision catches subtle bugs

**Application to MPS:**
```
[ ] Write verified reference implementation in Rust
[ ] Verify buffer index calculations
[ ] Check pointer arithmetic safety
[ ] Prove overflow safety in size calculations
```

**Worker Directive:**
```rust
#[kani::proof]
fn verify_slot_calculation() {
    let slot: usize = kani::any();
    kani::assume(slot < 32);
    let mask = 1u32 << slot;
    // Prove no overflow
    assert!(mask != 0);
}
```

**Estimated:** 8-12 commits

---

### 2.27 Infer - Facebook Static Analysis

**Tool:** https://github.com/facebook/infer

**What it does:** Static analysis for Java, C++, Objective-C, and C. Detects null pointer dereference, memory leaks, race conditions.

**Why it matters:**
- Supports Objective-C++ (MPS code is Obj-C++)
- Can detect races statically
- Integrates with CI workflows

**Application to MPS:**
```
[ ] Run Infer on MPS Objective-C++ code
[ ] Check for null pointer dereferences
[ ] Detect potential race conditions
[ ] Verify memory safety
```

**Worker Directive:**
```bash
# Run Infer on MPS code
infer run -- clang -c aten/src/ATen/mps/MPSStream.mm

# Analyze specific issues
infer analyze --racerd
```

**Estimated:** 3-5 commits

---

### 2.28 QuickCheck - Property-Based Testing for Rust

**Tool:** https://github.com/BurntSushi/quickcheck

**What it does:** Generates random test inputs and automatically shrinks failing inputs to find minimal counterexamples.

**Why it matters:**
- Finds edge cases human testers miss
- Shrinking helps identify root cause of failures
- Can test properties of concurrent data structures

**Application to MPS:**
```
[ ] Write property tests for Rust reference implementations
[ ] Test invariants: "pool never overflows", "slot always valid"
[ ] Use shrinking to find minimal failure scenarios
```

**Estimated:** 3-5 commits

---

### 2.29 PropTest - Advanced Property-Based Testing

**Tool:** https://github.com/proptest-rs/proptest

**What it does:** Flexible property-based testing with composable strategies. Inspired by Python's Hypothesis.

**Why it matters:**
- More flexible than QuickCheck
- Per-value generation and shrinking
- Better input constraints for complex types

**Application to MPS:**
```
[ ] Test concurrent scenarios in Rust reference
[ ] Define strategies for buffer sizes, thread counts
[ ] Verify allocator invariants across random inputs
```

**Estimated:** 3-5 commits

---

### 2.30 AFL - Coverage-Guided Fuzzing

**Tool:** https://github.com/google/AFL

**What it does:** Genetic algorithm-based fuzzer that uses instrumentation to guide mutation toward new code paths.

**Why it matters:**
- Finds security vulnerabilities automatically
- Coverage-guided mutation is highly effective
- Can run across multiple cores

**Application to MPS:**
```
[ ] Fuzz MPS API inputs
[ ] Find crashes in buffer handling
[ ] Generate corpus of interesting inputs for regression
```

**Worker Directive:**
```bash
# Build with AFL instrumentation
CC=afl-clang-fast CXX=afl-clang-fast++ cmake ..

# Run fuzzer
afl-fuzz -i input_corpus -o findings ./mps_test_binary @@
```

**Estimated:** 5-8 commits

---

### 2.31 LibFuzzer - LLVM In-Process Fuzzing

**Tool:** https://llvm.org/docs/LibFuzzer.html

**What it does:** Coverage-guided evolutionary fuzzing engine that runs in-process. Integrates with LLVM sanitizers.

**Why it matters:**
- Fast (no process restart per test)
- Combines with ASAN, UBSAN, MSAN
- Supports custom mutators and dictionaries

**Application to MPS:**
```
[ ] Create fuzz targets for MPS APIs
[ ] Combine with AddressSanitizer to find memory bugs
[ ] Combine with ThreadSanitizer to find races
```

**Worker Directive:**
```cpp
// Fuzz target for buffer allocation
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < sizeof(size_t)) return 0;
    size_t alloc_size = *reinterpret_cast<const size_t*>(data);
    // Test allocation with fuzzed size
    auto buffer = allocate_buffer(alloc_size);
    return 0;
}
```

**Estimated:** 5-8 commits

---

### 2.32 KLEE - Symbolic Execution

**Tool:** https://github.com/klee/klee

**What it does:** Symbolic virtual machine that executes programs with symbolic values to explore all paths.

**Why it matters:**
- Systematically explores all execution paths
- Generates test cases that cover each path
- Can find bugs unreachable by fuzzing

**Application to MPS:**
```
[ ] Symbolically execute buffer allocation paths
[ ] Generate tests covering all error conditions
[ ] Verify path coverage of critical functions
```

**Worker Directive:**
```bash
# Compile to LLVM bitcode
clang -emit-llvm -c -g test_allocator.c -o test_allocator.bc

# Run KLEE
klee --emit-all-errors test_allocator.bc
```

**Estimated:** 6-10 commits

---

## Part VI: Tool Installation Reference

```bash
# TLA+ Tools (already have)
brew install openjdk
export JAVA_HOME=$(/usr/libexec/java_home)

# Apalache - Symbolic TLA+ model checking
brew install apalache

# Lean 4 (already have)
brew install elan-init
elan default leanprover/lean4:stable

# Coq + Iris
opam install coq coq-iris

# CBMC (already have)
brew install cbmc

# Frama-C
opam install frama-c

# VeriFast
brew install verifast

# CPAchecker
# Download from https://cpachecker.sosy-lab.org/

# SMACK
# Build from source: https://github.com/smackers/smack

# Temporal Verifier
cargo install temporal-verifier

# GPUVerify
# Build from source: https://github.com/mc-imperial/gpuverify

# Herdtools7 - Memory model testing
opam install herdtools7

# GenMC - Stateless model checker
# Build from source: https://github.com/MPI-SWS/genmc

# P Language - State machine verification
dotnet tool install --global P

# Dafny - Verification-ready language
brew install dafny

# TLA+ Examples (patterns library)
git clone https://github.com/tlaplus/Examples tla-patterns

# Dat3M/Dartagnan - Memory model verification (ARM8, TSO, Power)
# Download from https://github.com/hernanponcedeleon/Dat3M/releases
# Requires Java 17+

# RCMC - RC11 memory model verification
# Build from source: https://github.com/MPI-SWS/rcmc (requires LLVM)

# Verus - Verified Rust
# Install via rustup: https://github.com/verus-lang/verus

# ESBMC - SMT-based bounded model checker
brew install esbmc

# K-C (c-semantics) - Formal C semantics
# Build from source: https://github.com/kframework/c-semantics
# Requires K framework

# GPU Litmus methodology - no install, adapt methodology

# SPIN - Classic model checker (Iteration 3)
brew install spin

# Loom - Rust C11 memory model testing (add to Cargo.toml)
# loom = "0.7"

# Shuttle - AWS randomized concurrency testing (add to Cargo.toml)
# shuttle = "0.7"

# Relacy - C++ race detector
# Build from source: https://github.com/dvyukov/relacy

# Kani - AWS bit-precise Rust verification
cargo install --locked kani-verifier
cargo kani setup

# Infer - Facebook static analysis
brew install infer

# QuickCheck (Rust) - Property-based testing (add to Cargo.toml)
# quickcheck = "1.0"

# PropTest (Rust) - Property-based testing (add to Cargo.toml)
# proptest = "1.0"

# AFL - Coverage-guided fuzzing
brew install afl-fuzz

# LibFuzzer - LLVM fuzzing (built into Clang)
# clang -fsanitize=fuzzer,address -g fuzz_target.c

# KLEE - Symbolic execution
# Build from source: https://klee.github.io/build-llvm13/
```

---

## Appendix: Verification Coverage Matrix

### Primary Tools
| Component | TLA+ | CBMC | Lean | TSA | Iris | Frama-C | VeriFast |
|-----------|------|------|------|-----|------|---------|----------|
| MPSStream | ✅ | ✅ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ |
| MPSAllocator | ✅ | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| MPSEvent | ✅ | ✅ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ |
| MPSBatchQueue | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSGraphCache | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSCommandBuffer | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| TLS Binding | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Fork Safety | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSKernelCache | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSFullSystem | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

### Additional Tools - Iteration 1 (Discovered in Ralph Loop)
| Component | Apalache | Herdtools7 | GenMC | P Lang | Dafny | CPAchecker |
|-----------|----------|------------|-------|--------|-------|------------|
| MPSStreamPool | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSAllocator | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSEvent | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSFullSystem | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSTLSBinding | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSCommandBuffer | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSBatchQueue | ⚠️ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSForkHandler | ⚠️ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSGraphCache | ⚠️ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSKernelCache | ⚠️ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Atomics | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Memory Order | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

### Additional Tools - Iteration 2 (Memory Model & SMT)
| Component | Dat3M | RCMC | Verus | ESBMC | K-C | GPU Litmus |
|-----------|-------|------|-------|-------|-----|------------|
| MPSStream | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | N/A |
| MPSAllocator | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | N/A |
| MPSEvent | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | N/A |
| Atomics (ARM8) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | N/A |
| Memory Order | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | N/A |
| Metal Shaders | N/A | N/A | N/A | N/A | N/A | ⬜ |

### Additional Tools - Iteration 3 (Classic & Reference Impl)
| Component | SPIN | Loom | Shuttle | Relacy | Kani | Infer |
|-----------|------|------|---------|--------|------|-------|
| MPSStream | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSAllocator | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSEvent | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Sync Algorithms | ⬜ | ⬜ | ⬜ | ⬜ | N/A | N/A |
| Rust Reference | N/A | ⬜ | ⬜ | N/A | ⬜ | N/A |
| Obj-C++ Static | N/A | N/A | N/A | N/A | N/A | ⬜ |

### Additional Tools - Iteration 4 (Testing & Fuzzing)
| Component | QuickCheck | PropTest | AFL | LibFuzzer | KLEE |
|-----------|------------|----------|-----|-----------|------|
| MPSStream | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSAllocator | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MPSEvent | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Rust Reference | ⬜ | ⬜ | N/A | N/A | N/A |
| API Inputs | N/A | N/A | ⬜ | ⬜ | ⬜ |
| Path Coverage | N/A | N/A | N/A | N/A | ⬜ |

Legend: ✅ Complete | ⚠️ In Progress | ⬜ Not Started | N/A Not Applicable

---

## Total Verification Tools Cataloged: 32

### Implemented (6)
1. TLA+ (TLC model checker) ✅
2. Lean 4 theorem prover ✅
3. CBMC bounded model checker ✅
4. Clang TSA static analysis ✅ (N=1310)
5. Iris/Coq separation logic ✅ (N=1298)
6. Apalache symbolic TLA+ ✅ (N=1346)

### Discovered - Iteration 1 (Now Partially Implemented)
7. Frama-C/ACSL contracts ⬜
8. VeriFast separation logic ⬜
9. CPAchecker configurable analysis ⬜
10. Temporal Verifier liveness ⬜
11. SMACK LLVM verification ⬜
12. GPUVerify kernel verification ⬜
13. Herdtools7 memory model testing ⬜
14. GenMC stateless model checking ⬜
15. P Language state machine verification ⬜

### Discovered - Iteration 2 (6)
16. Dat3M/Dartagnan - Memory model verification (ARM8, TSO, Power) ⬜
17. RCMC - RC11 repaired C++11 memory model verification ⬜
18. Verus - Verified Rust with concurrent code support ⬜
19. ESBMC - SMT-based bounded model checker (deadlocks, races, atomicity) ⬜
20. K-C (c-semantics) - Formal C semantics with undefined behavior detection ⬜
21. GPU Litmus - GPU memory model testing methodology ⬜

### Discovered - Iteration 3 (6)
22. SPIN - Classic model checker using Promela (ACM Award 2002) ⬜
23. Loom - C11 memory model exhaustive testing for Rust ⬜
24. Shuttle - AWS randomized concurrency testing for Rust ⬜
25. Relacy - C++ race detector for synchronization algorithms ⬜
26. Kani - AWS bit-precise model checker for Rust ⬜
27. Infer - Facebook static analysis (races, null safety, memory) ⬜

### Discovered - Iteration 4 (5 - Testing & Fuzzing)
28. QuickCheck (Rust) - Property-based random testing with shrinking ⬜
29. PropTest (Rust) - Advanced property-based testing framework ⬜
30. AFL - Coverage-guided fuzzing for C/C++ (genetic algorithm) ⬜
31. LibFuzzer - LLVM in-process coverage-guided fuzzing ⬜
32. KLEE - Symbolic execution for LLVM bitcode ⬜

### Foundational Infrastructure (Used by Above Tools)
- Z3 - SMT solver (powers CBMC, ESBMC, Kani, Apalache, Verus)
- CVC5 - Alternative SMT solver
- LLVM - Compiler infrastructure (powers LibFuzzer, KLEE, SMACK, GenMC)

### Iteration 5 Assessment: SATURATION REACHED
After 5 iterations, we have cataloged **32 verification tools** across all major categories:
- **Model Checking**: TLA+, Apalache, SPIN, CBMC, ESBMC, GenMC
- **Theorem Proving**: Lean 4, Coq/Iris
- **Memory Models**: Herdtools7, Dat3M, RCMC, Loom, Relacy
- **Static Analysis**: Clang TSA, Infer, Frama-C, VeriFast
- **Rust Verification**: Verus, Kani, Loom, Shuttle, QuickCheck, PropTest
- **Fuzzing/Testing**: AFL, LibFuzzer, KLEE

Tools searched but not applicable:
- Coyote (C# only), Manticore (EVM/binary), angr (binary analysis)

**RALPH LOOP COMPLETE** - No significant new verification opportunities remaining.

---

**Last Updated:** 2025-12-20 (N=1361: Added type annotations to MPSBatchQueue, MPSForkHandler, MPSGraphCache, MPSKernelCache)
**Next Review:** Maintenance mode - 10 TLA+ specs complete, all 10 have Apalache configs and type annotations
