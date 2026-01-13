# Formal Verification Enhancement and Scaling Fix Plan

**Created**: 2025-12-17 (N=1035+)
**Goal**: Achieve correctness guarantees that enable confident threading changes, then fix 4-8 thread scaling.

---

## Executive Summary

The current formal verification proves **safety** but not **scalability**. We need to:
1. Add properties that SHOULD hold in an optimal implementation (they may fail now, revealing issues)
2. Fix broken verification infrastructure (TSA not running)
3. Complete CBMC verification (3/4 harnesses incomplete)
4. Use verified properties to guide and validate scaling fixes

---

## Part 1: Formal Verification Enhancement

### 1.1 Critical Gaps in Current Verification

| Component | Status | Gap |
|-----------|--------|-----|
| TLA+ Specs | ✅ Pass | No scalability properties |
| CBMC | ⚠️ 1/4 verified | 3/4 hit unwinding limits |
| TSA | ❌ Broken | cmake_macros.h missing |
| Lean | ⚠️ 1 sorry | seq_cst conjecture unproven |
| Spec-Code correspondence | ❌ None | No proof specs match code |

### 1.2 New Properties to Add (May Fail - That's The Point!)

These properties SHOULD hold in an optimal implementation. If they fail, they reveal design issues.

#### Property 1: Parallel Progress (ADD TO MPSAllocator.tla)

```tla
(* SCALABILITY PROPERTY: Multiple threads CAN hold different locks simultaneously *)
(* In current impl with global m_mutex, this is FALSE - only 1 thread at a time *)
ParallelLockHolding ==
    \* At least 2 threads can be in different critical sections simultaneously
    \* This requires per-pool or per-thread locks, not a global m_mutex
    \E t1, t2 \in Threads:
        t1 # t2 /\
        (HoldsPoolMutex(t1) \/ HoldsMMutex(t1)) /\
        (HoldsPoolMutex(t2) \/ HoldsMMutex(t2))

(* EXPECTED: This property is VIOLATED by current design *)
(* If we shard m_mutex per-pool or per-thread, it would PASS *)
```

#### Property 2: No Global Serialization (ADD TO MPSAllocator.tla)

```tla
(* SCALABILITY PROPERTY: No single lock serializes ALL operations *)
(* Currently m_mutex violates this - every alloc/free takes it *)
NoGlobalSerializer ==
    \* There should be parallel paths through the allocator
    \* Currently FALSE because m_mutex is taken on every operation
    ~(\A t \in Threads: pc[t] \in {"alloc_lock_m", "getptr_lock_m1", "getptr_lock_m2"}
                         => HoldsMMutex(t))

(* To make this TRUE: shard m_mutex or use lock-free structures *)
```

#### Property 3: Bounded Lock Wait (ADD TO MPSAllocator.tla)

```tla
VARIABLE lock_wait_count  \* Per-thread counter of lock acquisition attempts

(* SCALABILITY PROPERTY: Threads don't wait unboundedly for locks *)
BoundedLockWait ==
    \A t \in Threads: lock_wait_count[t] <= 100

(* Model lock contention by incrementing counter when CanAcquire* is FALSE *)
```

#### Property 4: Lock Hierarchy (ADD - Prevents Deadlock Under Scaling)

```tla
(* CORRECTNESS PROPERTY: Locks acquired in consistent order *)
(* Current code: m_mutex -> pool_mutex (but also pool_mutex -> m_mutex in some paths!) *)
LockHierarchy ==
    \A t \in Threads:
        \* If holding pool_mutex, must not try to acquire m_mutex
        HoldsPoolMutex(t) => ~(pc[t] \in {"alloc_lock_m", "getptr_lock_m1"})

(* Check if current impl maintains this - may reveal deadlock potential *)
```

### 1.3 CBMC Deep Verification

Current: `--unwind=3` (too shallow)
Target: `--unwind=10` minimum, ideally `--unwind=20`

```bash
# Run with deeper unwinding
cbmc harnesses/alloc_free_harness.c \
    -I models -I stubs \
    --unwind 10 \
    --pointer-check \
    --bounds-check \
    --unwinding-assertions

# Add lock contention bound assertion
// In harness:
unsigned lock_attempts = 0;
while (!atomic_try_lock(&m_mutex)) {
    lock_attempts++;
    __CPROVER_assert(lock_attempts < 100, "Bounded lock wait");
}
```

### 1.4 Fix TSA Infrastructure

The TSA analysis is failing because it can't find cmake_macros.h. Need to:

1. Create standalone TSA check that doesn't need full PyTorch build
2. Or: Run TSA as part of actual PyTorch compilation

```bash
# Option 1: Standalone with mock headers
clang++ -fsyntax-only \
    -Wthread-safety \
    -Wthread-safety-negative \
    -I mps-verify/verification/static/mock_headers \
    aten/src/ATen/mps/MPSAllocator.mm

# Option 2: Add to PyTorch build flags
cmake -DCMAKE_CXX_FLAGS="-Wthread-safety" ...
```

### 1.5 Lean Proof Completion

Complete the one `sorry`:

```lean
-- In MemoryModel.lean:241
theorem seq_cst_race_free_conjecture (events : List MemoryEvent) :
    AllSeqCst events → isRaceFree events := by
  intro h_seq_cst
  -- Need to prove: seq_cst ordering implies total order implies no races
  unfold isRaceFree
  intro e1 e2 h_diff h_conflict
  -- seq_cst provides total order, so one happens-before the other
  sorry  -- Requires seq_cst total order model
```

---

## Part 2: Scaling Fix - 4-8 Thread Regression

### 2.1 Primary Hypothesis: Global m_mutex Contention

**Evidence:**
- m_mutex taken on: malloc, free, getSharedBufferPtr (twice!), recordStream, setBufferShape, getBufferShape, recordEvents
- Every memory operation serializes on ONE lock
- 8 threads = 8x contention on single lock

**Fix Strategy: Shard the Allocator**

```cpp
// BEFORE: Single global allocator
class MPSHeapAllocatorImpl {
    std::recursive_mutex m_mutex;  // GLOBAL - all threads serialize
    ...
};

// AFTER: Per-stream or sharded allocator
class MPSHeapAllocatorImpl {
    // Option A: Per-stream allocator state
    struct PerStreamState {
        std::recursive_mutex mutex;
        ska::flat_hash_map<void*, BufferBlock*> allocated_buffers;
    };
    std::array<PerStreamState, MAX_STREAMS> per_stream_state_;

    // Option B: Sharded by buffer address
    static constexpr int NUM_SHARDS = 16;
    std::array<std::recursive_mutex, NUM_SHARDS> shard_mutexes_;
    size_t get_shard(void* ptr) { return hash(ptr) % NUM_SHARDS; }
};
```

**Verification:** After sharding, the `ParallelLockHolding` TLA+ property should PASS.

### 2.2 Hypothesis 2: Double-Check Pattern Lock Thrashing

**Evidence:**
- getSharedBufferPtr does: lock m_mutex -> unlock -> lock pool_mutex -> lock m_mutex AGAIN
- This double acquisition of m_mutex per operation doubles contention

**Fix Strategy: Single-Lock Double-Check**

```cpp
// BEFORE: Two m_mutex acquisitions
void* getSharedBufferPtr(void* ptr) {
    BufferBlock* block;
    uint32_t saved_use_count;
    {
        std::lock_guard lock(m_mutex);  // LOCK 1
        block = find_buffer(ptr);
        saved_use_count = block->use_count;
    }  // UNLOCK 1

    std::lock_guard pool_lock(block->pool->pool_mutex);
    {
        std::lock_guard lock(m_mutex);  // LOCK 2 (re-acquire!)
        if (block->use_count != saved_use_count) return nullptr;  // ABA check
    }
    return block->buf;
}

// AFTER: Hold m_mutex throughout (shorter total time due to no re-acquire)
void* getSharedBufferPtr(void* ptr) {
    std::lock_guard lock(m_mutex);
    BufferBlock* block = find_buffer(ptr);
    if (!block) return nullptr;

    uint32_t saved_use_count = block->use_count.load(acquire);

    // Take pool lock while holding m_mutex (need to verify lock order)
    std::lock_guard pool_lock(block->pool->pool_mutex);

    // ABA check under both locks
    if (block->use_count.load(acquire) != saved_use_count) return nullptr;
    return block->buf;
}
```

**Note:** Must verify this doesn't introduce deadlock via TLA+ LockHierarchy property.

### 2.3 Hypothesis 3: dispatch_sync Serialization

**Evidence:**
- MPSStream uses dispatch_sync to serial queues
- If multiple threads hit same stream, they serialize
- Even with 8 streams, uneven distribution causes bunching

**Investigation:**
```cpp
// Add instrumentation to measure dispatch_sync contention
std::atomic<uint64_t> dispatch_sync_waits{0};

void dispatch_sync_with_rethrow(dispatch_queue_t queue, dispatch_block_t block) {
    auto start = std::chrono::high_resolution_clock::now();
    dispatch_sync(queue, block);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    if (elapsed > 1ms) {
        dispatch_sync_waits++;
    }
}
```

**Fix (if confirmed):**
- Use dispatch_async where possible
- Better stream distribution (currently round-robin)
- Stream affinity per thread

### 2.4 Hypothesis 4: GPU Command Buffer Depth Limit

**Evidence:**
- Metal may limit in-flight command buffers
- At 8 threads submitting rapidly, queue may back up
- Would cause blocking, not just saturation

**Investigation:**
```objc
// Check command buffer queue depth
NSUInteger maxCommandBufferCount = [device maxCommandBufferCount];
// Typical: 64 or 256

// Add tracking
std::atomic<int> in_flight_command_buffers{0};
// Increment on commit, decrement in completion handler
```

**Fix (if confirmed):**
- Batch operations into fewer command buffers
- Wait for completion before submitting more
- Use MTLSharedEvent for finer-grained sync

---

## Part 3: Implementation Order

### Phase 1: Verification Enhancement (2-3 commits)

1. **Add scalability properties to TLA+ specs**
   - ParallelLockHolding
   - NoGlobalSerializer
   - BoundedLockWait
   - LockHierarchy

2. **Run TLC - expect failures** (these failures are INFORMATIVE)
   - Document which properties fail
   - Use failures to confirm bottleneck locations

3. **Fix CBMC unwinding**
   - Increase to --unwind=10
   - Add bounded lock wait assertions

4. **Fix TSA infrastructure**
   - Create mock headers for standalone analysis
   - Or integrate into build

### Phase 2: Scaling Fix (3-5 commits)

1. **Instrument current code**
   - Add lock wait time tracking
   - Add dispatch_sync contention tracking
   - Profile under 8-thread load

2. **Implement allocator sharding**
   - Per-stream or hash-sharded m_mutex
   - Update TLA+ spec to match
   - Verify ParallelLockHolding now PASSES

3. **Fix double-check lock thrashing**
   - Single-pass double-check pattern
   - Verify with LockHierarchy property

4. **Address remaining bottlenecks**
   - Based on instrumentation data

### Phase 3: Verification Confirmation (1-2 commits)

1. **Re-run all TLA+ with new properties**
   - All scalability properties should now PASS

2. **Re-run CBMC with deeper unwinding**
   - All 4/4 harnesses should verify

3. **Benchmark 8-thread efficiency**
   - Target: ≥50% efficiency
   - Document improvement

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| 8-thread efficiency | 14-27% | ≥50% |
| TLA+ ParallelLockHolding | ❌ FAIL | ✅ PASS |
| TLA+ NoGlobalSerializer | ❌ FAIL | ✅ PASS |
| CBMC harnesses verified | 1/4 | 4/4 |
| TSA analysis | ❌ Broken | ✅ Running |
| Lean sorry count | 1 | 0 |

---

## Worker Directive

**Priority 1:** Add scalability properties to TLA+ specs and run TLC. Document failures.

**Priority 2:** Fix CBMC --unwind depth to 10, verify all 4 harnesses.

**Priority 3:** Implement allocator sharding to eliminate global m_mutex.

**Priority 4:** Re-verify with TLA+ - scalability properties should now pass.

**Priority 5:** Benchmark and verify 8-thread efficiency ≥50%.

---

## Files to Modify

- `mps-verify/specs/MPSAllocator.tla` - Add scalability properties
- `mps-verify/specs/MPSAllocator.cfg` - Add new invariants to check
- `mps-verify/verification/cbmc/harnesses/*.c` - Add bounded wait assertions
- `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm` - Shard allocator
- `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.h` - Per-stream state

---

## Appendix: Property Formalization Details

### ParallelLockHolding (Full TLA+)

```tla
(* To add to MPSAllocator.tla after line 457 *)

(* SCALABILITY: Multiple threads can hold different locks simultaneously *)
(* This property is FALSE with single m_mutex, TRUE with sharded locks *)
ParallelLockHolding ==
    \/ \* Either pool is dead (terminal state)
       ~allocator_alive
    \/ \* Or there exist two threads that could both be in critical sections
       \* with different locks (currently impossible with single m_mutex)
       \E t1, t2 \in Threads:
           t1 # t2 /\
           \* Different pool mutexes are independent
           (\E p1, p2 \in {1, 2}: p1 # p2 /\  \* Assuming 2 pools
               HoldsPoolMutexForPool(t1, p1) /\ HoldsPoolMutexForPool(t2, p2))

(* Note: Current spec has single pool_mutex_holder, need to extend to per-pool *)
```

### Bounded Lock Wait Counter Model

```tla
(* Add to VARIABLES *)
lock_wait_count,  \* Thread -> Nat, count of lock acquisition attempts

(* Modify lock acquisition actions to track waits *)
AllocLockM(t) ==
    /\ pc[t] = "alloc_lock_m"
    /\ \/ /\ CanAcquireMMutex(t)
          /\ m_mutex_holder' = t
          /\ pc' = [pc EXCEPT ![t] = "alloc_find"]
          /\ UNCHANGED lock_wait_count
       \/ /\ ~CanAcquireMMutex(t)
          /\ lock_wait_count' = [lock_wait_count EXCEPT ![t] = @ + 1]
          /\ UNCHANGED <<pc, m_mutex_holder>>
    /\ UNCHANGED <<...other vars...>>

(* Scalability invariant *)
BoundedLockWait ==
    \A t \in Threads: lock_wait_count[t] <= 10  \* Will fail with contention
```
