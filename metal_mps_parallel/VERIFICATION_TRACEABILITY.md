# MPS Verification Traceability Map

**Purpose:** Maps code anchors to specifications, harnesses, and properties.
**Audience:** WORKER AIs and engineers reviewing verification coverage.
**Reference:** `FORMAL_VERIFICATION_PARAGON_DESIGN.md` Section 4.1-4.2

---

## How to Use This Document

1. **Find Code:** Locate the C++ file/function you're modifying
2. **Check Specs:** See which TLA+ specs and CBMC harnesses cover it
3. **Verify Properties:** Confirm which properties must hold after your change
4. **Update Evidence:** After running verification, cite results here

---

## Property Catalog

### Safety Properties (Must Always Hold)

| Property ID | Name | Description | Spec/Harness | Assumptions |
|-------------|------|-------------|--------------|-------------|
| **SP.001** | NoUseAfterFree | No thread accesses stream/buffer after pool/allocator destruction | MPSStreamPool.tla, MPSAllocator.tla | Pool destruction only occurs during static destruction |
| **SP.002** | ABADetectionSound | use_count check prevents ABA race in buffer reuse | MPSAllocator.tla, aba_detection_harness.c | use_count never wraps (32-bit, ~4B allocations) |
| **SP.003** | NoDoubleFree | Buffer cannot be freed twice | MPSAllocator.tla, alloc_free_harness.c | Ownership tracked via buffer_owner |
| **SP.004** | MutexExclusivity | Only one thread holds a mutex at any time | All TLA+ specs | std::mutex semantics correct |
| **SP.005** | TLSBindingValid | TLS stream binding is always within valid bounds | MPSStreamPool.tla | NumStreams >= 2 |
| **SP.006** | ForkInvalidatesTLS | Fork clears all TLS bindings | MPSStreamPool.tla | Fork handler registered |
| **SP.007** | CallbackSafety | Callbacks never access dead state | MPSEvent.tla | shared_ptr semantics correct |
| **SP.008** | BufferConsistency | in_use=TRUE implies buffer is allocated | MPSAllocator.tla | None |
| **SP.009** | PoolsDisjoint | Available and allocated buffer sets are disjoint | MPSAllocator.tla | None |

### Liveness Properties (Eventually Hold)

| Property ID | Name | Description | Spec/Harness |
|-------------|------|-------------|--------------|
| **LP.001** | EventuallyCompletes | Operations eventually complete | MPSStreamPool.tla, MPSAllocator.tla |
| **LP.002** | NoDeadlock | System never deadlocks (some action enabled) | All TLA+ specs |

### Structural Properties (Pattern Conformance)

| Property ID | Name | Pattern | Code Locations |
|-------------|------|---------|----------------|
| **ST.001** | PoolAliveGuards | g_pool_alive guards TLS cleanup + slot recycle | MPSStream.mm (ThreadStreamSlot dtor, releaseSlotIfPoolAlive) |
| **ST.002** | ABADoubleCheck | use_count capture + verify pattern | MPSAllocator.mm:1064-1071 |
| **ST.003** | EventLifetimeSafety | shared_ptr-backed in-use events + explicit notify queue | MPSEvent.h (m_in_use_events), MPSEvent.mm (notifyLocked, elapsedTime) |
| **ST.004** | NoWaitWhileHolding | No waitUntilCompleted() while holding mutex | All MPS files |
| **ST.005** | EncoderLifetime | No commandEncoder capture outside dispatch blocks | All MPS files |
| **ST.006** | LockOrderConsistency | Mutex lock order: m_mutex -> pool_mutex | MPSAllocator.mm |
| **ST.007** | BlockCallbackLifetime | Block callbacks must not capture 'this' unsafely | MPSEvent.mm |
| **ST.008** | GlobalSerializationDetection | Identifies global mutex usage that may cause serialization (Phase 3 aspirational) | MPSStream.mm (getGlobalMetalEncodingMutex), MPSBatchQueue.mm (g_batch_queue_mutex) |
| **ST.009** | BoundedWaitInfrastructure | Verifies bounded wait detection exists (TLA+ spec + runtime test) (Phase 3) | specs/MPSStreamPoolBoundedWait.tla, tests/test_bounded_wait.py, mps-verify/bounded_wait_results.json |
| **ST.010** | ParallelCriticalSectionExists | Verifies design permits parallel progress (TLA+ existence check + runtime test) (Phase 3) | specs/MPSStreamPoolParallel.tla, tests/test_parallel_progress.py, mps-verify/parallel_progress_results.json |
| **ST.011** | RecordStreamCrossStreamProtocol | Verifies recordStream() event-based cross-stream lifetime protocol (Opportunity Map B1.1) | specs/MPSRecordStream.tla, mps-verify/recordstream_verification_results.json |
| **ST.012** | GlobalEncodingLockContract | Verifies lock hierarchy and deadlock freedom for MPSEncodingLock (Opportunity Map B1.3) | specs/MPSEncodingLock.tla, structural_checks.sh |
| **ST.013** | SlotAllocatorBackpressure | Verifies lock-free slot allocation with backpressure waiting (Opportunity Map B1.4) | specs/MPSStreamSlotAllocator.tla, structural_checks.sh |
| **ST.014** | DispatchQueueContextSafety | Verifies dispatch_sync reentrancy detection and TLS hazard avoidance (Opportunity Map B1.5) | specs/MPSDispatchQueueContext.tla, structural_checks.sh |

### Slot Allocator Properties (Lock-Free Bitmask Allocation)

| Property ID | Name | Description | Spec/Harness | Assumptions |
|-------------|------|-------------|--------------|-------------|
| **SA.001** | MutualExclusion | No two threads own the same slot | MPSStreamSlotAllocator.tla | Atomic CAS semantics correct |
| **SA.002** | SlotConsistency | Free mask correctly reflects slot ownership | MPSStreamSlotAllocator.tla | None |
| **SA.003** | MutexExclusivity | CV mutex held by at most one thread | MPSStreamSlotAllocator.tla | std::mutex semantics correct |
| **SA.004** | WaiterConsistency | Waiters are threads without slots | MPSStreamSlotAllocator.tla | None |
| **SA.005** | CASExclusivity | At most one thread in CAS critical section | MPSStreamSlotAllocator.tla | Modeled - real CAS is atomic |
| **SA.006** | NoDoubleOwnership | A slot is owned by at most one thread | MPSStreamSlotAllocator.tla | None |
| **SA.007** | BackpressureNoLostWakeup | Waiters eventually get slot when available (liveness) | MPSStreamSlotAllocator.tla | Fairness assumption |
| **SA.008** | SlotEventuallyAvailable | Released slots wake up waiters | MPSStreamSlotAllocator.tla | Fairness assumption |
| **SA.009** | DeadlockFree | System can always make progress | MPSStreamSlotAllocator.tla | None |

### RecordStream Properties (Cross-Stream Lifetime)

| Property ID | Name | Description | Spec/Harness | Assumptions |
|-------------|------|-------------|--------------|-------------|
| **RS.001** | NoEarlyReuse | Buffer cannot be recycled until all pending events complete | MPSRecordStream.tla | Event completion modeled as atomic flag |
| **RS.002** | EventAccountingConsistent | Events removed from pending_events only when query() returns true | MPSRecordStream.tla | None |
| **RS.003** | BoundedPendingEvents | pending_events size bounded by NumStreams (one per stream) | MPSRecordStream.tla | None |

### Global Encoding Lock Properties (Lock Hierarchy)

| Property ID | Name | Description | Spec/Harness | Assumptions |
|-------------|------|-------------|--------------|-------------|
| **GL.001** | DeadlockFree | System never reaches a state where all threads are blocked | MPSEncodingLock.tla | Lock hierarchy respected |
| **GL.002** | MutexExclusivity | Each non-recursive mutex is held by at most one thread | MPSEncodingLock.tla | std::mutex semantics correct |
| **GL.003** | NoReentrantDeadlock | recursive_mutex allows re-acquisition without deadlock | MPSEncodingLock.tla | recursive_mutex semantics correct |
| **GL.004** | LockOrderValid | Encoding lock (Level 5) is acquired after other locks | MPSEncodingLock.tla, MPSThreadSafety.h | None |
| **GL.005** | NoWaitUnderEncodingLock | (ASPIRATIONAL) No blocking GPU waits while holding encoding lock | MPSEncodingLock.tla | Scalability property, may fail today |

### Dispatch Queue Context Properties (Reentrancy + TLS Hazards)

| Property ID | Name | Description | Spec/Harness | Assumptions |
|-------------|------|-------------|--------------|-------------|
| **DQ.001** | NoReentrantDispatchSync | Re-entrant dispatch_sync detected via dispatch_get_specific, executed inline | MPSDispatchQueueContext.tla | dispatch_get_specific returns queue identity |
| **DQ.002** | NoTLSLookupInsideDispatchedBlock | Dispatch blocks use captured stream pointer, not TLS lookup | MPSDispatchQueueContext.tla | None |
| **DQ.003** | ExceptionPropagationSound | Exceptions caught in dispatch_sync_with_rethrow are properly rethrown | MPSDispatchQueueContext.tla | std::exception_ptr semantics correct |
| **DQ.004** | QueueExclusivity | Serial queue semantics: at most one thread per queue | MPSDispatchQueueContext.tla | GCD serial queue semantics |
| **DQ.005** | DeadlockFree | Safe dispatch pattern avoids reentrancy deadlock | MPSDispatchQueueContext.tla | None |

### Tensor Lifetime Properties (Use-After-Free Prevention)

| Property ID | Name | Description | Spec/Harness | Assumptions |
|-------------|------|-------------|--------------|-------------|
| **TL.001** | NoUseAfterFreeCrashes | No crash from accessing freed MTLBuffer during kernel encoding | TensorLifetime.tla | Refcount semantics correct |
| **TL.002** | OwnedTensorIsValid | If thread owns tensor (via __block capture), MTLBuffer remains valid | TensorLifetime.tla | Ownership implies refcount > 0 |
| **TL.003** | EncodingImpliesValidBuffer | Thread in encoding state always has valid MTLBuffer | TensorLifetime.tla | Mutex serialization works |
| **TL.004** | RefcountConsistent | Deallocated tensors have refcount = 0 | TensorLifetime.tla | None |

---

**Note (2025-12-19):** Structural properties (ST.*) track concrete patterns checked by `mps-verify/scripts/structural_checks.sh`. Some older documents and spec comments reference more prescriptive “TOCTOU triple-check” / “CallbackState” shapes; treat those as abstract models unless the code explicitly implements them.

## Code Anchor to Specification Mapping

### MPSStream.mm / MPSStream.h

| Code Location | Function | Issue Fixed | TLA+ Spec | CBMC Harness | Properties |
|---------------|----------|-------------|-----------|--------------|------------|
| `MPSStream.mm:getCurrentStream()` | `getCurrentStream()` | N/A | MPSStreamPool.tla | stream_pool_harness.c | SP.001, SP.005 |
| `MPSStream.mm:ThreadStreamSlot::~ThreadStreamSlot()` | TLS cleanup | N/A | N/A | N/A | ST.001 |
| `MPSStream.mm:releaseSlotIfPoolAlive()` | Shutdown-safe slot recycle | N/A | N/A | N/A | ST.001 |
| `MPSStream.mm:createStream()` | Stream creation | 32.37 | MPSStreamPool.tla | stream_pool_harness.c | SP.004, LP.002 |
| `MPSStream.mm:~MPSStreamPool()` | Destructor | N/A | MPSStreamPool.tla (DestroyPool) | stream_pool_harness.c | SP.001 |
| `MPSStream.mm:49-70` | `getGlobalMetalEncodingMutex()`, `MPSEncodingLock` | N/A | MPSEncodingLock.tla | N/A | GL.001, GL.002, GL.003, GL.004, ST.012 |
| `MPSStream.mm:224-255` | `synchronize()` with encoding lock | N/A | MPSEncodingLock.tla (BeginEncoding, WaitGPU) | N/A | GL.001, GL.005 |
| `MPSStream.mm:581-645` | `acquireSlot()` lock-free bitmask allocation | N/A | MPSStreamSlotAllocator.tla | N/A | SA.001, SA.002, SA.006, SA.009, ST.013 |
| `MPSStream.mm:648-665` | `releaseStreamSlot()` atomic OR + CV notify | N/A | MPSStreamSlotAllocator.tla | N/A | SA.002, SA.007, SA.008, ST.013 |
| `MPSStream.mm:489-509` | `~ThreadStreamSlot()` TLS destructor | N/A | MPSStreamSlotAllocator.tla | N/A | SA.001, ST.001, ST.013 |
| `MPSStream.mm:333-343` | `slot_cv_mutex_`, `slot_available_cv_` backpressure | N/A | MPSStreamSlotAllocator.tla | N/A | SA.003, SA.007, SA.008, ST.013 |
| `MPSThreadSafety.h:90-100` | Lock hierarchy documentation | N/A | MPSEncodingLock.tla | N/A | GL.004 |

### MPSAllocator.mm / MPSAllocator.h

| Code Location | Function | Issue Fixed | TLA+ Spec | CBMC Harness | Properties |
|---------------|----------|-------------|-----------|--------------|------------|
| `MPSAllocator.mm:getSharedBufferPtr()` | Buffer ptr access | 32.267 | MPSAllocator.tla | aba_detection_harness.c | SP.002, ST.002 |
| `MPSAllocator.mm:1064-1071` | ABA double-check | 32.19 | MPSAllocator.tla (GetPtrVerify) | aba_detection_harness.c | SP.002 |
| `MPSAllocator.mm:alloc()` | Allocation | N/A | MPSAllocator.tla | alloc_free_harness.c | SP.003, SP.008 |
| `MPSAllocator.mm:free()` | Deallocation | N/A | MPSAllocator.tla | alloc_free_harness.c | SP.003, SP.009 |
| `MPSAllocator.mm:recordStream()` | Stream recording | 32.110-32.112 | MPSAllocator.tla, MPSRecordStream.tla | N/A | SP.004, RS.001, RS.002, RS.003, ST.011 |
| `MPSAllocator.mm:free_buffer()` | Buffer release with event deferral | N/A | MPSRecordStream.tla | N/A | RS.001, RS.002 |
| `MPSAllocator.mm:process_pending_buffers_locked()` | Process deferred buffers | N/A | MPSRecordStream.tla | N/A | RS.001 |
| `MPSAllocator.mm:setBufferShape()` | Shape setting | 32.19 | MPSAllocator.tla | N/A | SP.002 |
| `MPSAllocator.h:TLSBlockCache` | TLS cache | 32.68, 32.93 | MPSAllocator.tla | tls_cache_harness.c | SP.001 |

### MPSEvent.mm / MPSEvent.h

| Code Location | Function | Issue Fixed | TLA+ Spec | CBMC Harness | Properties |
|---------------|----------|-------------|-----------|--------------|------------|
| `MPSEvent.mm:~MPSEvent()` | Destructor | 32.107 | MPSEvent.tla | N/A | SP.007 |
| `MPSEvent.mm:reset()` | Event reset | 32.89 | MPSEvent.tla | N/A | SP.007, ST.003 |
| `MPSEvent.mm:record()` | Event recording | N/A | MPSEvent.tla | N/A | SP.007 |
| `MPSEvent.mm:synchronize()` | Event sync | N/A | MPSEvent.tla | N/A | SP.007 |
| `MPSEvent.h:m_in_use_events` | In-use lifetime via shared_ptr | N/A | MPSEvent.tla | N/A | SP.007, ST.003 |
| `MPSEvent.mm:elapsedTime()` | Shared_ptr copies during wait | N/A | MPSEvent.tla | N/A | SP.007, ST.003 |
| `MPSEvent.mm:notifyLocked()` | Explicit dispatch queue for notifications | N/A | MPSEvent.tla | N/A | SP.007, ST.003 |

---

## TLA+ Specification Details

### MPSStreamPool.tla

**File:** `mps-verify/specs/MPSStreamPool.tla`
**Config:** `mps-verify/specs/MPSStreamPool.cfg`
**States Verified:** 535,293 states (NumThreads=4, NumStreams=4)

**Invariants Checked:**
- `NoUseAfterFree` - No thread in "using_stream" state when pool dead
- `TLSBindingValid` - TLS values within bounds when pool alive
- `StreamBoundsValid` - Stream IDs always in valid range
- `MainThreadGetsDefaultStream` - Thread 1 gets stream 0
- `WorkerThreadsAvoidDefaultStream` - Threads 2+ get streams 1..N-1
- `ForkInvalidatesTLS` - Fork clears all TLS bindings
- `PoolAliveImpliesCreated` - pool_alive → pool_ever_created
- `ExactlyOneMainThread` - Cardinality check

**Evidence Artifact:** `mps-verify/states/MPSStreamPool_<timestamp>/`

### MPSAllocator.tla

**File:** `mps-verify/specs/MPSAllocator.tla`
**Config:** `mps-verify/specs/MPSAllocator.cfg`
**States Verified:** 15.3M states (NumThreads=3, NumBuffers=4)

**Invariants Checked:**
- `ABADetectionSound` - use_count prevents ABA on buffer reuse
- `NoDoubleFree` - Buffer must be in_use to be freed
- `NoUseAfterFree` - Can't access non-allocated buffer
- `BufferConsistency` - in_use implies allocated
- `MutexExclusivity` - One mutex holder at a time
- `PoolsDisjoint` - available ∩ allocated = ∅

**Scalability Properties (Informational):**
- `ParallelLockHolding` - Documents parallelism possible
- `GlobalSerializerViolation` - Documents m_mutex bottleneck
- `ExcessiveLocking` - Documents 3 locks per getPtr
- `DoubleMutexBottleneck` - Documents 2x m_mutex per getPtr

**Evidence Artifact:** `mps-verify/states/MPSAllocator_<timestamp>/`

### MPSEvent.tla

**File:** `mps-verify/specs/MPSEvent.tla`
**Config:** `mps-verify/specs/MPSEvent.cfg`
**States Verified:** ~1.2M states (NumThreads=3, NumEvents=3)

**Invariants Checked:**
- `CallbackSafetyInvariant` - Callback only accesses alive state
- `DestructorWaitCorrectness` - Destructor sets alive=false before wait
- `PoolReuseSafety` - reset() invalidates old callback state
- `NoUAFInCallbacks` - shared_ptr prevents use-after-free

**Evidence Artifact:** `mps-verify/states/MPSEvent_<timestamp>/`

### MPSBatchQueue.tla (NEW - N=1304)

**File:** `specs/MPSBatchQueue.tla`
**Config:** `specs/MPSBatchQueue.cfg`
**States Verified:** 24,419 states (NumUserThreads=3, NumWorkers=2, MaxOperations=20)

**Invariants Checked:**
- `NoStuckFutures` - When queue drains, all submitted requests are completed
- `StopDrains` - After shutdown, queue is empty
- `SubmitStopRaceSafe` - Cannot submit after shutdown requested
- `CompletedNeverExceedsSubmitted` - Completed count <= submitted count
- `InFlightConsistent` - In-flight request count is consistent
- `WorkersOnlyWhenActive` - Workers only alive when queue running or draining

**Properties Verified (from Appendix B):**
- BQ.NoStuckFutures - Every successful submit() returns a future that is eventually fulfilled
- BQ.StopDrains - After stop() completes, there are no queued requests that can never be processed
- BQ.SubmitStopRaceSafe - submit() cannot enqueue work after shutdown requested

**Evidence Artifact:** TLC output 2025-12-19: 24,419 states, 6,635 distinct, 0 errors

### MPSRecordStream.tla (NEW - N=1305)

**File:** `specs/MPSRecordStream.tla`
**Config:** `specs/MPSRecordStream.cfg`
**States Verified:** 154,923,024 states (NumBuffers=3, NumStreams=3, NumThreads=2, MaxOperations=20)
**Depth:** 24

**Purpose:** Model the recordStream() cross-stream lifetime protocol with detailed event-based synchronization.

**Invariants Checked:**
- `RSNoEarlyReuse` - Buffer cannot be in available_buffers while pending events exist
- `RSEventAccountingConsistent` - Events removed only when query() observes completion
- `RSBoundedPendingEvents` - pending_events size bounded by NumStreams
- `BufferStateConsistency` - Buffer state matches set membership (available, pending_free)
- `StreamUsesConsistency` - pending_events is subset of stream_uses
- `DeadlockFree` - System can always make progress

**Key Actions Modeled:**
- `RecordStream(t, b, s)` - Creates event for cross-stream usage (per unique stream)
- `EventComplete(b, s)` - GPU signals event completion (async)
- `FreeBuffer(t, b)` - Queries events, defers if any pending
- `ProcessPendingBuffer(t, b)` - Processes deferred buffers

**Correspondence to Production Code:**
| TLA+ | Production Code (MPSAllocator.mm) |
|------|-----------------------------------|
| `pending_events[b]` (set) | `buffer_block->pending_events` (vector<MPSEventPtr>) |
| `event_completed[b][s]` | `MPSEvent::query()` return value |
| `RecordStream` action | `recordStream()` lines 898-924 |
| `FreeBuffer` deferral | `free_buffer()` lines 543-563 |
| `ProcessPendingBuffer` | `process_pending_buffers_locked()` lines 777-791 |

**Properties Verified (from Opportunity Map B1.1):**
- RS.NoEarlyReuse: Buffer cannot return to available pool until all pending events complete
- RS.EventAccounting: Events removed only when query() observes completion
- RS.NoUnboundedGrowth: pending_events bounded by NumStreams

**Evidence Artifact:** `mps-verify/recordstream_verification_results.json` (N=1305, 2025-12-19)

### MPSEncodingLock.tla (NEW - N=1306)

**File:** `specs/MPSEncodingLock.tla`
**Config:** `specs/MPSEncodingLock.cfg`
**States Verified:** 88,616,257 states (NumThreads=3, NumStreams=2)
**Distinct States:** 10,132,608
**Depth:** 119

**Purpose:** Model the global encoding lock hierarchy and verify deadlock freedom for the MPSEncodingLock used to serialize Metal encoding operations.

**Invariants Checked:**
- `TypeInvariant` - All variables within expected bounds
- `MutexExclusivity` - Non-recursive mutexes held by at most one thread
- `NoReentrantDeadlock` - recursive_mutex prevents self-deadlock
- `StreamLockConsistency` - thread_stream tracking matches stream_locks
- `EncodingStateConsistency` - "encoding" state implies holding encoding lock
- `Safety` - Combined safety invariant (all of the above)

**Key Actions Modeled:**
- `BeginEncoding(t, s)` - Acquire encoding lock then stream lock (proper order)
- `WaitGPU(t)` - GPU wait while holding encoding lock (aspirational violation)
- `EndEncoding(t)` - Release both locks in reverse order
- `AcquirePoolLock(t)` / `AcquireAllocatorLock(t)` / `AcquireCreationLock(t)` - Level 1-3 locks

**Correspondence to Production Code:**
| TLA+ | Production Code (MPSStream.mm) |
|------|--------------------------------|
| `encoding_lock_holder` | Thread holding `getGlobalMetalEncodingMutex()` |
| `encoding_lock_count` | recursive_mutex acquisition count |
| `stream_locks[s]` | `_streamMutex` per-stream mutex |
| `BeginEncoding` action | `synchronize()` lines 224-229 |
| `WaitGPU` action | `commitAndWait()` lines 240-255 |

**Properties Verified (from Opportunity Map B1.3):**
- GL.001 DeadlockFree: No lock-order cycle causes deadlock
- GL.002 MutexExclusivity: Single holder for non-recursive mutexes
- GL.003 NoReentrantDeadlock: recursive_mutex semantics allow re-entry
- GL.004 LockOrderValid: Encoding lock acquired last per hierarchy
- GL.005 NoWaitUnderEncodingLock: (ASPIRATIONAL) Tracks GPU waits under lock

**Aspirational Property Note:**
The spec tracks `wait_under_encoding_count` to detect when `waitUntilCompleted()` is called while holding the global encoding lock. This is a scalability concern (not a correctness bug): holding the global lock during GPU waits serializes encoding across all threads. Current code DOES wait under the lock in `synchronize()` - this is documented for future optimization.

**Evidence Artifact:** TLC output 2025-12-19: 88.6M states, 10.1M distinct, 0 errors

### MPSStreamSlotAllocator.tla (NEW - N=1307)

**File:** `specs/MPSStreamSlotAllocator.tla`
**Config:** `specs/MPSStreamSlotAllocator.cfg`
**States Verified:** 59,133 states (NumSlots=5, NumThreads=3, MaxOperations=20, BackpressureEnabled=TRUE)
**Distinct States:** 15,264
**Depth:** 28

**Purpose:** Model the lock-free stream slot allocator with condition variable-based backpressure waiting.

**Invariants Checked:**
- `SafetyInvariant` - Combined: TypeOK + SA.001-SA.006
- `SA_MutualExclusion` - No two threads own the same slot
- `SA_SlotConsistency` - Free mask correctly reflects slot ownership
- `SA_MutexExclusivity` - CV mutex held by at most one thread
- `SA_WaiterConsistency` - Waiters are threads without slots
- `SA_CASExclusivity` - At most one thread in CAS critical section
- `SA_NoDoubleOwnership` - A slot is owned by at most one thread
- `SA_DeadlockFree` - System can always make progress

**Key Actions Modeled:**
- `TryAcquireSlotFast(t)` - Lock-free CAS attempt on bitmask
- `CompleteCAS(t)` - Complete atomic CAS operation
- `EnterBackpressureWait(t)` - Pool exhausted, enter CV wait queue
- `WakeFromWait(t)` - Waiter wakes up (CV signaled or spurious)
- `ReacquireAfterWait(t)` - Try to acquire slot after wake
- `ReleaseSlot(t)` - Release slot back to freelist (atomic OR)
- `NotifyWaiter` - Wake up one waiter after release
- `ThreadExit(t)` - TLS destructor releases slot
- `DoubleReleaseAttempt(t)` - Test double-release detection
- `PoolShutdown` - Pool destruction sets pool_alive=false

**Correspondence to Production Code:**
| TLA+ | Production Code (MPSStream.mm) |
|------|-------------------------------|
| `free_mask` | `free_slots_mask_` (std::atomic<uint32_t>) |
| `thread_slots[t]` | `tls_stream_slot.slot_index` (thread_local) |
| `waiters` | Threads blocked on `slot_available_cv_.wait()` |
| `cv_mutex_holder` | Thread holding `slot_cv_mutex_` |
| `pool_alive` | `g_pool_alive` (std::atomic<bool>) |
| `TryAcquireSlotFast` | `acquireSlot()` lines 581-645 |
| `ReleaseSlot` | `releaseStreamSlot()` lines 648-665 |
| `ThreadExit` | `~ThreadStreamSlot()` destructor lines 489-509 |
| `NotifyWaiter` | `slot_available_cv_.notify_one()` line 664 |

**Properties Verified (from Opportunity Map B1.4):**
- SA.001 MutualExclusion: No two threads own the same slot
- SA.002 SlotConsistency: Bitmask consistent with ownership
- SA.006 NoDoubleOwnership: Verified via CAS semantics
- SA.009 DeadlockFree: System always has enabled action

**Backpressure Model:**
The spec models the backpressure mechanism introduced in Phase 24.0:
- When `BackpressureEnabled=TRUE` and `free_mask=0`, threads enter `waiters` set
- `ReleaseSlot` triggers `NotifyWaiter` to wake one waiter
- Waiter then re-attempts acquisition via `ReacquireAfterWait`
- Spurious wakeups are allowed in the model for completeness

**Evidence Artifact:** `mps-verify/slotallocator_verification_results.json` (N=1307, 2025-12-19)

### MPSDispatchQueueContext.tla (NEW - N=1308)

**File:** `specs/MPSDispatchQueueContext.tla`
**Config:** `specs/MPSDispatchQueueContext.cfg`
**States Verified:** 606,106 states (NumThreads=3, NumQueues=2, MaxOps=5, AllowUnsafePatterns=FALSE)
**Distinct States:** 152,171
**Depth:** 22

**Purpose:** Model GCD dispatch queue execution context hazards including re-entrant dispatch_sync deadlock, TLS lookup inside dispatch blocks, and exception propagation soundness.

**Invariants Checked:**
- `TypeInvariant` - All variables within expected bounds
- `QueueExclusivity` - At most one thread executes on each queue (serial queue semantics)
- `ThreadQueueConsistency` - Thread queue binding consistent with queue executing state
- `ExceptionNotLost` - Exceptions caught in dispatch blocks are properly rethrown
- `Safety` - Combined safety (exclusivity + consistency + exception propagation)
- `NoTLSLookupInBlock` - No TLS lookup inside dispatch blocks

**Key Actions Modeled:**
- `SafeDispatchSync(t, q)` - Dispatch with queue_has_specific reentrancy check
- `ReentrantDispatchInline(t, q)` - Inline execution when already on queue (safe path)
- `UnsafeDispatchSync(t, q)` - Dispatch without reentrancy check (for proving hazard)
- `TLSLookupInBlock(t)` - Incorrect TLS lookup inside dispatch block (for proving hazard)
- `CompleteDispatch(t)` - Thread finishes execution and releases queue
- `ThrowException(t)` / `CatchException(t)` / `RethrowException(t)` - Exception flow

**Correspondence to Production Code:**
| TLA+ | Production Code |
|------|-----------------|
| `queue_executing[q]` | Thread currently running on `mpsStream->queue()` |
| `queue_has_specific` | `dispatch_queue_set_specific()` with `kMPSStreamQueueSpecificKey` |
| `SafeDispatchSync` | `dispatch_sync_with_rethrow()` with `dispatch_get_specific()` check |
| `ReentrantDispatchInline` | `executeMPSGraph()` pattern (inline when already on queue) |
| `UnsafeDispatchSync` | Missing `dispatch_get_specific()` check (bug case) |
| `thread_captured_stream` | Stream pointer captured before dispatch block |
| `thread_uses_tls` | Incorrect `getCurrentMPSStream()` inside dispatch block |

**Properties Verified (from Opportunity Map B1.5):**
- DQ.001 NoReentrantDispatchSync: dispatch_get_specific detects same-queue dispatch
- DQ.002 NoTLSLookupInsideDispatchedBlock: Blocks use captured stream, not TLS
- DQ.003 ExceptionPropagationSound: Caught exceptions are rethrown
- DQ.004 QueueExclusivity: Serial queue semantics maintained
- DQ.005 DeadlockFree: Safe pattern prevents reentrancy deadlock

**Model Modes:**
- `AllowUnsafePatterns=FALSE` (default): Only safe actions enabled, all invariants pass
- `AllowUnsafePatterns=TRUE`: Enables unsafe actions (UnsafeDispatchSync, TLSLookupInBlock); NoReentrantDeadlock will FAIL (proves hazard exists)

**Code Anchors:**
- `MPSStream.mm:dispatch_sync_with_rethrow` - Exception-safe dispatch wrapper
- `MPSGraph.mm:executeMPSGraph` - Uses `dispatch_get_specific` for inline execution
- `tests/repro_dispatch_sync_with_rethrow_reentrancy.mm` - Standalone proof-of-concept

**Evidence Artifact:** `mps-verify/dispatchqueue_verification_results.json` (N=1308, 2025-12-19)

---

## CBMC Harness Details

### aba_detection_harness.c

**File:** `mps-verify/verification/cbmc/harnesses/aba_detection_harness.c`
**Run:** `cbmc aba_detection_harness.c -I ../models -I ../stubs --unwind 5`

**Properties Verified:**
- ABA scenario: Thread A sees in_use=true, releases lock, Thread B frees+reallocs
- use_count check detects generation change
- Thread A aborts instead of using stale buffer

**Assertions:**
- `aba_detection_sound`: use_count mismatch detected on ABA
- `no_use_after_free`: Thread doesn't use reallocated buffer

### alloc_free_harness.c

**File:** `mps-verify/verification/cbmc/harnesses/alloc_free_harness.c`
**Run:** `cbmc alloc_free_harness.c -I ../models -I ../stubs --unwind 10`

**Properties Verified:**
- Allocation returns valid buffer or fails gracefully
- Free returns buffer to available pool
- No double-free possible under mutex protection
- Buffer ownership tracking prevents cross-thread double-free

### tls_cache_harness.c

**File:** `mps-verify/verification/cbmc/harnesses/tls_cache_harness.c`
**Run:** `cbmc tls_cache_harness.c -I ../models -I ../stubs --unwind 10`

**Properties Verified:**
- TLS cache initialization safety
- flush() during shutdown race prevention (32.68, 32.93)
- s_flush_in_progress_count coordination

### stream_pool_harness.c

**File:** `mps-verify/verification/cbmc/harnesses/stream_pool_harness.c`
**Run:** `cbmc stream_pool_harness.c -I ../models -I ../stubs --unwind 10`

**Properties Verified:**
- Pool lifecycle (create, use, destroy)
- TLS binding assignment
- Fork safety (TLS cleared)
- TLS slot lifecycle + shutdown-safety guards (ST.001)

---

## Static Analysis (Clang TSA)

**Tool:** Clang Thread Safety Analysis
**Script:** `mps-verify/scripts/run_clang_tsa.sh`
**Output:** `mps-verify/tsa_results.json`

**Files Analyzed:**
- `MPSStream.mm` - stream_mutex, pool_creation_mutex
- `MPSAllocator.mm` - allocator_mutex (per-shard), pool_mutex
- `MPSEvent.mm` - event_mutex, callback_sync_mutex, event_pool_mutex
- `MPSDevice.mm` - device_mutex

**TSA Annotations Applied:**
- `GUARDED_BY(mutex)` - Data protected by mutex
- `REQUIRES(mutex)` - Function requires mutex held
- `EXCLUDES(mutex)` - Function must not hold mutex
- `ACQUIRE/RELEASE` - Lock acquisition/release points

**Known Warnings:** ~280 warnings, mostly from std::mutex lacking capability attribute annotation in system headers.

---

## Issue-to-Fix-to-Verification Mapping

| Issue | Severity | Fix Description | Fix Commit | Verified By | Property |
|-------|----------|-----------------|------------|-------------|----------|
| 32.19 | HIGH | TOCTOU race in buffer lookup | N=350 | MPSAllocator.tla, aba_detection_harness.c | SP.002 |
| 32.37 | MEDIUM | ensureInitialized() TOCTOU | N=321 | MPSStreamPool.tla | SP.004, LP.002 |
| 32.68 | HIGH | TLSBlockCache::flush() TOCTOU | N=335 | tls_cache_harness.c | SP.001 |
| 32.89 | HIGH | reset() callback state | N=353 | MPSEvent.tla | SP.007 |
| 32.93 | HIGH | try_get() TOCTOU | N=361 | tls_cache_harness.c | SP.001 |
| 32.99 | HIGH | getCurrentStream() TOCTOU | N=367 | MPSStreamPool.tla | SP.001, ST.001 |
| 32.104 | HIGH | getCurrentStream() assignment TOCTOU | N=370 | MPSStreamPool.tla | SP.001, ST.001 |
| 32.107 | HIGH | Callback UAF | N=373 | MPSEvent.tla | SP.007, ST.003 |
| 32.267 | HIGH | ABA race in getSharedBufferPtr | N=350 | MPSAllocator.tla, aba_detection_harness.c | SP.002, ST.002 |

---

## Assumption Ledger

| Assumption ID | Description | Justification | Impact if Violated |
|---------------|-------------|---------------|-------------------|
| **A.001** | std::mutex provides mutual exclusion | C++ standard guarantee | All mutex-based properties fail |
| **A.002** | std::atomic provides sequential consistency | C++ standard guarantee | Memory ordering properties fail |
| **A.003** | Fork handler registered via pthread_atfork | Code inspection verified | Fork safety (SP.006) fails |
| **A.004** | use_count never wraps (32-bit, ~4B allocs) | Extremely unlikely in practice | ABA detection (SP.002) could miss |
| **A.005** | Static destruction order is well-defined | C++ standard: reverse init order | Pool shutdown races possible |
| **A.006** | Metal command queue operations are thread-safe | Apple documentation | GPU operations may corrupt |
| **A.007** | shared_ptr is thread-safe | C++ standard guarantee | Callback safety (SP.007) fails |

---

## Verification Commands Quick Reference

```bash
# Run all verification
cd mps-verify && .lake/build/bin/mpsverify check --all

# TLA+ only
.lake/build/bin/mpsverify tla --all

# Specific TLA+ spec
.lake/build/bin/mpsverify tla --spec=MPSStreamPool

# CBMC only
.lake/build/bin/mpsverify cbmc --all

# Static analysis only
.lake/build/bin/mpsverify static

# Generate report
.lake/build/bin/mpsverify report --format=md --output=verification_report.md
```

---

## Verification Impact Statement Template

When modifying concurrency code, answer these questions in your commit message:

1. **What properties could be affected?** (List Property IDs)
2. **What specs/harnesses cover this code?** (List from mapping above)
3. **Did you run verification?** (Include PASS/FAIL + state counts)
4. **Any new assumptions?** (Add to Assumption Ledger if needed)

Example:
```
# 1047: Fix race in getBufferShape()

**Verification Impact:**
1. Properties affected: SP.002, SP.008
2. Specs: MPSAllocator.tla (GetPtrVerify)
3. Verification: TLA+ PASS (15.3M states), CBMC PASS (0/249 assertions failed)
4. No new assumptions
```

---

## CBMC Harness Correspondence Notes

### stream_pool_harness.c (N=1304)

**Purpose:** Verify stream pool lifecycle, TLS binding safety, fork safety, and shutdown-safe slot recycle.

**Production Code:** `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm` (lines 479-742)

**Correspondence Analysis:**

| Aspect | Harness Model | Production Code | Correspondence Level |
|--------|---------------|-----------------|---------------------|
| **Stream acquisition** | Round-robin counter | Lock-free bitmask (`free_slots_mask_`) with CAS | Abstracted - harness uses simpler allocation model |
| **Main thread detection** | `is_main_thread[thread_id]` array | `pthread_main_np() == 1` | Equivalent semantics (deterministic in harness) |
| **TLS binding** | `tls_stream[thread_id]` array | `thread_local ThreadStreamSlot tls_stream_slot` | Faithful - same semantics, different implementation |
| **Pool alive flag** | `pool_alive` boolean | `g_pool_alive` std::atomic<bool> | Faithful - same semantics |
| **Fork handling** | `in_forked_child` flag + TLS clear | Fork handler sets `g_pool_alive=false`, clears TLS | Faithful |
| **Shutdown-safe release** | Modeled in CHECK3 | `releaseSlotIfPoolAlive()` guards with `g_pool_alive.load()` | Faithful (ST.001) |
| **Active user counting** | Not modeled | `g_active_stream_users` atomic counter | Not modeled (parallel mode detection only) |
| **TOCTOU triple-check** | Explicit CHECK1/CHECK2/CHECK3 | Guards in TLS dtor + releaseSlotIfPoolAlive | **Model drift** - harness models older spec |

**What Harness Verifies (Still Valid):**
- SP.001: No use-after-free (pool must be alive when stream is used)
- SP.005: TLS binding validity (within stream bounds)
- SP.006: Fork safety (TLS invalidation)
- ST.001: Pool alive guards TLS cleanup + slot recycle

**What Harness Does NOT Model:**
- Lock-free bitmask freelist algorithm (verified separately in TLA+)
- `g_active_stream_users` parallel mode detection counter
- Actual `pthread_main_np()` behavior (abstracted to deterministic flag)

**Model Drift Note (2025-12-19):**
The harness models explicit TOCTOU triple-check pattern (CHECK1/CHECK2/CHECK3) within `getCurrentStream()`. Production code no longer has this pattern inline - instead, shutdown-safety is enforced via:
1. `g_pool_alive` check in `ThreadStreamSlot::~ThreadStreamSlot()` destructor
2. `releaseSlotIfPoolAlive()` function that guards slot release

The harness remains valid for verifying the semantic properties (SP.001, SP.005, SP.006) but the code structure correspondence has drifted. The TLA+ spec (MPSStreamPool.tla) is the primary verification source for pool lifecycle; CBMC harness provides bounded interleavings verification.

**Recommendation:** Future refactor could extract a "stream slot allocator" concurrency kernel that both harness and production code import, improving correspondence.

---

### aba_detection_harness.c

**Production Code:** `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm` (lines 1064-1071)

**Correspondence Level:** HIGH

The harness directly models the ABA detection pattern:
- Thread A: Read `in_use=true`, capture `use_count`, release lock
- Thread B: Free buffer, reallocate buffer (use_count increments)
- Thread A: Re-acquire lock, verify `use_count` unchanged

This pattern is directly implemented in production code (`getSharedBufferPtr()`).

**What is abstracted:**
- Metal buffer details (irrelevant to ABA logic)
- Buffer shape setting (separate from ABA pattern)

---

## Evidence Retention

Verification artifacts are stored in:
- `mps-verify/states/<timestamp>/` - TLC state traces
- `mps-verify/tsa_results.json` - TSA output
- `mps-verify/verification_report.json` - Combined summary

Keep artifacts for at least the last 5 verification runs to enable regression comparison.

---

## Changelog

| Date | Change | By |
|------|--------|-----|
| 2025-12-17 | Initial traceability map created | N=1045 |
| 2025-12-19 | Added CBMC Harness Correspondence Notes section (Phase 4) | N=1304 |
| 2025-12-19 | Added MPSBatchQueue.tla TLA+ spec (Opportunity Map B1.2) | N=1304 |
| 2025-12-19 | Added MPSStreamSlotAllocator.tla TLA+ spec + ST.013 (Opportunity Map B1.4) | N=1307 |
| 2025-12-22 | Added TensorLifetime.tla TLA+ spec for use-after-free race (CRASH_FIX_ANALYSIS) | N=1971 |
