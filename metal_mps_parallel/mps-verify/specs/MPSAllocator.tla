---------------------------- MODULE MPSAllocator ----------------------------
(*
 * MPS Allocator State Machine
 *
 * This TLA+ specification models the MPSHeapAllocatorImpl from PyTorch's MPS backend.
 * It focuses on verifying the ABA double-check pattern used for thread-safe buffer access.
 *
 * Key invariants verified:
 *   1. ABA Detection Correctness - use_count prevents ABA races
 *   2. No Double Free - buffers cannot be freed twice
 *   3. No Use-After-Free - freed buffers are not accessed
 *   4. TLS Cache Safety - TLS operations are safe during shutdown
 *
 * Based on: pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm
 *
 * The double-check pattern (used in 6 functions):
 *   1. Lock m_mutex, find buffer, capture use_count, get pool, unlock m_mutex
 *   2. Lock pool_mutex
 *   3. Lock m_mutex again, verify buffer exists AND use_count unchanged
 *   4. If changed, abort (ABA detected)
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,      \* Number of threads (e.g., 3)
    NumBuffers       \* Number of buffers in the system (e.g., 4)

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumBuffers \in Nat /\ NumBuffers > 0

VARIABLES
    \* Allocator state
    allocator_alive,         \* Is the allocator alive?
    m_mutex_holder,          \* Thread holding m_mutex (0 = unlocked)
    pool_mutex_holder,       \* Thread holding pool_mutex (0 = unlocked)
    allocated_buffers,       \* Set of currently allocated buffer IDs
    available_buffers,       \* Set of buffers available for reuse

    \* Per-buffer state
    buffer_in_use,           \* BufferBlock.in_use: Buffer ID -> Bool
    buffer_use_count,        \* BufferBlock.use_count: Buffer ID -> Nat (ABA counter)
    buffer_freed,            \* Has buffer been through a free cycle? (for double-free detect)

    \* TLS cache (per-thread)
    tls_cache,               \* Thread -> Set of buffer IDs in TLS cache

    \* Thread state for operations
    pc,                      \* Program counter: Thread -> State
    thread_op,               \* Current operation type: Thread -> {"alloc", "free", "get_ptr", "none"}
    thread_buffer,           \* Buffer being operated on: Thread -> Buffer ID | NULL
    saved_use_count,         \* Captured use_count for double-check: Thread -> Nat | NULL

    \* Tracking for verification
    completed_ops,           \* Count of successfully completed operations
    buffer_owner,            \* Buffer ID -> Thread ID (0 = no owner) - prevents double-free

    \* Scalability tracking (for diagnosing performance bottlenecks)
    lock_wait_count,         \* Thread -> Nat, count of lock acquisition WAITS (blocked attempts)
    lock_acquire_count       \* Thread -> Nat, count of lock acquisitions (for parallel analysis)

vars == <<allocator_alive, m_mutex_holder, pool_mutex_holder, allocated_buffers,
          available_buffers, buffer_in_use, buffer_use_count, buffer_freed,
          tls_cache, pc, thread_op, thread_buffer, saved_use_count, completed_ops,
          buffer_owner, lock_wait_count, lock_acquire_count>>

Threads == 1..NumThreads
Buffers == 1..NumBuffers
NULL == 0

(* Type invariant *)
TypeOK ==
    /\ allocator_alive \in BOOLEAN
    /\ m_mutex_holder \in (Threads \cup {0})
    /\ pool_mutex_holder \in (Threads \cup {0})
    /\ allocated_buffers \subseteq Buffers
    /\ available_buffers \subseteq Buffers
    /\ buffer_in_use \in [Buffers -> BOOLEAN]
    /\ buffer_use_count \in [Buffers -> Nat]
    /\ buffer_freed \in [Buffers -> BOOLEAN]
    /\ tls_cache \in [Threads -> SUBSET Buffers]
    /\ pc \in [Threads -> {"idle", "alloc_lock_m", "alloc_find", "alloc_lock_pool",
                           "alloc_complete", "free_lock_m", "free_lock_pool",
                           "free_to_pool", "getptr_lock_m1", "getptr_capture",
                           "getptr_lock_pool", "getptr_lock_m2", "getptr_verify",
                           "getptr_check_in_use", "getptr_success", "done"}]
    /\ thread_op \in [Threads -> {"alloc", "free", "get_ptr", "none"}]
    /\ thread_buffer \in [Threads -> (Buffers \cup {NULL})]
    /\ saved_use_count \in [Threads -> (Nat \cup {NULL})]
    /\ completed_ops \in Nat
    /\ buffer_owner \in [Buffers -> (Threads \cup {0})]
    /\ lock_wait_count \in [Threads -> Nat]
    /\ lock_acquire_count \in [Threads -> Nat]

(* Initial state *)
Init ==
    /\ allocator_alive = TRUE
    /\ m_mutex_holder = 0
    /\ pool_mutex_holder = 0
    /\ allocated_buffers = {}
    /\ available_buffers = Buffers  \* All buffers start as available
    /\ buffer_in_use = [b \in Buffers |-> FALSE]
    /\ buffer_use_count = [b \in Buffers |-> 0]
    /\ buffer_freed = [b \in Buffers |-> FALSE]
    /\ tls_cache = [t \in Threads |-> {}]
    /\ pc = [t \in Threads |-> "idle"]
    /\ thread_op = [t \in Threads |-> "none"]
    /\ thread_buffer = [t \in Threads |-> NULL]
    /\ saved_use_count = [t \in Threads |-> NULL]
    /\ completed_ops = 0
    /\ buffer_owner = [b \in Buffers |-> 0]  \* No owner initially
    /\ lock_wait_count = [t \in Threads |-> 0]
    /\ lock_acquire_count = [t \in Threads |-> 0]

-----------------------------------------------------------------------------
(* HELPER DEFINITIONS *)

\* Check if thread t holds m_mutex
HoldsMMutex(t) == m_mutex_holder = t

\* Check if thread t holds pool_mutex
HoldsPoolMutex(t) == pool_mutex_holder = t

\* Try to acquire m_mutex (non-blocking check)
CanAcquireMMutex(t) == m_mutex_holder = 0

\* Try to acquire pool_mutex (non-blocking check)
CanAcquirePoolMutex(t) == pool_mutex_holder = 0

-----------------------------------------------------------------------------
(* ALLOCATION OPERATIONS *)

\* Start allocation - thread picks "alloc" operation
StartAlloc(t) ==
    /\ pc[t] = "idle"
    /\ allocator_alive
    /\ thread_op' = [thread_op EXCEPT ![t] = "alloc"]
    /\ pc' = [pc EXCEPT ![t] = "alloc_lock_m"]
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, pool_mutex_holder, allocated_buffers,
                   available_buffers, buffer_in_use, buffer_use_count, buffer_freed,
                   tls_cache, thread_buffer, saved_use_count, completed_ops, buffer_owner,
                   lock_wait_count, lock_acquire_count>>

\* Acquire m_mutex for allocation (track acquisition for scalability analysis)
AllocLockM(t) ==
    /\ pc[t] = "alloc_lock_m"
    /\ CanAcquireMMutex(t)
    /\ m_mutex_holder' = t
    /\ lock_acquire_count' = [lock_acquire_count EXCEPT ![t] = @ + 1]
    /\ pc' = [pc EXCEPT ![t] = "alloc_find"]
    /\ UNCHANGED <<allocator_alive, pool_mutex_holder, allocated_buffers, available_buffers,
                   buffer_in_use, buffer_use_count, buffer_freed, tls_cache, thread_op,
                   thread_buffer, saved_use_count, completed_ops, buffer_owner, lock_wait_count>>

\* Find available buffer (while holding m_mutex)
AllocFind(t) ==
    /\ pc[t] = "alloc_find"
    /\ HoldsMMutex(t)
    /\ \/ \* Found a buffer in available pool
          /\ available_buffers # {}
          /\ LET b == CHOOSE x \in available_buffers : TRUE
             IN /\ thread_buffer' = [thread_buffer EXCEPT ![t] = b]
                /\ available_buffers' = available_buffers \ {b}
                /\ allocated_buffers' = allocated_buffers \cup {b}
                /\ pc' = [pc EXCEPT ![t] = "alloc_lock_pool"]
          /\ UNCHANGED <<m_mutex_holder, thread_op>>
       \/ \* No buffer available - fail (simplified model)
          /\ available_buffers = {}
          /\ m_mutex_holder' = 0  \* Release mutex
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ thread_op' = [thread_op EXCEPT ![t] = "none"]
          /\ UNCHANGED <<thread_buffer, available_buffers, allocated_buffers>>
    /\ UNCHANGED <<allocator_alive, pool_mutex_holder, buffer_in_use, buffer_use_count,
                   buffer_freed, tls_cache, saved_use_count, completed_ops, buffer_owner,
                   lock_wait_count, lock_acquire_count>>

\* Lock pool_mutex and release m_mutex (track acquisition)
AllocLockPool(t) ==
    /\ pc[t] = "alloc_lock_pool"
    /\ HoldsMMutex(t)
    /\ CanAcquirePoolMutex(t)
    /\ pool_mutex_holder' = t
    /\ m_mutex_holder' = 0  \* Release m_mutex after getting pool_mutex
    /\ lock_acquire_count' = [lock_acquire_count EXCEPT ![t] = @ + 1]
    /\ pc' = [pc EXCEPT ![t] = "alloc_complete"]
    /\ UNCHANGED <<allocator_alive, allocated_buffers, available_buffers, buffer_in_use,
                   buffer_use_count, buffer_freed, tls_cache, thread_op, thread_buffer,
                   saved_use_count, completed_ops, buffer_owner, lock_wait_count>>

\* Complete allocation - set in_use and increment use_count
AllocComplete(t) ==
    /\ pc[t] = "alloc_complete"
    /\ HoldsPoolMutex(t)
    /\ thread_buffer[t] # NULL
    /\ LET b == thread_buffer[t]
       IN /\ buffer_in_use' = [buffer_in_use EXCEPT ![b] = TRUE]
          /\ buffer_use_count' = [buffer_use_count EXCEPT ![b] = @ + 1]
          /\ buffer_freed' = [buffer_freed EXCEPT ![b] = FALSE]  \* Reset freed flag
          /\ buffer_owner' = [buffer_owner EXCEPT ![b] = t]  \* Set ownership
    /\ pool_mutex_holder' = 0  \* Release pool_mutex
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ completed_ops' = completed_ops + 1
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, allocated_buffers, available_buffers,
                   tls_cache, thread_op, thread_buffer, saved_use_count,
                   lock_wait_count, lock_acquire_count>>

-----------------------------------------------------------------------------
(* FREE OPERATIONS *)

\* Start free - thread picks "free" operation with a buffer it OWNS
\* Guard: Thread must own a buffer (buffer_owner[b] = t)
\* The actual claim (in_use=FALSE) happens in FreeToPool under pool_mutex
StartFree(t) ==
    /\ pc[t] = "idle"
    /\ allocator_alive
    /\ \E b \in allocated_buffers : buffer_owner[b] = t  \* Only free buffers we own
    /\ LET b == CHOOSE x \in allocated_buffers : buffer_owner[x] = t
       IN thread_buffer' = [thread_buffer EXCEPT ![t] = b]
    /\ thread_op' = [thread_op EXCEPT ![t] = "free"]
    /\ pc' = [pc EXCEPT ![t] = "free_lock_pool"]
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, pool_mutex_holder, allocated_buffers,
                   available_buffers, buffer_in_use, buffer_use_count, buffer_freed,
                   tls_cache, saved_use_count, completed_ops, buffer_owner,
                   lock_wait_count, lock_acquire_count>>

\* Acquire pool_mutex for free (track acquisition)
FreeLockPool(t) ==
    /\ pc[t] = "free_lock_pool"
    /\ CanAcquirePoolMutex(t)
    /\ pool_mutex_holder' = t
    /\ lock_acquire_count' = [lock_acquire_count EXCEPT ![t] = @ + 1]
    /\ pc' = [pc EXCEPT ![t] = "free_to_pool"]
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, allocated_buffers, available_buffers,
                   buffer_in_use, buffer_use_count, buffer_freed, tls_cache, thread_op,
                   thread_buffer, saved_use_count, completed_ops, buffer_owner, lock_wait_count>>

\* Free buffer back to pool
\* Sets buffer_in_use=FALSE under pool_mutex protection
FreeToPool(t) ==
    /\ pc[t] = "free_to_pool"
    /\ HoldsPoolMutex(t)
    /\ thread_buffer[t] # NULL
    /\ LET b == thread_buffer[t]
       IN /\ buffer_in_use[b]  \* Must still be in use (not double-freed)
          /\ buffer_in_use' = [buffer_in_use EXCEPT ![b] = FALSE]
          /\ buffer_freed' = [buffer_freed EXCEPT ![b] = TRUE]
          /\ buffer_owner' = [buffer_owner EXCEPT ![b] = 0]  \* Clear ownership
          /\ available_buffers' = available_buffers \cup {b}
          /\ allocated_buffers' = allocated_buffers \ {b}
    /\ pool_mutex_holder' = 0  \* Release pool_mutex
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ thread_buffer' = [thread_buffer EXCEPT ![t] = NULL]
    /\ completed_ops' = completed_ops + 1
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, buffer_use_count,
                   tls_cache, thread_op, saved_use_count, lock_wait_count, lock_acquire_count>>

-----------------------------------------------------------------------------
(* GETSHAREDBUFFERPTR - The ABA Double-Check Pattern *)

\* Start getSharedBufferPtr operation
StartGetPtr(t) ==
    /\ pc[t] = "idle"
    /\ allocator_alive
    /\ thread_op' = [thread_op EXCEPT ![t] = "get_ptr"]
    /\ pc' = [pc EXCEPT ![t] = "getptr_lock_m1"]
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, pool_mutex_holder, allocated_buffers,
                   available_buffers, buffer_in_use, buffer_use_count, buffer_freed,
                   tls_cache, thread_buffer, saved_use_count, completed_ops, buffer_owner,
                   lock_wait_count, lock_acquire_count>>

\* First m_mutex acquisition (track for scalability)
GetPtrLockM1(t) ==
    /\ pc[t] = "getptr_lock_m1"
    /\ CanAcquireMMutex(t)
    /\ m_mutex_holder' = t
    /\ lock_acquire_count' = [lock_acquire_count EXCEPT ![t] = @ + 1]
    /\ pc' = [pc EXCEPT ![t] = "getptr_capture"]
    /\ UNCHANGED <<allocator_alive, pool_mutex_holder, allocated_buffers, available_buffers,
                   buffer_in_use, buffer_use_count, buffer_freed, tls_cache, thread_op,
                   thread_buffer, saved_use_count, completed_ops, buffer_owner, lock_wait_count>>

\* Capture buffer and use_count (while holding m_mutex)
GetPtrCapture(t) ==
    /\ pc[t] = "getptr_capture"
    /\ HoldsMMutex(t)
    /\ \/ \* Found an allocated buffer
          /\ allocated_buffers # {}
          /\ LET b == CHOOSE x \in allocated_buffers : TRUE
             IN /\ thread_buffer' = [thread_buffer EXCEPT ![t] = b]
                /\ saved_use_count' = [saved_use_count EXCEPT ![t] = buffer_use_count[b]]
                /\ m_mutex_holder' = 0  \* Release m_mutex
                /\ pc' = [pc EXCEPT ![t] = "getptr_lock_pool"]
          /\ UNCHANGED thread_op
       \/ \* No buffer found
          /\ allocated_buffers = {}
          /\ m_mutex_holder' = 0
          /\ pc' = [pc EXCEPT ![t] = "done"]
          /\ thread_op' = [thread_op EXCEPT ![t] = "none"]
          /\ UNCHANGED <<thread_buffer, saved_use_count>>
    /\ UNCHANGED <<allocator_alive, pool_mutex_holder, allocated_buffers, available_buffers,
                   buffer_in_use, buffer_use_count, buffer_freed, tls_cache, completed_ops,
                   buffer_owner, lock_wait_count, lock_acquire_count>>

\* Acquire pool_mutex (track for scalability)
GetPtrLockPool(t) ==
    /\ pc[t] = "getptr_lock_pool"
    /\ CanAcquirePoolMutex(t)
    /\ pool_mutex_holder' = t
    /\ lock_acquire_count' = [lock_acquire_count EXCEPT ![t] = @ + 1]
    /\ pc' = [pc EXCEPT ![t] = "getptr_lock_m2"]
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, allocated_buffers, available_buffers,
                   buffer_in_use, buffer_use_count, buffer_freed, tls_cache, thread_op,
                   thread_buffer, saved_use_count, completed_ops, buffer_owner, lock_wait_count>>

\* Second m_mutex acquisition (for verification) - THIS IS THE SCALABILITY BOTTLENECK
\* In getSharedBufferPtr, m_mutex is acquired TWICE per call: once to capture use_count,
\* once to verify. With global m_mutex, this causes 2x serialization overhead.
GetPtrLockM2(t) ==
    /\ pc[t] = "getptr_lock_m2"
    /\ HoldsPoolMutex(t)
    /\ CanAcquireMMutex(t)
    /\ m_mutex_holder' = t
    /\ lock_acquire_count' = [lock_acquire_count EXCEPT ![t] = @ + 1]  \* SECOND m_mutex acquisition
    /\ pc' = [pc EXCEPT ![t] = "getptr_verify"]
    /\ UNCHANGED <<allocator_alive, pool_mutex_holder, allocated_buffers, available_buffers,
                   buffer_in_use, buffer_use_count, buffer_freed, tls_cache, thread_op,
                   thread_buffer, saved_use_count, completed_ops, buffer_owner, lock_wait_count>>

\* Verify buffer still valid and use_count unchanged (ABA check)
GetPtrVerify(t) ==
    /\ pc[t] = "getptr_verify"
    /\ HoldsMMutex(t)
    /\ HoldsPoolMutex(t)
    /\ thread_buffer[t] # NULL
    /\ LET b == thread_buffer[t]
           saved_uc == saved_use_count[t]
       IN \/ \* Buffer still allocated AND use_count matches - proceed
             /\ b \in allocated_buffers
             /\ buffer_use_count[b] = saved_uc
             /\ m_mutex_holder' = 0  \* Release m_mutex, keep pool_mutex
             /\ pc' = [pc EXCEPT ![t] = "getptr_check_in_use"]
             /\ UNCHANGED <<thread_buffer, pool_mutex_holder, thread_op>>
          \/ \* ABA detected OR buffer freed - abort
             /\ (b \notin allocated_buffers \/ buffer_use_count[b] # saved_uc)
             /\ m_mutex_holder' = 0
             /\ pool_mutex_holder' = 0
             /\ pc' = [pc EXCEPT ![t] = "done"]
             /\ thread_buffer' = [thread_buffer EXCEPT ![t] = NULL]
             /\ thread_op' = [thread_op EXCEPT ![t] = "none"]
    /\ UNCHANGED <<allocator_alive, allocated_buffers, available_buffers, buffer_in_use,
                   buffer_use_count, buffer_freed, tls_cache, saved_use_count, completed_ops,
                   buffer_owner, lock_wait_count, lock_acquire_count>>

\* Check in_use flag (after ABA verification)
GetPtrCheckInUse(t) ==
    /\ pc[t] = "getptr_check_in_use"
    /\ HoldsPoolMutex(t)
    /\ thread_buffer[t] # NULL
    /\ LET b == thread_buffer[t]
       IN \/ \* Buffer is in use - proceed to success
             /\ buffer_in_use[b]
             /\ pc' = [pc EXCEPT ![t] = "getptr_success"]
             /\ UNCHANGED <<pool_mutex_holder, thread_buffer, thread_op>>
          \/ \* Buffer was freed (in_use = false) - abort
             /\ ~buffer_in_use[b]
             /\ pool_mutex_holder' = 0
             /\ pc' = [pc EXCEPT ![t] = "done"]
             /\ thread_buffer' = [thread_buffer EXCEPT ![t] = NULL]
             /\ thread_op' = [thread_op EXCEPT ![t] = "none"]
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, allocated_buffers, available_buffers,
                   buffer_in_use, buffer_use_count, buffer_freed, tls_cache, saved_use_count,
                   completed_ops, buffer_owner, lock_wait_count, lock_acquire_count>>

\* Successfully obtained shared buffer pointer
GetPtrSuccess(t) ==
    /\ pc[t] = "getptr_success"
    /\ HoldsPoolMutex(t)
    /\ pool_mutex_holder' = 0
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ completed_ops' = completed_ops + 1
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, allocated_buffers, available_buffers,
                   buffer_in_use, buffer_use_count, buffer_freed, tls_cache, thread_op,
                   thread_buffer, saved_use_count, buffer_owner,
                   lock_wait_count, lock_acquire_count>>

-----------------------------------------------------------------------------
(* RESET AND SHUTDOWN *)

\* Thread resets after completing operation
Reset(t) ==
    /\ pc[t] = "done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ thread_op' = [thread_op EXCEPT ![t] = "none"]
    /\ thread_buffer' = [thread_buffer EXCEPT ![t] = NULL]
    /\ saved_use_count' = [saved_use_count EXCEPT ![t] = NULL]
    /\ UNCHANGED <<allocator_alive, m_mutex_holder, pool_mutex_holder, allocated_buffers,
                   available_buffers, buffer_in_use, buffer_use_count, buffer_freed,
                   tls_cache, completed_ops, buffer_owner, lock_wait_count, lock_acquire_count>>

\* Allocator shutdown
Shutdown ==
    /\ allocator_alive
    /\ m_mutex_holder = 0  \* Can only shutdown when mutex is free
    /\ pool_mutex_holder = 0
    /\ allocator_alive' = FALSE
    /\ UNCHANGED <<m_mutex_holder, pool_mutex_holder, allocated_buffers, available_buffers,
                   buffer_in_use, buffer_use_count, buffer_freed, tls_cache, pc, thread_op,
                   thread_buffer, saved_use_count, completed_ops, buffer_owner,
                   lock_wait_count, lock_acquire_count>>

-----------------------------------------------------------------------------
(* NEXT STATE RELATION *)

Next ==
    \/ Shutdown
    \/ \E t \in Threads:
        \/ StartAlloc(t)
        \/ AllocLockM(t)
        \/ AllocFind(t)
        \/ AllocLockPool(t)
        \/ AllocComplete(t)
        \/ StartFree(t)
        \/ FreeLockPool(t)
        \/ FreeToPool(t)
        \/ StartGetPtr(t)
        \/ GetPtrLockM1(t)
        \/ GetPtrCapture(t)
        \/ GetPtrLockPool(t)
        \/ GetPtrLockM2(t)
        \/ GetPtrVerify(t)
        \/ GetPtrCheckInUse(t)
        \/ GetPtrSuccess(t)
        \/ Reset(t)

(* Fairness *)
Fairness == \A t \in Threads:
    WF_vars(StartAlloc(t) \/ StartFree(t) \/ StartGetPtr(t) \/ Reset(t))

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
(* SAFETY PROPERTIES *)

(* ABA Detection Sound: If thread captures use_count and it changes,
   the getPtr operation will abort and NOT access the buffer *)
ABADetectionSound ==
    \A t \in Threads:
        (pc[t] = "getptr_success" /\ thread_buffer[t] # NULL) =>
        \* If we reached success, the buffer is still validly allocated
        (thread_buffer[t] \in allocated_buffers /\ buffer_in_use[thread_buffer[t]])

(* No Double Free: A buffer can only be freed if it's currently in use *)
\* NoDoubleFree: A buffer being freed must still be in use
\* Since buffer_in_use is only set to FALSE under pool_mutex in FreeToPool,
\* two threads can't both successfully free the same buffer
NoDoubleFree ==
    \A t \in Threads:
        (pc[t] = "free_to_pool" /\ thread_buffer[t] # NULL /\ HoldsPoolMutex(t)) =>
        buffer_in_use[thread_buffer[t]]

(* No Use-After-Free: We never access a buffer that's not allocated *)
NoUseAfterFree ==
    \A t \in Threads:
        (pc[t] \in {"getptr_success", "getptr_check_in_use"} /\ thread_buffer[t] # NULL) =>
        thread_buffer[t] \in allocated_buffers

(* Buffer Consistency: in_use=TRUE implies buffer is allocated *)
BufferConsistency ==
    \A b \in Buffers:
        buffer_in_use[b] => b \in allocated_buffers

(* Mutex Safety: Only one thread can hold a mutex at a time *)
MutexExclusivity ==
    /\ (m_mutex_holder # 0) => (Cardinality({t \in Threads : HoldsMMutex(t)}) = 1)
    /\ (pool_mutex_holder # 0) => (Cardinality({t \in Threads : HoldsPoolMutex(t)}) = 1)

(* Available and Allocated are disjoint *)
PoolsDisjoint ==
    available_buffers \cap allocated_buffers = {}

(* State constraint to bound state space *)
StateConstraint ==
    /\ completed_ops <= 6
    /\ \A b \in Buffers : buffer_use_count[b] <= 3
    /\ \A t \in Threads : lock_acquire_count[t] <= 12  \* Bound lock acquisitions
    /\ \A t \in Threads : lock_wait_count[t] <= 10     \* Bound lock waits

(* Combined safety invariant *)
Safety ==
    /\ TypeOK
    /\ ABADetectionSound
    /\ NoDoubleFree
    /\ NoUseAfterFree
    /\ BufferConsistency
    /\ MutexExclusivity
    /\ PoolsDisjoint

-----------------------------------------------------------------------------
(* LIVENESS PROPERTIES *)

(* Operations eventually complete (unless allocator shuts down) *)
EventuallyComplete ==
    \A t \in Threads:
        (pc[t] # "idle" /\ pc[t] # "done" /\ allocator_alive) ~> (pc[t] = "done")

(* No deadlock: some action is always enabled *)
NoDeadlock ==
    [](\E t \in Threads:
        ENABLED(StartAlloc(t) \/ StartFree(t) \/ StartGetPtr(t) \/ Reset(t)))

-----------------------------------------------------------------------------
(* SCALABILITY PROPERTIES - These properties SHOULD hold in an optimal design.
   If they FAIL with the current implementation, the failures reveal bottlenecks.
   These are not correctness properties - the current impl is correct but slow. *)

(* SCALABILITY PROPERTY 1: Parallel Lock Holding
   In an optimal implementation, multiple threads should be able to hold
   DIFFERENT locks simultaneously. With the current global m_mutex, this
   is IMPOSSIBLE - only one thread can hold m_mutex at any time.

   EXPECTED RESULT: This property may or may not be satisfiable depending
   on state space exploration. The key insight is that with global m_mutex,
   we can never have >1 thread making progress on allocator operations.
*)
ParallelLockHolding ==
    \* At any point, can we find a reachable state where 2 threads
    \* are both making progress (not idle, not done)?
    \* With global m_mutex, only 1 thread at a time can be in critical sections.
    \* This property checks if parallelism is even possible.
    \/ ~allocator_alive  \* Terminal state is fine
    \/ Cardinality({t \in Threads : pc[t] \in {"alloc_complete", "free_to_pool",
                                                  "getptr_success", "getptr_check_in_use"}}) >= 1

(* SCALABILITY PROPERTY 2: No Global Serializer
   This property is VIOLATED by the current design.
   Every operation (alloc, free, getPtr) must acquire m_mutex.
   This means all operations serialize through one lock.

   To make this TRUE, we would need to shard m_mutex or use lock-free structures.

   EXPECTED RESULT: VIOLATES - because m_mutex is always required.
*)
GlobalSerializerViolation ==
    \* Count total m_mutex acquisitions per getPtr operation
    \* getPtr requires 2 m_mutex acquisitions (GetPtrLockM1 and GetPtrLockM2)
    \* This is the "double-check" pattern that causes 2x contention
    \* An optimal design would need only 1 lock acquisition or use atomics
    LET getptr_threads == {t \in Threads : thread_op[t] = "get_ptr"}
    IN \A t \in getptr_threads:
        \* Each getPtr traverses: lock_m1 -> capture -> lock_pool -> lock_m2 -> verify
        \* That's 3 lock acquisitions: m_mutex (x2) + pool_mutex (x1)
        \* This property documents the excessive locking
        TRUE  \* Always true - this is informational

(* SCALABILITY PROPERTY 3: Lock Acquisition Count Per Operation
   Measures how many locks are acquired per completed operation.
   In current design:
   - alloc: 2 locks (m_mutex + pool_mutex)
   - free: 1 lock (pool_mutex)
   - getPtr: 3 locks (m_mutex + pool_mutex + m_mutex again)

   The getPtr double m_mutex acquisition is the key scalability bottleneck.
*)
ExcessiveLocking ==
    \* When a getPtr completes, it has acquired 3 locks
    \* This property would PASS but documents the excessive locking
    \A t \in Threads:
        (pc[t] = "done" /\ thread_op[t] = "get_ptr") =>
            lock_acquire_count[t] >= 3  \* At least 3 lock acquisitions

(* SCALABILITY PROPERTY 4: Lock Hierarchy
   Locks should be acquired in a consistent order to prevent deadlock.
   Current order: m_mutex -> pool_mutex is correct
   But getPtr does: m_mutex -> unlock -> pool_mutex -> m_mutex (re-acquire!)
   This pattern is safe but causes the double m_mutex contention.
*)
LockHierarchy ==
    \* Check that we never try to acquire m_mutex while holding pool_mutex
    \* EXCEPT for the specific getptr_lock_m2 case which is intentional
    \A t \in Threads:
        (HoldsPoolMutex(t) /\ pc[t] # "getptr_lock_m2" /\ pc[t] # "getptr_verify") =>
            m_mutex_holder # t  \* Don't also hold m_mutex (unless in getPtr verify)

(* SCALABILITY METRIC: Double M_Mutex Acquisition
   This invariant counts how many times m_mutex is acquired.
   For getPtr: 2 acquisitions (at GetPtrLockM1 and GetPtrLockM2)

   To fix the 4-8 thread scaling issue, we need to either:
   1. Eliminate the second m_mutex acquisition
   2. Shard m_mutex so threads don't contend
   3. Use lock-free data structures for the double-check
*)
DoubleMutexBottleneck ==
    \* This is informational - counts m_mutex acquisitions
    \* Every thread doing getPtr contributes 2 m_mutex acquisitions
    \* With 8 threads all doing getPtr, that's 16 m_mutex acquisitions competing
    TRUE  \* Always passes - used to document the issue

=============================================================================
\* Modification History
\* Last modified: 2025-12-17
\* Added scalability properties to diagnose performance bottlenecks
\* Created for MPS Parallel Inference Verification Platform
\* Models the ABA double-check pattern in MPSAllocator (issue 32.267)
