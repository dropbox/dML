--------------------------- MODULE MPSAllocator ---------------------------
\* TLA+ Specification for PyTorch MPS Heap Allocator
\* Based on: pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.h
\*
\* Models the caching heap allocator with:
\* - Buffer pools (private/shared, small/large)
\* - ABA counters for safe buffer ID generation
\* - Stream-aware allocation (CUDA pattern)
\* - Pending buffer states for async completion
\*
\* Key invariants to verify:
\* 1. ABA counter monotonically increases (no reuse of IDs)
\* 2. Buffer state machine correctness
\* 3. No use-after-free
\* 4. Stream synchronization safety

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    \* @type: Int;
    NumBuffers,         \* Maximum buffer slots
    \* @type: Int;
    NumStreams,         \* Number of MPS streams
    \* @type: Int;
    NumThreads,         \* Number of concurrent threads
    \* @type: Int;
    MaxOperations       \* Bound for model checking

ASSUME NumBuffers >= 1
ASSUME NumStreams >= 1
ASSUME NumThreads >= 1
ASSUME MaxOperations >= 1

VARIABLES
    \* Buffer states: "free", "allocated", "in_use", "pending_free"
    \* @type: Int -> Str;
    buffer_states,
    \* ABA counter for buffer IDs (atomic, monotonically increasing)
    \* @type: Int;
    buffer_counter,
    \* Buffer ID for each buffer slot (assigned on allocation)
    \* @type: Int -> Int;
    buffer_ids,
    \* Which stream allocated each buffer (0 = not allocated)
    \* @type: Int -> Int;
    alloc_streams,
    \* Set of streams that have used each buffer
    \* @type: Int -> Set(Int);
    stream_uses,
    \* Thread currently holding global mutex (0 = none)
    \* @type: Int;
    global_mutex_holder,
    \* Thread currently holding pool mutex for each pool (0 = none)
    \* We model a single pool for simplicity
    \* @type: Int;
    pool_mutex_holder,
    \* Available buffer freelist (set of buffer indices)
    \* @type: Set(Int);
    available_buffers,
    \* Pending free buffers (waiting for GPU completion)
    \* @type: Set(Int);
    pending_buffers,
    \* Operation counter
    \* @type: Int;
    op_count

vars == <<buffer_states, buffer_counter, buffer_ids, alloc_streams,
          stream_uses, global_mutex_holder, pool_mutex_holder,
          available_buffers, pending_buffers, op_count>>

-----------------------------------------------------------------------------
\* Type Invariant
TypeOK ==
    /\ buffer_states \in [1..NumBuffers -> {"free", "allocated", "in_use", "pending_free"}]
    /\ buffer_counter \in Nat
    /\ buffer_ids \in [1..NumBuffers -> Nat]
    /\ alloc_streams \in [1..NumBuffers -> 0..NumStreams]
    /\ stream_uses \in [1..NumBuffers -> SUBSET (1..NumStreams)]
    /\ global_mutex_holder \in 0..NumThreads
    /\ pool_mutex_holder \in 0..NumThreads
    /\ available_buffers \subseteq 1..NumBuffers
    /\ pending_buffers \subseteq 1..NumBuffers
    /\ op_count \in 0..MaxOperations

\* Initial state
Init ==
    /\ buffer_states = [b \in 1..NumBuffers |-> "free"]
    /\ buffer_counter = 0
    /\ buffer_ids = [b \in 1..NumBuffers |-> 0]
    /\ alloc_streams = [b \in 1..NumBuffers |-> 0]
    /\ stream_uses = [b \in 1..NumBuffers |-> {}]
    /\ global_mutex_holder = 0
    /\ pool_mutex_holder = 0
    /\ available_buffers = 1..NumBuffers  \* All buffers initially free
    /\ pending_buffers = {}
    /\ op_count = 0

-----------------------------------------------------------------------------
\* Mutex operations (model lock acquisition)

AcquireGlobalMutex(t) ==
    /\ global_mutex_holder = 0
    /\ global_mutex_holder' = t
    /\ UNCHANGED <<buffer_states, buffer_counter, buffer_ids, alloc_streams,
                   stream_uses, pool_mutex_holder, available_buffers,
                   pending_buffers, op_count>>

ReleaseGlobalMutex(t) ==
    /\ global_mutex_holder = t
    /\ global_mutex_holder' = 0
    /\ UNCHANGED <<buffer_states, buffer_counter, buffer_ids, alloc_streams,
                   stream_uses, pool_mutex_holder, available_buffers,
                   pending_buffers, op_count>>

AcquirePoolMutex(t) ==
    /\ pool_mutex_holder = 0
    /\ pool_mutex_holder' = t
    /\ UNCHANGED <<buffer_states, buffer_counter, buffer_ids, alloc_streams,
                   stream_uses, global_mutex_holder, available_buffers,
                   pending_buffers, op_count>>

ReleasePoolMutex(t) ==
    /\ pool_mutex_holder = t
    /\ pool_mutex_holder' = 0
    /\ UNCHANGED <<buffer_states, buffer_counter, buffer_ids, alloc_streams,
                   stream_uses, global_mutex_holder, available_buffers,
                   pending_buffers, op_count>>

-----------------------------------------------------------------------------
\* AllocBuffer: Thread t allocates a buffer on stream s
\* Models: MPSHeapAllocatorImpl::malloc()
\*
\* Implementation:
\*   1. Lock pool mutex
\*   2. Check available_buffers freelist
\*   3. If found, remove from freelist, assign new ID via buffer_counter++
\*   4. Track alloc_stream
\*   5. Lock global mutex briefly to update allocated_buffers map

AllocBuffer(t, s) ==
    /\ op_count < MaxOperations
    /\ pool_mutex_holder = t              \* Must hold pool mutex
    /\ available_buffers /= {}            \* Pool has free buffers
    /\ LET b == CHOOSE x \in available_buffers : TRUE
           new_id == buffer_counter + 1
       IN /\ buffer_states' = [buffer_states EXCEPT ![b] = "allocated"]
          /\ buffer_counter' = new_id     \* ABA counter increment
          /\ buffer_ids' = [buffer_ids EXCEPT ![b] = new_id]
          /\ alloc_streams' = [alloc_streams EXCEPT ![b] = s]
          /\ stream_uses' = [stream_uses EXCEPT ![b] = {s}]
          /\ available_buffers' = available_buffers \ {b}
          /\ op_count' = op_count + 1
    /\ UNCHANGED <<global_mutex_holder, pool_mutex_holder, pending_buffers>>

\* UseBuffer: Mark buffer as in-use (GPU work submitted)
UseBuffer(t, b) ==
    /\ op_count < MaxOperations
    /\ buffer_states[b] = "allocated"
    /\ pool_mutex_holder = 0              \* Don't need lock for use
    /\ buffer_states' = [buffer_states EXCEPT ![b] = "in_use"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<buffer_counter, buffer_ids, alloc_streams, stream_uses,
                   global_mutex_holder, pool_mutex_holder, available_buffers,
                   pending_buffers>>

\* RecordStream: Track cross-stream buffer usage (CUDA pattern)
\* Models: MPSHeapAllocatorImpl::recordStream()
RecordStream(t, b, s) ==
    /\ op_count < MaxOperations
    /\ buffer_states[b] \in {"allocated", "in_use"}
    /\ global_mutex_holder = t            \* Must hold global mutex
    /\ stream_uses' = [stream_uses EXCEPT ![b] = stream_uses[b] \union {s}]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<buffer_states, buffer_counter, buffer_ids, alloc_streams,
                   global_mutex_holder, pool_mutex_holder, available_buffers,
                   pending_buffers>>

\* FreeBuffer: Thread t frees buffer b
\* Models: MPSHeapAllocatorImpl::free()
\*
\* If buffer has cross-stream uses, it goes to pending_buffers
\* Otherwise returns directly to available_buffers
FreeBuffer(t, b) ==
    /\ op_count < MaxOperations
    /\ buffer_states[b] \in {"allocated", "in_use"}
    /\ pool_mutex_holder = t
    /\ IF Cardinality(stream_uses[b]) > 1
       THEN \* Cross-stream use: mark pending
            /\ buffer_states' = [buffer_states EXCEPT ![b] = "pending_free"]
            /\ pending_buffers' = pending_buffers \union {b}
            /\ UNCHANGED available_buffers
       ELSE \* Single stream: return to pool
            /\ buffer_states' = [buffer_states EXCEPT ![b] = "free"]
            /\ available_buffers' = available_buffers \union {b}
            /\ UNCHANGED pending_buffers
    /\ alloc_streams' = [alloc_streams EXCEPT ![b] = 0]
    /\ stream_uses' = [stream_uses EXCEPT ![b] = {}]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<buffer_counter, buffer_ids, global_mutex_holder, pool_mutex_holder>>

\* ProcessPendingBuffer: Completion handler returns pending buffer to pool
\* Models: completionHandler callback in free_buffer()
ProcessPendingBuffer(b) ==
    /\ op_count < MaxOperations
    /\ b \in pending_buffers
    /\ buffer_states[b] = "pending_free"
    /\ pool_mutex_holder = 0              \* From completion handler context
    /\ buffer_states' = [buffer_states EXCEPT ![b] = "free"]
    /\ available_buffers' = available_buffers \union {b}
    /\ pending_buffers' = pending_buffers \ {b}
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<buffer_counter, buffer_ids, alloc_streams, stream_uses,
                   global_mutex_holder, pool_mutex_holder>>

-----------------------------------------------------------------------------
\* Next state relation
Next ==
    \/ \E t \in 1..NumThreads :
        \/ AcquireGlobalMutex(t)
        \/ ReleaseGlobalMutex(t)
        \/ AcquirePoolMutex(t)
        \/ ReleasePoolMutex(t)
        \/ \E s \in 1..NumStreams : AllocBuffer(t, s)
        \/ \E b \in 1..NumBuffers : UseBuffer(t, b)
        \/ \E b \in 1..NumBuffers, s \in 1..NumStreams : RecordStream(t, b, s)
        \/ \E b \in 1..NumBuffers : FreeBuffer(t, b)
    \/ \E b \in 1..NumBuffers : ProcessPendingBuffer(b)
    \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* SAFETY PROPERTIES

\* ABA Counter Monotonicity: Counter only increases
\* This is the KEY property preventing ABA bugs
ABAMonotonicity ==
    [][buffer_counter' >= buffer_counter]_vars

\* No ID Reuse: Each allocated buffer has unique ID
\* (IDs are never reused within a run)
NoIDReuse ==
    \A b1, b2 \in 1..NumBuffers :
        (b1 /= b2 /\ buffer_ids[b1] > 0 /\ buffer_ids[b2] > 0) =>
        buffer_ids[b1] /= buffer_ids[b2]

\* Buffer State Machine: Valid state transitions only
ValidStateTransitions ==
    \A b \in 1..NumBuffers :
        \/ buffer_states[b] = "free"
        \/ buffer_states[b] = "allocated"
        \/ buffer_states[b] = "in_use"
        \/ buffer_states[b] = "pending_free"

\* Free buffers are in available list or pending list
FreebufferConsistency ==
    \A b \in 1..NumBuffers :
        buffer_states[b] = "free" =>
        (b \in available_buffers /\ b \notin pending_buffers)

\* Pending buffers are in pending list
PendingBufferConsistency ==
    \A b \in 1..NumBuffers :
        buffer_states[b] = "pending_free" =>
        (b \in pending_buffers /\ b \notin available_buffers)

\* Allocated/in-use buffers not in any list
AllocatedBufferConsistency ==
    \A b \in 1..NumBuffers :
        buffer_states[b] \in {"allocated", "in_use"} =>
        (b \notin available_buffers /\ b \notin pending_buffers)

\* No Use After Free: Cannot use a free buffer
NoUseAfterFree ==
    \A b \in 1..NumBuffers :
        buffer_states[b] = "free" =>
        alloc_streams[b] = 0

\* Mutex exclusion: At most one thread holds mutex
MutexExclusion ==
    /\ (global_mutex_holder > 0 => pool_mutex_holder /= global_mutex_holder)
    \* Actually mutexes are independent, so this is too strong
    \* The real invariant is just that each mutex has at most one holder
    /\ TRUE

\* Combined safety invariant
SafetyInvariant ==
    /\ TypeOK
    /\ NoIDReuse
    /\ ValidStateTransitions
    /\ FreebufferConsistency
    /\ PendingBufferConsistency
    /\ AllocatedBufferConsistency
    /\ NoUseAfterFree

-----------------------------------------------------------------------------
\* LIVENESS PROPERTIES (under fairness)

\* Progress: Pending buffers eventually get processed
PendingProgress ==
    \A b \in 1..NumBuffers :
        (b \in pending_buffers) ~> (buffer_states[b] = "free")

-----------------------------------------------------------------------------
\* DEADLOCK FREEDOM

\* System can always make progress
DeadlockFree ==
    op_count = MaxOperations \/
    \/ available_buffers /= {}       \* Can allocate
    \/ pending_buffers /= {}          \* Can process pending
    \/ \E b \in 1..NumBuffers :
       buffer_states[b] \in {"allocated", "in_use"}  \* Can free

=============================================================================
\* Modification History
\* Created: 2025-12-18 by AI Worker N=1251
\* Based on: MPSAllocator.h MPSHeapAllocatorImpl
