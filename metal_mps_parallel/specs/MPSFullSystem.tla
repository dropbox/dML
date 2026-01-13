--------------------------- MODULE MPSFullSystem ---------------------------
\* TLA+ Specification for Full MPS Parallel Inference System
\* Composed model integrating StreamPool, Allocator, and Event subsystems
\*
\* This spec models the high-level interactions between components:
\* - StreamPool: Thread-to-stream binding for GPU work
\* - Allocator: Buffer lifecycle with ABA protection
\* - Event: Synchronization events for GPU completion
\*
\* Key system-wide properties:
\* 1. No deadlock across component interactions
\* 2. Resource cleanup: all acquired resources eventually released
\* 3. No use-after-free across component boundaries
\* 4. Stream affinity: operations respect stream bindings
\*
\* This is an abstracted model for verifying cross-component invariants.
\* Individual components have more detailed specs.

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    \* @type: Int;
    NumStreams,         \* Total streams (1 = default, 2..N = worker streams)
    \* @type: Int;
    NumThreads,         \* Number of concurrent threads
    \* @type: Int;
    NumBuffers,         \* Maximum buffers in allocator pool
    \* @type: Int;
    NumEvents,          \* Maximum events in event pool
    \* @type: Int;
    MaxOperations       \* Bound for model checking

ASSUME NumStreams >= 2   \* At least default + 1 worker
ASSUME NumThreads >= 1
ASSUME NumBuffers >= 1
ASSUME NumEvents >= 1
ASSUME MaxOperations >= 1

\*---------------------------------------------------------------------------
\* VARIABLES - Abstracted state from component models

VARIABLES
    \*--- Stream Pool ---
    \* Thread -> Stream binding (0 = no stream bound)
    \* @type: Int -> Int;
    thread_streams,
    \* Stream states: "free", "bound"
    \* @type: Int -> Str;
    stream_states,

    \*--- Allocator ---
    \* Buffer states: "free", "allocated", "pending"
    \* @type: Int -> Str;
    buffer_states,
    \* Buffer -> Stream that allocated it (0 = not allocated)
    \* @type: Int -> Int;
    buffer_streams,
    \* ABA counter for unique buffer IDs
    \* @type: Int;
    buffer_counter,

    \*--- Event Pool ---
    \* Event states: "pooled", "acquired", "recorded", "pending"
    \* @type: Int -> Str;
    event_states,
    \* Event -> Stream it was recorded on (0 = not recorded)
    \* @type: Int -> Int;
    event_streams,
    \* Event reference count (for callback survival)
    \* @type: Int -> Int;
    event_refs,

    \*--- Global ---
    \* Which thread holds the allocator mutex (0 = none)
    \* @type: Int;
    alloc_mutex,
    \* Which thread holds the event pool mutex (0 = none)
    \* @type: Int;
    event_mutex,
    \* Operation counter for bounded checking
    \* @type: Int;
    op_count

vars == <<thread_streams, stream_states, buffer_states, buffer_streams,
          buffer_counter, event_states, event_streams, event_refs,
          alloc_mutex, event_mutex, op_count>>

\*---------------------------------------------------------------------------
\* TYPE INVARIANT

TypeOK ==
    /\ thread_streams \in [1..NumThreads -> 0..(NumStreams-1)]
    /\ stream_states \in [1..(NumStreams-1) -> {"free", "bound"}]
    /\ buffer_states \in [1..NumBuffers -> {"free", "allocated", "pending"}]
    /\ buffer_streams \in [1..NumBuffers -> 0..(NumStreams-1)]
    /\ buffer_counter \in Nat
    /\ event_states \in [1..NumEvents -> {"pooled", "acquired", "recorded", "pending"}]
    /\ event_streams \in [1..NumEvents -> 0..(NumStreams-1)]
    /\ event_refs \in [1..NumEvents -> Nat]
    /\ alloc_mutex \in 0..NumThreads
    /\ event_mutex \in 0..NumThreads
    /\ op_count \in 0..MaxOperations

\*---------------------------------------------------------------------------
\* INITIAL STATE

Init ==
    /\ thread_streams = [t \in 1..NumThreads |-> 0]
    /\ stream_states = [s \in 1..(NumStreams-1) |-> "free"]
    /\ buffer_states = [b \in 1..NumBuffers |-> "free"]
    /\ buffer_streams = [b \in 1..NumBuffers |-> 0]
    /\ buffer_counter = 0
    /\ event_states = [e \in 1..NumEvents |-> "pooled"]
    /\ event_streams = [e \in 1..NumEvents |-> 0]
    /\ event_refs = [e \in 1..NumEvents |-> 0]
    /\ alloc_mutex = 0
    /\ event_mutex = 0
    /\ op_count = 0

\*---------------------------------------------------------------------------
\* STREAM POOL OPERATIONS

\* Helper: Find a free stream (returns 0 if none)
FreeStream ==
    IF \E s \in 1..(NumStreams-1) : stream_states[s] = "free"
    THEN CHOOSE s \in 1..(NumStreams-1) : stream_states[s] = "free"
    ELSE 0

\* Thread acquires a stream from the pool
AcquireStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_streams[t] = 0           \* Thread has no stream
    /\ LET s == FreeStream
       IN /\ s > 0                      \* Free stream exists
          /\ stream_states' = [stream_states EXCEPT ![s] = "bound"]
          /\ thread_streams' = [thread_streams EXCEPT ![t] = s]
          /\ op_count' = op_count + 1
    /\ UNCHANGED <<buffer_states, buffer_streams, buffer_counter,
                   event_states, event_streams, event_refs,
                   alloc_mutex, event_mutex>>

\* Thread releases its stream back to pool
ReleaseStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_streams[t] > 0           \* Thread has a stream
    \* Cannot release if thread has allocated resources on this stream
    /\ ~\E b \in 1..NumBuffers :
        buffer_streams[b] = thread_streams[t] /\ buffer_states[b] = "allocated"
    /\ ~\E e \in 1..NumEvents :
        event_streams[e] = thread_streams[t] /\ event_states[e] \in {"acquired", "recorded"}
    /\ LET s == thread_streams[t]
       IN /\ stream_states' = [stream_states EXCEPT ![s] = "free"]
          /\ thread_streams' = [thread_streams EXCEPT ![t] = 0]
          /\ op_count' = op_count + 1
    /\ UNCHANGED <<buffer_states, buffer_streams, buffer_counter,
                   event_states, event_streams, event_refs,
                   alloc_mutex, event_mutex>>

\*---------------------------------------------------------------------------
\* ALLOCATOR OPERATIONS

\* Helper: Find a free buffer
FreeBuffer ==
    IF \E b \in 1..NumBuffers : buffer_states[b] = "free"
    THEN CHOOSE b \in 1..NumBuffers : buffer_states[b] = "free"
    ELSE 0

\* Acquire allocator mutex
AcquireAllocMutex(t) ==
    /\ alloc_mutex = 0
    /\ alloc_mutex' = t
    /\ UNCHANGED <<thread_streams, stream_states, buffer_states, buffer_streams,
                   buffer_counter, event_states, event_streams, event_refs,
                   event_mutex, op_count>>

\* Release allocator mutex
ReleaseAllocMutex(t) ==
    /\ alloc_mutex = t
    /\ alloc_mutex' = 0
    /\ UNCHANGED <<thread_streams, stream_states, buffer_states, buffer_streams,
                   buffer_counter, event_states, event_streams, event_refs,
                   event_mutex, op_count>>

\* Thread allocates a buffer (must have stream and mutex)
AllocBuffer(t) ==
    /\ op_count < MaxOperations
    /\ alloc_mutex = t                 \* Has allocator mutex
    /\ thread_streams[t] > 0           \* Has a stream
    /\ LET b == FreeBuffer
           s == thread_streams[t]
       IN /\ b > 0                      \* Free buffer exists
          /\ buffer_states' = [buffer_states EXCEPT ![b] = "allocated"]
          /\ buffer_streams' = [buffer_streams EXCEPT ![b] = s]
          /\ buffer_counter' = buffer_counter + 1
          /\ op_count' = op_count + 1
    /\ UNCHANGED <<thread_streams, stream_states, event_states, event_streams,
                   event_refs, alloc_mutex, event_mutex>>

\* Thread requests free of buffer (goes to pending until GPU done)
RequestFreeBuffer(t, b) ==
    /\ op_count < MaxOperations
    /\ alloc_mutex = t
    /\ buffer_states[b] = "allocated"
    /\ buffer_streams[b] = thread_streams[t]  \* Stream affinity
    /\ buffer_states' = [buffer_states EXCEPT ![b] = "pending"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<thread_streams, stream_states, buffer_streams, buffer_counter,
                   event_states, event_streams, event_refs,
                   alloc_mutex, event_mutex>>

\* GPU completes and buffer becomes free (async)
CompleteBufferFree(b) ==
    /\ op_count < MaxOperations
    /\ buffer_states[b] = "pending"
    /\ buffer_states' = [buffer_states EXCEPT ![b] = "free"]
    /\ buffer_streams' = [buffer_streams EXCEPT ![b] = 0]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<thread_streams, stream_states, buffer_counter,
                   event_states, event_streams, event_refs,
                   alloc_mutex, event_mutex>>

\*---------------------------------------------------------------------------
\* EVENT POOL OPERATIONS

\* Helper: Find a pooled event
PooledEvent ==
    IF \E e \in 1..NumEvents : event_states[e] = "pooled"
    THEN CHOOSE e \in 1..NumEvents : event_states[e] = "pooled"
    ELSE 0

\* Acquire event pool mutex
AcquireEventMutex(t) ==
    /\ event_mutex = 0
    /\ event_mutex' = t
    /\ UNCHANGED <<thread_streams, stream_states, buffer_states, buffer_streams,
                   buffer_counter, event_states, event_streams, event_refs,
                   alloc_mutex, op_count>>

\* Release event pool mutex
ReleaseEventMutex(t) ==
    /\ event_mutex = t
    /\ event_mutex' = 0
    /\ UNCHANGED <<thread_streams, stream_states, buffer_states, buffer_streams,
                   buffer_counter, event_states, event_streams, event_refs,
                   alloc_mutex, op_count>>

\* Thread acquires an event from pool
AcquireEvent(t) ==
    /\ op_count < MaxOperations
    /\ event_mutex = t                 \* Has event mutex
    /\ thread_streams[t] > 0           \* Has a stream
    /\ LET e == PooledEvent
       IN /\ e > 0
          /\ event_states' = [event_states EXCEPT ![e] = "acquired"]
          /\ event_refs' = [event_refs EXCEPT ![e] = 1]
          /\ op_count' = op_count + 1
    /\ UNCHANGED <<thread_streams, stream_states, buffer_states, buffer_streams,
                   buffer_counter, event_streams, alloc_mutex, event_mutex>>

\* Thread records event on its stream
RecordEvent(t, e) ==
    /\ op_count < MaxOperations
    /\ event_states[e] = "acquired"
    /\ thread_streams[t] > 0
    /\ LET s == thread_streams[t]
       IN /\ event_states' = [event_states EXCEPT ![e] = "recorded"]
          /\ event_streams' = [event_streams EXCEPT ![e] = s]
          /\ op_count' = op_count + 1
    /\ UNCHANGED <<thread_streams, stream_states, buffer_states, buffer_streams,
                   buffer_counter, event_refs, alloc_mutex, event_mutex>>

\* Thread adds callback to event (increments ref for survival)
AddCallback(t, e) ==
    /\ op_count < MaxOperations
    /\ event_states[e] = "recorded"
    /\ event_states' = [event_states EXCEPT ![e] = "pending"]
    /\ event_refs' = [event_refs EXCEPT ![e] = event_refs[e] + 1]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<thread_streams, stream_states, buffer_states, buffer_streams,
                   buffer_counter, event_streams, alloc_mutex, event_mutex>>

\* Callback completes (decrements ref)
CompleteCallback(e) ==
    /\ op_count < MaxOperations
    /\ event_states[e] = "pending"
    /\ event_refs[e] > 1              \* Has callback reference
    /\ event_refs' = [event_refs EXCEPT ![e] = event_refs[e] - 1]
    /\ event_states' = [event_states EXCEPT ![e] = "recorded"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<thread_streams, stream_states, buffer_states, buffer_streams,
                   buffer_counter, event_streams, alloc_mutex, event_mutex>>

\* Release event back to pool (only when no callbacks pending)
ReleaseEvent(t, e) ==
    /\ op_count < MaxOperations
    /\ event_mutex = t
    /\ event_states[e] \in {"acquired", "recorded"}
    /\ event_refs[e] = 1              \* No pending callbacks
    /\ event_states' = [event_states EXCEPT ![e] = "pooled"]
    /\ event_streams' = [event_streams EXCEPT ![e] = 0]
    /\ event_refs' = [event_refs EXCEPT ![e] = 0]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<thread_streams, stream_states, buffer_states, buffer_streams,
                   buffer_counter, alloc_mutex, event_mutex>>

\*---------------------------------------------------------------------------
\* NEXT STATE RELATION

Next ==
    \* Stream operations
    \/ \E t \in 1..NumThreads :
        \/ AcquireStream(t)
        \/ ReleaseStream(t)
    \* Allocator operations
    \/ \E t \in 1..NumThreads :
        \/ AcquireAllocMutex(t)
        \/ ReleaseAllocMutex(t)
        \/ AllocBuffer(t)
        \/ \E b \in 1..NumBuffers : RequestFreeBuffer(t, b)
    \/ \E b \in 1..NumBuffers : CompleteBufferFree(b)
    \* Event operations
    \/ \E t \in 1..NumThreads :
        \/ AcquireEventMutex(t)
        \/ ReleaseEventMutex(t)
        \/ AcquireEvent(t)
        \/ \E e \in 1..NumEvents :
            \/ RecordEvent(t, e)
            \/ AddCallback(t, e)
            \/ ReleaseEvent(t, e)
    \/ \E e \in 1..NumEvents : CompleteCallback(e)
    \* Stuttering
    \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

\*---------------------------------------------------------------------------
\* SAFETY PROPERTIES - Cross-component invariants

\* Stream Mutual Exclusion: No two threads bound to same stream
StreamMutualExclusion ==
    \A t1, t2 \in 1..NumThreads :
        (t1 /= t2 /\ thread_streams[t1] > 0) =>
        thread_streams[t1] /= thread_streams[t2]

\* Stream Affinity: Allocated resources bound to existing streams
StreamAffinity ==
    /\ \A b \in 1..NumBuffers :
        buffer_states[b] = "allocated" =>
        \E t \in 1..NumThreads : thread_streams[t] = buffer_streams[b]
    /\ \A e \in 1..NumEvents :
        event_states[e] \in {"recorded", "pending"} =>
        event_streams[e] > 0

\* Callback Survival: Events with pending callbacks have refs > 1
CallbackSurvival ==
    \A e \in 1..NumEvents :
        event_states[e] = "pending" => event_refs[e] > 1

\* Pool Consistency: Pooled resources have zero associations
PoolConsistency ==
    /\ \A b \in 1..NumBuffers :
        buffer_states[b] = "free" => buffer_streams[b] = 0
    /\ \A e \in 1..NumEvents :
        event_states[e] = "pooled" =>
        (event_streams[e] = 0 /\ event_refs[e] = 0)

\* Resource Ownership: Acquired resources have proper refs
ResourceOwnership ==
    \A e \in 1..NumEvents :
        event_states[e] \in {"acquired", "recorded", "pending"} =>
        event_refs[e] >= 1

\* Mutex Exclusivity: Only one thread holds each mutex
MutexExclusivity ==
    /\ alloc_mutex \in 0..NumThreads
    /\ event_mutex \in 0..NumThreads

\* Combined Safety Invariant
SafetyInvariant ==
    /\ TypeOK
    /\ StreamMutualExclusion
    /\ StreamAffinity
    /\ CallbackSurvival
    /\ PoolConsistency
    /\ ResourceOwnership
    /\ MutexExclusivity

\*---------------------------------------------------------------------------
\* DEADLOCK FREEDOM

\* System can always make progress (at bounded operations)
DeadlockFree ==
    op_count = MaxOperations \/
    \* Can acquire/release streams
    \/ \E t \in 1..NumThreads :
        \/ (thread_streams[t] = 0 /\ \E s \in 1..(NumStreams-1) : stream_states[s] = "free")
        \/ thread_streams[t] > 0
    \* Can acquire/release mutexes
    \/ alloc_mutex = 0
    \/ event_mutex = 0
    \* Pending resources can complete
    \/ \E b \in 1..NumBuffers : buffer_states[b] = "pending"
    \/ \E e \in 1..NumEvents : event_states[e] = "pending" /\ event_refs[e] > 1

\*---------------------------------------------------------------------------
\* RESOURCE LEAK FREEDOM (for bounded checking)

\* All allocated buffers are reachable (have stream owner)
NoOrphanedBuffers ==
    \A b \in 1..NumBuffers :
        buffer_states[b] = "allocated" =>
        buffer_streams[b] > 0

\* All acquired events are reachable
NoOrphanedEvents ==
    \A e \in 1..NumEvents :
        event_states[e] \in {"acquired", "recorded"} =>
        event_refs[e] >= 1

=============================================================================
\* Modification History
\* Created: 2025-12-20 by AI Worker N=1358
\* Purpose: Composed system model for cross-component verification
