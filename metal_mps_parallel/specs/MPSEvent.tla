--------------------------- MODULE MPSEvent ---------------------------
\* TLA+ Specification for PyTorch MPS Event Pool and Lifecycle
\* Based on: pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.h
\*
\* Models the event pool mechanism with:
\* - Event pool freelist (m_pool)
\* - In-use event tracking (m_in_use_events via shared_ptr)
\* - MTLSharedEvent signal counter semantics
\* - Callback survival pattern
\*
\* Key invariants to verify:
\* 1. Event IDs are unique while in use
\* 2. Callback survival: events live until callbacks complete
\* 3. Signal counter monotonicity
\* 4. No use-after-release bugs

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    \* @type: Int;
    NumEvents,          \* Maximum events in pool
    \* @type: Int;
    NumStreams,         \* Number of MPS streams
    \* @type: Int;
    NumThreads,         \* Number of concurrent threads
    \* @type: Int;
    MaxOperations       \* Bound for model checking

ASSUME NumEvents >= 1
ASSUME NumStreams >= 1
ASSUME NumThreads >= 1
ASSUME MaxOperations >= 1

VARIABLES
    \* Event states: "pooled", "acquired", "recorded", "signaled", "pending_callback"
    \* @type: Int -> Str;
    event_states,
    \* Global event ID counter (atomic, monotonically increasing)
    \* @type: Int;
    event_counter,
    \* Assigned ID for each event (0 = not assigned)
    \* @type: Int -> Int;
    event_ids,
    \* Signal counter for each event (MTLSharedEvent semantics)
    \* @type: Int -> Int;
    signal_counters,
    \* Which stream recorded each event (0 = not recorded)
    \* @type: Int -> Int;
    recording_streams,
    \* Reference count for callback survival (shared_ptr pattern)
    \* 1 = in_use_events holds reference, +1 for each pending callback
    \* @type: Int -> Int;
    ref_counts,
    \* Pool freelist (set of event indices currently in pool)
    \* @type: Set(Int);
    pool_freelist,
    \* In-use events map (set of event indices currently acquired)
    \* @type: Set(Int);
    in_use_events,
    \* Events with pending callbacks (waiting for GPU signal)
    \* @type: Set(Int);
    pending_callbacks,
    \* Pool mutex holder (0 = none)
    \* @type: Int;
    pool_mutex_holder,
    \* Per-event mutex holder (0 = none)
    \* @type: Int -> Int;
    event_mutex_holder,
    \* Operation counter
    \* @type: Int;
    op_count

vars == <<event_states, event_counter, event_ids, signal_counters,
          recording_streams, ref_counts, pool_freelist, in_use_events,
          pending_callbacks, pool_mutex_holder, event_mutex_holder, op_count>>

-----------------------------------------------------------------------------
\* Type Invariant
TypeOK ==
    /\ event_states \in [1..NumEvents -> {"pooled", "acquired", "recorded", "signaled", "pending_callback"}]
    /\ event_counter \in Nat
    /\ event_ids \in [1..NumEvents -> Nat]
    /\ signal_counters \in [1..NumEvents -> Nat]
    /\ recording_streams \in [1..NumEvents -> 0..NumStreams]
    /\ ref_counts \in [1..NumEvents -> Nat]
    /\ pool_freelist \subseteq 1..NumEvents
    /\ in_use_events \subseteq 1..NumEvents
    /\ pending_callbacks \subseteq 1..NumEvents
    /\ pool_mutex_holder \in 0..NumThreads
    /\ event_mutex_holder \in [1..NumEvents -> 0..NumThreads]
    /\ op_count \in 0..MaxOperations

\* Initial state
Init ==
    /\ event_states = [e \in 1..NumEvents |-> "pooled"]
    /\ event_counter = 0
    /\ event_ids = [e \in 1..NumEvents |-> 0]
    /\ signal_counters = [e \in 1..NumEvents |-> 0]
    /\ recording_streams = [e \in 1..NumEvents |-> 0]
    /\ ref_counts = [e \in 1..NumEvents |-> 0]
    /\ pool_freelist = 1..NumEvents       \* All events in pool initially
    /\ in_use_events = {}
    /\ pending_callbacks = {}
    /\ pool_mutex_holder = 0
    /\ event_mutex_holder = [e \in 1..NumEvents |-> 0]
    /\ op_count = 0

-----------------------------------------------------------------------------
\* Mutex operations

AcquirePoolMutex(t) ==
    /\ pool_mutex_holder = 0
    /\ pool_mutex_holder' = t
    /\ UNCHANGED <<event_states, event_counter, event_ids, signal_counters,
                   recording_streams, ref_counts, pool_freelist, in_use_events,
                   pending_callbacks, event_mutex_holder, op_count>>

ReleasePoolMutex(t) ==
    /\ pool_mutex_holder = t
    /\ pool_mutex_holder' = 0
    /\ UNCHANGED <<event_states, event_counter, event_ids, signal_counters,
                   recording_streams, ref_counts, pool_freelist, in_use_events,
                   pending_callbacks, event_mutex_holder, op_count>>

AcquireEventMutex(t, e) ==
    /\ event_mutex_holder[e] = 0
    /\ event_mutex_holder' = [event_mutex_holder EXCEPT ![e] = t]
    /\ UNCHANGED <<event_states, event_counter, event_ids, signal_counters,
                   recording_streams, ref_counts, pool_freelist, in_use_events,
                   pending_callbacks, pool_mutex_holder, op_count>>

ReleaseEventMutex(t, e) ==
    /\ event_mutex_holder[e] = t
    /\ event_mutex_holder' = [event_mutex_holder EXCEPT ![e] = 0]
    /\ UNCHANGED <<event_states, event_counter, event_ids, signal_counters,
                   recording_streams, ref_counts, pool_freelist, in_use_events,
                   pending_callbacks, pool_mutex_holder, op_count>>

-----------------------------------------------------------------------------
\* AcquireEvent: Get event from pool
\* Models: MPSEventPool::acquireEvent()
\*
\* Implementation:
\*   1. Lock pool mutex
\*   2. If pool empty, create new event
\*   3. Otherwise pop from pool freelist
\*   4. Assign new ID via event_counter++
\*   5. Add to in_use_events map
\*   6. Return event

AcquireEvent(t) ==
    /\ op_count < MaxOperations
    /\ pool_mutex_holder = t              \* Must hold pool mutex
    /\ pool_freelist /= {}                \* Pool has events
    /\ LET e == CHOOSE x \in pool_freelist : TRUE
           new_id == event_counter + 1
       IN /\ event_states' = [event_states EXCEPT ![e] = "acquired"]
          /\ event_counter' = new_id
          /\ event_ids' = [event_ids EXCEPT ![e] = new_id]
          /\ ref_counts' = [ref_counts EXCEPT ![e] = 1]  \* in_use_events ref
          /\ pool_freelist' = pool_freelist \ {e}
          /\ in_use_events' = in_use_events \union {e}
          /\ op_count' = op_count + 1
    /\ UNCHANGED <<signal_counters, recording_streams, pending_callbacks,
                   pool_mutex_holder, event_mutex_holder>>

\* ReleaseEvent: Return event to pool
\* Models: MPSEventPool::releaseEvent() via m_default_deleter
\*
\* CALLBACK SURVIVAL: Only release when ref_count = 1 (no pending callbacks)

ReleaseEvent(t, e) ==
    /\ op_count < MaxOperations
    /\ pool_mutex_holder = t
    /\ e \in in_use_events
    /\ ref_counts[e] = 1                  \* No pending callbacks
    /\ event_states[e] \in {"acquired", "recorded", "signaled"}
    /\ event_states' = [event_states EXCEPT ![e] = "pooled"]
    /\ event_ids' = [event_ids EXCEPT ![e] = 0]
    /\ signal_counters' = [signal_counters EXCEPT ![e] = 0]
    /\ recording_streams' = [recording_streams EXCEPT ![e] = 0]
    /\ ref_counts' = [ref_counts EXCEPT ![e] = 0]
    /\ pool_freelist' = pool_freelist \union {e}
    /\ in_use_events' = in_use_events \ {e}
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<event_counter, pending_callbacks, pool_mutex_holder,
                   event_mutex_holder>>

\* RecordEvent: Record event on a stream
\* Models: MPSEvent::record()
\*
\* Implementation:
\*   1. Lock event mutex
\*   2. Increment signal counter
\*   3. Schedule MTLSharedEvent signal

RecordEvent(t, e, s) ==
    /\ op_count < MaxOperations
    /\ event_mutex_holder[e] = t          \* Must hold event mutex
    /\ event_states[e] \in {"acquired", "signaled"}  \* Can re-record
    /\ event_states' = [event_states EXCEPT ![e] = "recorded"]
    /\ signal_counters' = [signal_counters EXCEPT ![e] = signal_counters[e] + 1]
    /\ recording_streams' = [recording_streams EXCEPT ![e] = s]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<event_counter, event_ids, ref_counts, pool_freelist,
                   in_use_events, pending_callbacks, pool_mutex_holder,
                   event_mutex_holder>>

\* NotifyEvent: Schedule callback for event completion
\* Models: MPSEvent::notify()
\*
\* CALLBACK SURVIVAL: Increment ref_count to keep event alive

NotifyEvent(t, e) ==
    /\ op_count < MaxOperations
    /\ event_mutex_holder[e] = t
    /\ event_states[e] = "recorded"
    /\ event_states' = [event_states EXCEPT ![e] = "pending_callback"]
    /\ ref_counts' = [ref_counts EXCEPT ![e] = ref_counts[e] + 1]  \* Callback ref
    /\ pending_callbacks' = pending_callbacks \union {e}
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<event_counter, event_ids, signal_counters, recording_streams,
                   pool_freelist, in_use_events, pool_mutex_holder,
                   event_mutex_holder>>

\* SignalEvent: GPU signals event completion
\* Models: MTLSharedEvent signaling (async, from GPU)

SignalEvent(e) ==
    /\ op_count < MaxOperations
    /\ event_states[e] = "recorded"
    /\ event_mutex_holder[e] = 0          \* Signal happens async
    /\ event_states' = [event_states EXCEPT ![e] = "signaled"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<event_counter, event_ids, signal_counters, recording_streams,
                   ref_counts, pool_freelist, in_use_events, pending_callbacks,
                   pool_mutex_holder, event_mutex_holder>>

\* CallbackComplete: Callback finishes, decrement ref count
\* Models: notifyListener callback completion

CallbackComplete(e) ==
    /\ op_count < MaxOperations
    /\ e \in pending_callbacks
    /\ event_states[e] \in {"pending_callback", "signaled"}
    /\ ref_counts[e] > 1                  \* Has callback reference
    /\ ref_counts' = [ref_counts EXCEPT ![e] = ref_counts[e] - 1]
    /\ pending_callbacks' = pending_callbacks \ {e}
    /\ event_states' = [event_states EXCEPT ![e] = "signaled"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<event_counter, event_ids, signal_counters, recording_streams,
                   pool_freelist, in_use_events, pool_mutex_holder,
                   event_mutex_holder>>

\* WaitEvent: Wait for event to be signaled (CPU blocks)
\* Models: MPSEvent::wait() or synchronize()

WaitEvent(t, e) ==
    /\ op_count < MaxOperations
    /\ event_mutex_holder[e] = t
    /\ event_states[e] = "signaled"
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<event_states, event_counter, event_ids, signal_counters,
                   recording_streams, ref_counts, pool_freelist, in_use_events,
                   pending_callbacks, pool_mutex_holder, event_mutex_holder>>

-----------------------------------------------------------------------------
\* Next state relation
Next ==
    \/ \E t \in 1..NumThreads :
        \/ AcquirePoolMutex(t)
        \/ ReleasePoolMutex(t)
        \/ \E e \in 1..NumEvents :
            \/ AcquireEventMutex(t, e)
            \/ ReleaseEventMutex(t, e)
            \/ ReleaseEvent(t, e)
            \/ WaitEvent(t, e)
            \/ \E s \in 1..NumStreams : RecordEvent(t, e, s)
            \/ NotifyEvent(t, e)
        \/ AcquireEvent(t)
    \/ \E e \in 1..NumEvents :
        \/ SignalEvent(e)
        \/ CallbackComplete(e)
    \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* SAFETY PROPERTIES

\* Event ID Uniqueness: No two in-use events have same ID
EventIDUniqueness ==
    \A e1, e2 \in 1..NumEvents :
        (e1 /= e2 /\ e1 \in in_use_events /\ e2 \in in_use_events) =>
        event_ids[e1] /= event_ids[e2]

\* Callback Survival: Events with pending callbacks can't be released
\* This is the KEY property preventing use-after-free in callbacks
CallbackSurvival ==
    \A e \in 1..NumEvents :
        e \in pending_callbacks => ref_counts[e] > 1

\* Signal Counter Monotonicity: Counter only increases (per event)
SignalCounterMonotonicity ==
    \A e \in 1..NumEvents :
        [][signal_counters'[e] >= signal_counters[e]]_vars

\* Pool/InUse Partition: Events are in pool XOR in use
PoolInUsePartition ==
    \A e \in 1..NumEvents :
        (e \in pool_freelist) = (e \notin in_use_events)

\* State Consistency: Pooled events have zero refs and IDs
PooledEventConsistency ==
    \A e \in 1..NumEvents :
        event_states[e] = "pooled" =>
        (event_ids[e] = 0 /\ ref_counts[e] = 0 /\ recording_streams[e] = 0)

\* Acquired events have positive ref count
AcquiredEventConsistency ==
    \A e \in 1..NumEvents :
        event_states[e] \in {"acquired", "recorded", "signaled", "pending_callback"} =>
        ref_counts[e] >= 1

\* No use after release
NoUseAfterRelease ==
    \A e \in 1..NumEvents :
        e \in pool_freelist => event_states[e] = "pooled"

\* Combined safety invariant
SafetyInvariant ==
    /\ TypeOK
    /\ EventIDUniqueness
    /\ CallbackSurvival
    /\ PoolInUsePartition
    /\ PooledEventConsistency
    /\ AcquiredEventConsistency
    /\ NoUseAfterRelease

-----------------------------------------------------------------------------
\* LIVENESS PROPERTIES

\* Progress: Pending callbacks eventually complete
CallbackProgress ==
    \A e \in 1..NumEvents :
        (e \in pending_callbacks) ~> (e \notin pending_callbacks)

\* Recorded events eventually signal
SignalProgress ==
    \A e \in 1..NumEvents :
        (event_states[e] = "recorded") ~> (event_states[e] = "signaled")

-----------------------------------------------------------------------------
\* DEADLOCK FREEDOM

DeadlockFree ==
    op_count = MaxOperations \/
    \/ pool_freelist /= {}           \* Can acquire events
    \/ \E e \in in_use_events :
       ref_counts[e] = 1             \* Can release events
    \/ pending_callbacks /= {}        \* Callbacks can complete

=============================================================================
\* Modification History
\* Created: 2025-12-18 by AI Worker N=1251
\* Based on: MPSEvent.h MPSEvent and MPSEventPool classes
