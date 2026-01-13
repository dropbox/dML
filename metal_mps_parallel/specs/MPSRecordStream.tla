--------------------------- MODULE MPSRecordStream ---------------------------
\* TLA+ Specification for MPS recordStream Cross-Stream Lifetime Protocol
\* Based on: pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm
\*
\* This spec models the detailed event-based cross-stream synchronization:
\* - recordStream() creates events for cross-stream buffer usage
\* - pending_events tracks events that must complete before recycling
\* - free_buffer() defers recycling until all events complete
\*
\* Key Properties:
\* RS.NoEarlyReuse: Buffer cannot be recycled until all pending events complete
\* RS.EventAccounting: Events are removed only when query() observes completion
\* RS.NoUnboundedGrowth: pending_events bounded by NumStreams (one per stream)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    NumBuffers,         \* Maximum buffer slots
    NumStreams,         \* Number of MPS streams
    NumThreads,         \* Number of concurrent threads
    MaxOperations       \* Bound for model checking

ASSUME NumBuffers >= 1
ASSUME NumStreams >= 1
ASSUME NumThreads >= 1
ASSUME MaxOperations >= 1

VARIABLES
    \* Buffer states: "free", "allocated", "in_use", "pending_free"
    buffer_states,
    \* Which stream originally allocated each buffer (0 = unallocated)
    alloc_streams,
    \* Set of streams that have used each buffer (for recordStream tracking)
    stream_uses,
    \* Pending events per buffer: subset of streams whose events haven't completed
    \* In real code this is vector<MPSEventPtr>, we model as set of stream IDs
    pending_events,
    \* Event completion status: (buffer, stream) -> completed flag
    \* Models MPSEvent::query() result
    event_completed,
    \* Pool mutex holder (0 = none)
    pool_mutex_holder,
    \* Available buffer set (returned to pool)
    available_buffers,
    \* Buffers waiting for pending events (buffers_pending_free)
    buffers_pending_free,
    \* Operation counter
    op_count

vars == <<buffer_states, alloc_streams, stream_uses, pending_events,
          event_completed, pool_mutex_holder, available_buffers,
          buffers_pending_free, op_count>>

-----------------------------------------------------------------------------
\* Type Invariant
TypeOK ==
    /\ buffer_states \in [1..NumBuffers -> {"free", "allocated", "in_use", "pending_free"}]
    /\ alloc_streams \in [1..NumBuffers -> 0..NumStreams]
    /\ stream_uses \in [1..NumBuffers -> SUBSET (1..NumStreams)]
    /\ pending_events \in [1..NumBuffers -> SUBSET (1..NumStreams)]
    /\ event_completed \in [1..NumBuffers -> [1..NumStreams -> BOOLEAN]]
    /\ pool_mutex_holder \in 0..NumThreads
    /\ available_buffers \subseteq 1..NumBuffers
    /\ buffers_pending_free \subseteq 1..NumBuffers
    /\ op_count \in 0..MaxOperations

\* Initial state
Init ==
    /\ buffer_states = [b \in 1..NumBuffers |-> "free"]
    /\ alloc_streams = [b \in 1..NumBuffers |-> 0]
    /\ stream_uses = [b \in 1..NumBuffers |-> {}]
    /\ pending_events = [b \in 1..NumBuffers |-> {}]
    /\ event_completed = [b \in 1..NumBuffers |-> [s \in 1..NumStreams |-> TRUE]]
    /\ pool_mutex_holder = 0
    /\ available_buffers = 1..NumBuffers
    /\ buffers_pending_free = {}
    /\ op_count = 0

-----------------------------------------------------------------------------
\* Mutex operations

AcquirePoolMutex(t) ==
    /\ pool_mutex_holder = 0
    /\ pool_mutex_holder' = t
    /\ UNCHANGED <<buffer_states, alloc_streams, stream_uses, pending_events,
                   event_completed, available_buffers, buffers_pending_free, op_count>>

ReleasePoolMutex(t) ==
    /\ pool_mutex_holder = t
    /\ pool_mutex_holder' = 0
    /\ UNCHANGED <<buffer_states, alloc_streams, stream_uses, pending_events,
                   event_completed, available_buffers, buffers_pending_free, op_count>>

-----------------------------------------------------------------------------
\* AllocBuffer: Thread t allocates a buffer on stream s
\* Models: MPSHeapAllocatorImpl::malloc() + process_pending_buffers_locked()
AllocBuffer(t, s) ==
    /\ op_count < MaxOperations
    /\ pool_mutex_holder = t
    /\ available_buffers /= {}
    /\ LET b == CHOOSE x \in available_buffers : TRUE
       IN /\ buffer_states' = [buffer_states EXCEPT ![b] = "allocated"]
          /\ alloc_streams' = [alloc_streams EXCEPT ![b] = s]
          /\ stream_uses' = [stream_uses EXCEPT ![b] = {s}]
          /\ pending_events' = [pending_events EXCEPT ![b] = {}]
          /\ event_completed' = [event_completed EXCEPT ![b] = [s2 \in 1..NumStreams |-> TRUE]]
          /\ available_buffers' = available_buffers \ {b}
          /\ op_count' = op_count + 1
    /\ UNCHANGED <<pool_mutex_holder, buffers_pending_free>>

\* UseBuffer: Mark buffer as in-use (GPU work submitted)
UseBuffer(b) ==
    /\ op_count < MaxOperations
    /\ buffer_states[b] = "allocated"
    /\ buffer_states' = [buffer_states EXCEPT ![b] = "in_use"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<alloc_streams, stream_uses, pending_events, event_completed,
                   pool_mutex_holder, available_buffers, buffers_pending_free>>

-----------------------------------------------------------------------------
\* RecordStream: Track cross-stream buffer usage
\* Models: MPSHeapAllocatorImpl::recordStream()
\*
\* Implementation (MPSAllocator.mm:898-924):
\*   1. Get buffer_block from ptr
\*   2. Lock pool.pool_mutex
\*   3. If stream == alloc_stream, return (no cross-stream sync needed)
\*   4. If first time this stream uses buffer (stream_uses.insert().second):
\*      - Create new MPSEvent: event = m_event_pool->acquireEvent(false, stream)
\*      - Record event on stream: event->record(stream, true, false)
\*      - Add to pending_events: buffer_block->pending_events.push_back(event)

RecordStream(t, b, s) ==
    /\ op_count < MaxOperations
    /\ buffer_states[b] \in {"allocated", "in_use"}
    /\ pool_mutex_holder = t                    \* Must hold pool mutex
    /\ s /= alloc_streams[b]                    \* Cross-stream (not allocating stream)
    /\ s \notin stream_uses[b]                  \* First time this stream uses buffer
    /\ stream_uses' = [stream_uses EXCEPT ![b] = stream_uses[b] \union {s}]
    /\ pending_events' = [pending_events EXCEPT ![b] = pending_events[b] \union {s}]
    /\ event_completed' = [event_completed EXCEPT ![b] = [event_completed[b] EXCEPT ![s] = FALSE]]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<buffer_states, alloc_streams, pool_mutex_holder,
                   available_buffers, buffers_pending_free>>

-----------------------------------------------------------------------------
\* EventComplete: GPU signals that stream s has completed for buffer b
\* Models: Metal callback / MPSEvent::query() returning true
\*
\* In reality, the GPU signals MTLSharedEvent completion asynchronously.
\* This action models that completion becoming visible.

EventComplete(b, s) ==
    /\ op_count < MaxOperations
    /\ s \in pending_events[b]                  \* Event exists
    /\ ~event_completed[b][s]                   \* Not yet completed
    /\ event_completed' = [event_completed EXCEPT ![b] = [event_completed[b] EXCEPT ![s] = TRUE]]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<buffer_states, alloc_streams, stream_uses, pending_events,
                   pool_mutex_holder, available_buffers, buffers_pending_free>>

-----------------------------------------------------------------------------
\* FreeBuffer: Thread t frees buffer b
\* Models: MPSHeapAllocatorImpl::free_buffer()
\*
\* Implementation (MPSAllocator.mm:543-579):
\*   1. Check pending_events (iterate through vector)
\*   2. For each event, call query():
\*      - If completed: erase from pending_events
\*      - If not completed: keep in pending_events
\*   3. If pending_events still non-empty after scan:
\*      - Insert buffer into buffers_pending_free
\*      - Return (defer recycling)
\*   4. Otherwise:
\*      - Clear alloc_stream and stream_uses
\*      - Return buffer to available_buffers

FreeBuffer(t, b) ==
    /\ op_count < MaxOperations
    /\ buffer_states[b] \in {"allocated", "in_use"}
    /\ pool_mutex_holder = t
    \* Query all pending events and remove completed ones
    /\ LET completed_events == {s \in pending_events[b] : event_completed[b][s]}
           remaining_events == pending_events[b] \ completed_events
       IN IF remaining_events /= {}
          THEN \* Some events still pending: defer to buffers_pending_free
               /\ buffer_states' = [buffer_states EXCEPT ![b] = "pending_free"]
               /\ pending_events' = [pending_events EXCEPT ![b] = remaining_events]
               /\ buffers_pending_free' = buffers_pending_free \union {b}
               /\ UNCHANGED <<alloc_streams, stream_uses, available_buffers>>
          ELSE \* All events complete: return to pool
               /\ buffer_states' = [buffer_states EXCEPT ![b] = "free"]
               /\ alloc_streams' = [alloc_streams EXCEPT ![b] = 0]
               /\ stream_uses' = [stream_uses EXCEPT ![b] = {}]
               /\ pending_events' = [pending_events EXCEPT ![b] = {}]
               /\ available_buffers' = available_buffers \union {b}
               /\ UNCHANGED buffers_pending_free
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<event_completed, pool_mutex_holder>>

-----------------------------------------------------------------------------
\* ProcessPendingBuffer: Process deferred buffers
\* Models: MPSHeapAllocatorImpl::process_pending_buffers_locked()
\*
\* Implementation (MPSAllocator.mm:777-791):
\*   For each buffer in buffers_pending_free:
\*   - Check retainCount <= 1 (GPU done)
\*   - If done, remove from buffers_pending_free, call free_buffer()
\*
\* We model this as: when all pending_events for a buffer are completed,
\* the buffer can be processed (returned to pool).

ProcessPendingBuffer(t, b) ==
    /\ op_count < MaxOperations
    /\ pool_mutex_holder = t
    /\ b \in buffers_pending_free
    /\ buffer_states[b] = "pending_free"
    \* Query all pending events
    /\ LET completed_events == {s \in pending_events[b] : event_completed[b][s]}
           remaining_events == pending_events[b] \ completed_events
       IN IF remaining_events = {}
          THEN \* All events complete: return to pool
               /\ buffer_states' = [buffer_states EXCEPT ![b] = "free"]
               /\ alloc_streams' = [alloc_streams EXCEPT ![b] = 0]
               /\ stream_uses' = [stream_uses EXCEPT ![b] = {}]
               /\ pending_events' = [pending_events EXCEPT ![b] = {}]
               /\ available_buffers' = available_buffers \union {b}
               /\ buffers_pending_free' = buffers_pending_free \ {b}
          ELSE \* Still waiting: update pending_events (remove completed)
               /\ pending_events' = [pending_events EXCEPT ![b] = remaining_events]
               /\ UNCHANGED <<buffer_states, alloc_streams, stream_uses,
                             available_buffers, buffers_pending_free>>
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<event_completed, pool_mutex_holder>>

-----------------------------------------------------------------------------
\* Next state relation
Next ==
    \/ \E t \in 1..NumThreads :
        \/ AcquirePoolMutex(t)
        \/ ReleasePoolMutex(t)
        \/ \E s \in 1..NumStreams : AllocBuffer(t, s)
        \/ \E b \in 1..NumBuffers, s \in 1..NumStreams : RecordStream(t, b, s)
        \/ \E b \in 1..NumBuffers : FreeBuffer(t, b)
        \/ \E b \in 1..NumBuffers : ProcessPendingBuffer(t, b)
    \/ \E b \in 1..NumBuffers : UseBuffer(b)
    \/ \E b \in 1..NumBuffers, s \in 1..NumStreams : EventComplete(b, s)
    \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* SAFETY PROPERTIES

\* RS.NoEarlyReuse: Buffer cannot be in available_buffers while pending events exist
\* A buffer is only recycled when ALL pending events have completed
RSNoEarlyReuse ==
    \A b \in 1..NumBuffers :
        b \in available_buffers =>
        \A s \in 1..NumStreams : s \notin pending_events[b]

\* RS.EventAccounting: Events in pending_events that are not completed stay in pending_events
\* This checks consistency: completed events should have been removed
RSEventAccountingConsistent ==
    \A b \in 1..NumBuffers :
        buffer_states[b] = "free" => pending_events[b] = {}

\* RS.NoUnboundedGrowth: pending_events size bounded by streams (one event per stream)
RSBoundedPendingEvents ==
    \A b \in 1..NumBuffers :
        Cardinality(pending_events[b]) <= NumStreams

\* Buffer state consistency
BufferStateConsistency ==
    \A b \in 1..NumBuffers :
        /\ (buffer_states[b] = "free") =>
           (b \in available_buffers /\ b \notin buffers_pending_free)
        /\ (buffer_states[b] = "pending_free") =>
           (b \in buffers_pending_free /\ b \notin available_buffers)
        /\ (buffer_states[b] \in {"allocated", "in_use"}) =>
           (b \notin available_buffers /\ b \notin buffers_pending_free)

\* Stream uses tracking: pending_events subset of stream_uses (minus alloc_stream)
StreamUsesConsistency ==
    \A b \in 1..NumBuffers :
        pending_events[b] \subseteq stream_uses[b]

\* Combined safety invariant
SafetyInvariant ==
    /\ TypeOK
    /\ RSNoEarlyReuse
    /\ RSEventAccountingConsistent
    /\ RSBoundedPendingEvents
    /\ BufferStateConsistency
    /\ StreamUsesConsistency

-----------------------------------------------------------------------------
\* LIVENESS PROPERTIES (under fairness)

\* RS.EventualRecycle: Any freed buffer eventually becomes available
\* (requires weak fairness on EventComplete)
EventualRecycle ==
    \A b \in 1..NumBuffers :
        (b \in buffers_pending_free) ~> (buffer_states[b] = "free")

-----------------------------------------------------------------------------
\* DEADLOCK FREEDOM

\* System can always make progress (or has reached operation limit)
DeadlockFree ==
    op_count = MaxOperations \/
    \/ available_buffers /= {}
    \/ buffers_pending_free /= {}
    \/ \E b \in 1..NumBuffers :
       buffer_states[b] \in {"allocated", "in_use"}
    \/ \E b \in 1..NumBuffers, s \in 1..NumStreams :
       (s \in pending_events[b] /\ ~event_completed[b][s])

=============================================================================
\* Modification History
\* Created: 2025-12-19 by AI Worker N=1305
\* Purpose: Model recordStream() cross-stream lifetime protocol
\* Based on: MPSAllocator.mm lines 543-579 (free_buffer), 777-791 (process_pending),
\*           898-924 (recordStream)
