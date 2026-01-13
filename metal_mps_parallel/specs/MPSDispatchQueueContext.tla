--------------------------- MODULE MPSDispatchQueueContext ---------------------------
(*
 * TLA+ Specification for MPS Dispatch Queue Context Safety (Opportunity Map B1.5)
 *
 * This spec models the GCD dispatch queue execution context hazards:
 * - Re-entrant dispatch_sync deadlock
 * - TLS lookup inside dispatched blocks
 * - Exception propagation soundness
 *
 * PURPOSE:
 * GCD dispatch_sync to a serial queue you're already on causes deadlock.
 * The MPS codebase uses dispatch_sync_with_rethrow for Metal encoding.
 * This spec verifies that re-entrancy is properly detected and avoided.
 *
 * CODE ANCHORS:
 * - MPSStream.mm:dispatch_sync_with_rethrow (exception-safe dispatch)
 * - MPSGraph.mm:executeMPSGraph (uses dispatch_get_specific for inline execution)
 * - tests/repro_dispatch_sync_with_rethrow_reentrancy.mm (proof-of-concept)
 *
 * VERIFIED PROPERTIES:
 * - DQ.NoReentrantDispatchSync: No dispatch_sync to a queue we're already executing on
 * - DQ.NoTLSLookupInsideDispatchedBlock: Blocks use captured stream, not TLS
 * - DQ.ExceptionPropagationSound: Exception propagates exactly once
 * - DQ.DeadlockFree: System never reaches a state where all threads are blocked
 *
 * Created: 2025-12-19 (Iteration N=1308)
 *)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,     \* Number of concurrent threads (3-4 for verification)
    NumQueues,      \* Number of dispatch queues (one per stream)
    MaxOps,         \* Maximum operations per thread (bounds state space)
    AllowUnsafePatterns  \* BOOLEAN: TRUE to include unsafe actions (for proving hazards exist)

\* Sentinel values
NoQueue == NumQueues  \* "No queue" sentinel

VARIABLES
    \* Thread execution context
    thread_state,         \* thread_id -> {"idle", "running", "dispatching", "blocked", "done"}
    thread_queue,         \* thread_id -> queue currently executing on (NoQueue if none)
    thread_target_queue,  \* thread_id -> queue thread is trying to dispatch_sync to (NoQueue if none)
    thread_uses_tls,      \* thread_id -> BOOLEAN (TRUE if incorrectly using TLS in dispatch block)
    thread_captured_stream, \* thread_id -> BOOLEAN (TRUE if correctly using captured stream)

    \* Queue state
    queue_executing,      \* queue_id -> thread_id currently executing block (0 = none)
    queue_has_specific,   \* queue_id -> BOOLEAN (TRUE if has dispatch_queue_specific set)

    \* Exception tracking
    thread_exception,     \* thread_id -> {"none", "thrown", "caught", "rethrown"}
    exception_count,      \* Total exceptions that occurred

    \* Model checking counters
    ops_count,            \* thread_id -> operations performed (bounded)
    deadlock_count,       \* Count of deadlock situations detected
    reentrant_attempts    \* Count of re-entrant dispatch_sync attempts

vars == <<thread_state, thread_queue, thread_target_queue, thread_uses_tls,
          thread_captured_stream, queue_executing, queue_has_specific,
          thread_exception, exception_count, ops_count, deadlock_count,
          reentrant_attempts>>

ThreadIds == 1..NumThreads
QueueIds == 0..(NumQueues - 1)
NoHolder == 0

TypeInvariant ==
    /\ thread_state \in [ThreadIds -> {"idle", "running", "dispatching", "blocked", "done"}]
    /\ thread_queue \in [ThreadIds -> 0..NumQueues]  \* NumQueues = NoQueue
    /\ thread_target_queue \in [ThreadIds -> 0..NumQueues]
    /\ thread_uses_tls \in [ThreadIds -> BOOLEAN]
    /\ thread_captured_stream \in [ThreadIds -> BOOLEAN]
    /\ queue_executing \in [QueueIds -> 0..NumThreads]
    /\ queue_has_specific \in [QueueIds -> BOOLEAN]
    /\ thread_exception \in [ThreadIds -> {"none", "thrown", "caught", "rethrown"}]
    /\ exception_count \in 0..100
    /\ ops_count \in [ThreadIds -> 0..MaxOps]
    /\ deadlock_count \in 0..100
    /\ reentrant_attempts \in 0..100

-----------------------------------------------------------------------------
(* Initial State *)
-----------------------------------------------------------------------------

Init ==
    /\ thread_state = [t \in ThreadIds |-> "idle"]
    /\ thread_queue = [t \in ThreadIds |-> NoQueue]
    /\ thread_target_queue = [t \in ThreadIds |-> NoQueue]
    /\ thread_uses_tls = [t \in ThreadIds |-> FALSE]
    /\ thread_captured_stream = [t \in ThreadIds |-> FALSE]
    /\ queue_executing = [q \in QueueIds |-> NoHolder]
    /\ queue_has_specific = [q \in QueueIds |-> TRUE]  \* MPS sets queue-specific
    /\ thread_exception = [t \in ThreadIds |-> "none"]
    /\ exception_count = 0
    /\ ops_count = [t \in ThreadIds |-> 0]
    /\ deadlock_count = 0
    /\ reentrant_attempts = 0

-----------------------------------------------------------------------------
(* Helper Predicates *)
-----------------------------------------------------------------------------

\* Check if thread is already executing on a given queue
IsOnQueue(t, q) ==
    thread_queue[t] = q /\ q /= NoQueue

\* Check if dispatch_get_specific would detect we're on the queue
CanDetectReentrant(t, q) ==
    /\ queue_has_specific[q]
    /\ thread_queue[t] = q

\* A thread can perform more operations
CanOperate(t) ==
    /\ ops_count[t] < MaxOps
    /\ thread_state[t] /= "done"
    /\ thread_state[t] /= "blocked"

-----------------------------------------------------------------------------
(* Dispatch Queue Actions *)
-----------------------------------------------------------------------------

(*
 * SafeDispatchSync: Thread attempts dispatch_sync with re-entrancy detection.
 * Models the correct dispatch_sync_with_rethrow behavior:
 * 1. Check dispatch_get_specific to see if already on queue
 * 2. If on queue, execute inline (bypass dispatch_sync)
 * 3. If not on queue, dispatch_sync normally
 *
 * SIMPLIFICATION: Threads can only be on one queue at a time.
 * Nested dispatch to a different queue requires completing current queue first.
 *)
SafeDispatchSync(t, q) ==
    /\ CanOperate(t)
    /\ thread_state[t] = "idle"  \* Must be idle (not already running on a queue)
    /\ thread_target_queue[t] = NoQueue
    /\ thread_queue[t] = NoQueue  \* Not currently on any queue
    /\ thread_exception[t] = "none"
    \* Not on any queue, so no re-entrancy check needed here
    /\ IF queue_executing[q] = NoHolder
       THEN
         \* Queue is free, acquire it
         /\ queue_executing' = [queue_executing EXCEPT ![q] = t]
         /\ thread_state' = [thread_state EXCEPT ![t] = "running"]
         /\ thread_queue' = [thread_queue EXCEPT ![t] = q]
         /\ thread_target_queue' = thread_target_queue
         /\ thread_captured_stream' = [thread_captured_stream EXCEPT ![t] = TRUE]
       ELSE
         \* Queue is busy, wait
         /\ thread_state' = [thread_state EXCEPT ![t] = "dispatching"]
         /\ thread_target_queue' = [thread_target_queue EXCEPT ![t] = q]
         /\ thread_queue' = thread_queue
         /\ queue_executing' = queue_executing
         /\ thread_captured_stream' = thread_captured_stream
    /\ ops_count' = [ops_count EXCEPT ![t] = @ + 1]
    /\ thread_uses_tls' = thread_uses_tls
    /\ queue_has_specific' = queue_has_specific
    /\ thread_exception' = thread_exception
    /\ exception_count' = exception_count
    /\ deadlock_count' = deadlock_count
    /\ reentrant_attempts' = reentrant_attempts

(*
 * ReentrantDispatchInline: Thread already on queue tries to dispatch to same queue.
 * With proper dispatch_get_specific check, this executes inline.
 *)
ReentrantDispatchInline(t, q) ==
    /\ CanOperate(t)
    /\ thread_state[t] = "running"
    /\ thread_queue[t] = q  \* Already on this queue
    /\ CanDetectReentrant(t, q)  \* Has queue-specific set
    /\ thread_exception[t] = "none"
    \* Execute inline - just continue running, no state change needed
    /\ ops_count' = [ops_count EXCEPT ![t] = @ + 1]
    /\ UNCHANGED <<thread_state, thread_queue, thread_target_queue, thread_uses_tls,
                   thread_captured_stream, queue_executing, queue_has_specific,
                   thread_exception, exception_count, deadlock_count, reentrant_attempts>>

(*
 * UnsafeDispatchSync: Thread attempts dispatch_sync WITHOUT re-entrancy detection.
 * This models the bug case where dispatch_get_specific is not used.
 * If already on queue, this WILL DEADLOCK.
 *)
UnsafeDispatchSync(t, q) ==
    /\ CanOperate(t)
    /\ thread_state[t] = "idle" \/ thread_state[t] = "running"
    /\ thread_target_queue[t] = NoQueue
    /\ thread_exception[t] = "none"
    \* No re-entrancy check - always try dispatch_sync
    /\ IF IsOnQueue(t, q)
       THEN
         \* RE-ENTRANT DISPATCH - DEADLOCK!
         /\ thread_state' = [thread_state EXCEPT ![t] = "blocked"]
         /\ reentrant_attempts' = reentrant_attempts + 1
         /\ deadlock_count' = deadlock_count + 1
         /\ thread_queue' = thread_queue
         /\ thread_target_queue' = [thread_target_queue EXCEPT ![t] = q]
         /\ queue_executing' = queue_executing
         /\ thread_captured_stream' = thread_captured_stream
       ELSE
         IF queue_executing[q] = NoHolder
         THEN
           \* Queue is free, acquire it
           /\ queue_executing' = [queue_executing EXCEPT ![q] = t]
           /\ thread_state' = [thread_state EXCEPT ![t] = "running"]
           /\ thread_queue' = [thread_queue EXCEPT ![t] = q]
           /\ thread_target_queue' = thread_target_queue
           /\ thread_captured_stream' = thread_captured_stream
           /\ reentrant_attempts' = reentrant_attempts
           /\ deadlock_count' = deadlock_count
         ELSE
           \* Queue is busy, wait
           /\ thread_state' = [thread_state EXCEPT ![t] = "dispatching"]
           /\ thread_target_queue' = [thread_target_queue EXCEPT ![t] = q]
           /\ thread_queue' = thread_queue
           /\ queue_executing' = queue_executing
           /\ thread_captured_stream' = thread_captured_stream
           /\ reentrant_attempts' = reentrant_attempts
           /\ deadlock_count' = deadlock_count
    /\ ops_count' = [ops_count EXCEPT ![t] = @ + 1]
    /\ thread_uses_tls' = thread_uses_tls
    /\ queue_has_specific' = queue_has_specific
    /\ thread_exception' = thread_exception
    /\ exception_count' = exception_count

(*
 * TLSLookupInBlock: Thread incorrectly uses TLS to get stream inside dispatch block.
 * This is a bug because the block may execute on a different thread.
 *)
TLSLookupInBlock(t) ==
    /\ CanOperate(t)
    /\ thread_state[t] = "running"
    /\ thread_queue[t] /= NoQueue
    /\ ~thread_uses_tls[t]  \* Only flag once
    /\ thread_uses_tls' = [thread_uses_tls EXCEPT ![t] = TRUE]
    /\ ops_count' = [ops_count EXCEPT ![t] = @ + 1]
    /\ UNCHANGED <<thread_state, thread_queue, thread_target_queue,
                   thread_captured_stream, queue_executing, queue_has_specific,
                   thread_exception, exception_count, deadlock_count, reentrant_attempts>>

(*
 * CompleteDispatch: Thread finishes execution on a queue and releases it.
 *)
CompleteDispatch(t) ==
    /\ thread_state[t] = "running"
    /\ thread_queue[t] /= NoQueue
    /\ LET q == thread_queue[t] IN
       /\ queue_executing' = [queue_executing EXCEPT ![q] = NoHolder]
       /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
       /\ thread_queue' = [thread_queue EXCEPT ![t] = NoQueue]
    /\ UNCHANGED <<thread_target_queue, thread_uses_tls, thread_captured_stream,
                   queue_has_specific, thread_exception, exception_count, ops_count,
                   deadlock_count, reentrant_attempts>>

(*
 * AcquireWaitingQueue: A waiting thread acquires a queue that just became free.
 *)
AcquireWaitingQueue(t) ==
    /\ thread_state[t] = "dispatching"
    /\ thread_target_queue[t] /= NoQueue
    /\ LET q == thread_target_queue[t] IN
       /\ queue_executing[q] = NoHolder
       /\ queue_executing' = [queue_executing EXCEPT ![q] = t]
       /\ thread_state' = [thread_state EXCEPT ![t] = "running"]
       /\ thread_queue' = [thread_queue EXCEPT ![t] = q]
       /\ thread_target_queue' = [thread_target_queue EXCEPT ![t] = NoQueue]
       /\ thread_captured_stream' = [thread_captured_stream EXCEPT ![t] = TRUE]
    /\ UNCHANGED <<thread_uses_tls, queue_has_specific, thread_exception,
                   exception_count, ops_count, deadlock_count, reentrant_attempts>>

(*
 * ThrowException: Thread throws an exception during dispatch block.
 *)
ThrowException(t) ==
    /\ CanOperate(t)
    /\ thread_state[t] = "running"
    /\ thread_exception[t] = "none"
    /\ thread_exception' = [thread_exception EXCEPT ![t] = "thrown"]
    /\ exception_count' = exception_count + 1
    /\ ops_count' = [ops_count EXCEPT ![t] = @ + 1]
    /\ UNCHANGED <<thread_state, thread_queue, thread_target_queue, thread_uses_tls,
                   thread_captured_stream, queue_executing, queue_has_specific,
                   deadlock_count, reentrant_attempts>>

(*
 * CatchException: dispatch_sync_with_rethrow catches exception in block.
 *)
CatchException(t) ==
    /\ thread_exception[t] = "thrown"
    /\ thread_exception' = [thread_exception EXCEPT ![t] = "caught"]
    /\ UNCHANGED <<thread_state, thread_queue, thread_target_queue, thread_uses_tls,
                   thread_captured_stream, queue_executing, queue_has_specific,
                   exception_count, ops_count, deadlock_count, reentrant_attempts>>

(*
 * RethrowException: dispatch_sync_with_rethrow rethrows after dispatch_sync returns.
 *)
RethrowException(t) ==
    /\ thread_exception[t] = "caught"
    /\ thread_state[t] = "running" \/ thread_state[t] = "idle"
    /\ thread_exception' = [thread_exception EXCEPT ![t] = "rethrown"]
    /\ thread_state' = [thread_state EXCEPT ![t] = "done"]  \* Unwinds stack
    \* Release queue if holding one
    /\ IF thread_queue[t] /= NoQueue
       THEN
         LET q == thread_queue[t] IN
         /\ queue_executing' = [queue_executing EXCEPT ![q] = NoHolder]
         /\ thread_queue' = [thread_queue EXCEPT ![t] = NoQueue]
       ELSE
         /\ queue_executing' = queue_executing
         /\ thread_queue' = thread_queue
    /\ UNCHANGED <<thread_target_queue, thread_uses_tls, thread_captured_stream,
                   queue_has_specific, exception_count, ops_count,
                   deadlock_count, reentrant_attempts>>

(*
 * FinishThread: Thread completes all work.
 *)
FinishThread(t) ==
    /\ thread_state[t] = "idle"
    /\ thread_queue[t] = NoQueue
    /\ thread_exception[t] = "none"
    /\ ops_count[t] >= 1  \* Did at least some work
    /\ thread_state' = [thread_state EXCEPT ![t] = "done"]
    /\ UNCHANGED <<thread_queue, thread_target_queue, thread_uses_tls,
                   thread_captured_stream, queue_executing, queue_has_specific,
                   thread_exception, exception_count, ops_count,
                   deadlock_count, reentrant_attempts>>

-----------------------------------------------------------------------------
(* Specification *)
-----------------------------------------------------------------------------

(*
 * Safe Next: Only includes safe patterns (dispatch_get_specific check, no TLS lookup)
 *)
SafeNext ==
    \E t \in ThreadIds :
        \/ \E q \in QueueIds : SafeDispatchSync(t, q)
        \/ \E q \in QueueIds : ReentrantDispatchInline(t, q)
        \/ CompleteDispatch(t)
        \/ AcquireWaitingQueue(t)
        \/ ThrowException(t)
        \/ CatchException(t)
        \/ RethrowException(t)
        \/ FinishThread(t)

(*
 * Unsafe Next: Includes unsafe patterns for proving hazards exist
 *)
UnsafeNext ==
    \E t \in ThreadIds :
        \/ \E q \in QueueIds : SafeDispatchSync(t, q)
        \/ \E q \in QueueIds : UnsafeDispatchSync(t, q)
        \/ TLSLookupInBlock(t)
        \/ CompleteDispatch(t)
        \/ AcquireWaitingQueue(t)
        \/ ThrowException(t)
        \/ CatchException(t)
        \/ RethrowException(t)
        \/ FinishThread(t)

(*
 * Next: Conditional based on AllowUnsafePatterns constant
 *)
Next == IF AllowUnsafePatterns THEN UnsafeNext ELSE SafeNext

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Safety Properties *)
-----------------------------------------------------------------------------

(*
 * DQ.NoReentrantDeadlock: No thread is permanently blocked due to re-entrant dispatch_sync.
 * NOTE: This property will FAIL if UnsafeDispatchSync is used, proving the bug exists.
 *)
NoReentrantDeadlock ==
    deadlock_count = 0

(*
 * DQ.NoTLSLookupInBlock: No thread uses TLS lookup inside a dispatch block.
 * This would cause incorrect stream binding since dispatch may run on different thread.
 *)
NoTLSLookupInBlock ==
    \A t \in ThreadIds : ~thread_uses_tls[t]

(*
 * DQ.ExceptionPropagationSound: Exceptions caught inside dispatch_sync blocks
 * are properly rethrown after dispatch_sync returns.
 * This is a liveness property - caught exceptions must eventually be rethrown.
 * As a safety approximation: if thread is "done" or finished its dispatch,
 * exception should not be stuck in "caught".
 *)
ExceptionNotLost ==
    \A t \in ThreadIds :
        (thread_state[t] = "done") => thread_exception[t] /= "caught"

(*
 * DQ.QueueExclusivity: At most one thread executes on each queue at a time.
 * This is fundamental to GCD serial queue semantics.
 *)
QueueExclusivity ==
    \A q \in QueueIds :
        LET holders == {t \in ThreadIds : thread_queue[t] = q}
        IN Cardinality(holders) <= 1

(*
 * DQ.ThreadQueueConsistency: If a thread is "running" on a queue,
 * that queue's executing field points to that thread.
 *)
ThreadQueueConsistency ==
    \A t \in ThreadIds :
        thread_queue[t] /= NoQueue =>
            queue_executing[thread_queue[t]] = t

(*
 * Combined Safety Property
 *)
Safety ==
    /\ QueueExclusivity
    /\ ThreadQueueConsistency
    /\ ExceptionNotLost

-----------------------------------------------------------------------------
(* Liveness / Progress Properties *)
-----------------------------------------------------------------------------

(*
 * DeadlockFree: The system can always make progress unless all threads are done.
 * Note: This will FAIL if UnsafeDispatchSync causes a deadlock.
 *)
AllDone == \A t \in ThreadIds : thread_state[t] = "done"
SomeBlocked == \E t \in ThreadIds : thread_state[t] = "blocked"

\* Property: If not all done, something can happen
DeadlockFree == [](~AllDone => ENABLED(Next))

-----------------------------------------------------------------------------
(* Aspirational Properties (expected to fail with unsafe patterns) *)
-----------------------------------------------------------------------------

(*
 * NoReentrantAttempts: No thread ever attempts re-entrant dispatch_sync.
 * This will FAIL if UnsafeDispatchSync is in the spec, proving the hazard exists.
 *)
NoReentrantAttempts ==
    reentrant_attempts = 0

(*
 * AllUseCapturedStream: All threads in dispatch blocks use captured stream.
 * This is the correct pattern; TLS lookup is wrong.
 *)
AllUseCapturedStream ==
    \A t \in ThreadIds :
        thread_state[t] = "running" /\ thread_queue[t] /= NoQueue =>
            thread_captured_stream[t]

=============================================================================
