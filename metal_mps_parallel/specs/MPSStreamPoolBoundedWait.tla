--------------------------- MODULE MPSStreamPoolBoundedWait ---------------------------
\* TLA+ Specification for MPSStreamPool with Bounded Wait Verification
\* Extends MPSStreamPool to track wait times and verify bounded wait property
\*
\* Purpose: Verify that threads don't wait indefinitely when acquiring streams.
\* This is a Phase 3 "aspirational property" for tracking progress/liveness.
\*
\* Key additions:
\* 1. wait_count[t] - tracks how many steps thread t has been waiting
\* 2. max_wait_bound - configurable threshold for bounded wait
\* 3. BoundedWaitInvariant - verifies no thread waits beyond threshold
\* 4. WaitHistogram - tracks distribution of wait times

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    NumStreams,         \* Total streams in pool (32 in implementation)
    NumThreads,         \* Number of concurrent threads for model checking
    MaxOperations,      \* Bound for model checking
    MaxWaitBound        \* Maximum allowed wait steps (bounded wait threshold)

ASSUME NumStreams > 1
ASSUME NumThreads >= 1
ASSUME MaxOperations >= 1
ASSUME MaxWaitBound >= 1

VARIABLES
    \* Core stream pool state (from MPSStreamPool.tla)
    streams,            \* Stream states: "free", "bound", "in_use"
    free_mask,          \* Bitmask of free worker slots
    thread_bindings,    \* Thread -> Stream binding
    cas_in_progress,    \* Which thread is in CAS critical section
    op_count,           \* Operation counter for bounded checking

    \* Bounded wait tracking (new)
    wait_count,         \* [1..NumThreads -> Nat] - steps each thread has waited
    want_stream,        \* [1..NumThreads -> Bool] - thread wants a stream
    total_waits,        \* Nat - total wait events (for statistics)
    max_observed_wait   \* Nat - maximum wait time observed

vars == <<streams, free_mask, thread_bindings, cas_in_progress, op_count,
          wait_count, want_stream, total_waits, max_observed_wait>>

core_vars == <<streams, free_mask, thread_bindings, cas_in_progress, op_count>>
wait_vars == <<wait_count, want_stream, total_waits, max_observed_wait>>

-----------------------------------------------------------------------------
\* Helper functions (from MPSStreamPool.tla)
SlotToBit(slot) == slot - 1
BitSet(mask, bit) == (mask \div (2^bit)) % 2 = 1
SetBit(mask, bit) == mask + 2^bit
ClearBit(mask, bit) == mask - 2^bit

FindLowestSetBit(mask) ==
    LET bits == {b \in 0..(NumStreams-2) : BitSet(mask, b)}
    IN CHOOSE b \in bits : \A b2 \in bits : b <= b2

AllWorkerSlotsMask == (2^(NumStreams-1)) - 1

-----------------------------------------------------------------------------
\* Type Invariant (extended)
TypeOK ==
    /\ streams \in [1..NumStreams -> {"free", "bound", "in_use"}]
    /\ free_mask \in 0..AllWorkerSlotsMask
    /\ thread_bindings \in [1..NumThreads -> 0..(NumStreams-1)]
    /\ cas_in_progress \in 0..NumThreads
    /\ op_count \in 0..MaxOperations
    /\ wait_count \in [1..NumThreads -> 0..MaxOperations]
    /\ want_stream \in [1..NumThreads -> BOOLEAN]
    /\ total_waits \in Nat
    /\ max_observed_wait \in 0..MaxOperations

\* Initial state
Init ==
    /\ streams = [s \in 1..NumStreams |-> "free"]
    /\ free_mask = AllWorkerSlotsMask
    /\ thread_bindings = [t \in 1..NumThreads |-> 0]
    /\ cas_in_progress = 0
    /\ op_count = 0
    /\ wait_count = [t \in 1..NumThreads |-> 0]
    /\ want_stream = [t \in 1..NumThreads |-> FALSE]
    /\ total_waits = 0
    /\ max_observed_wait = 0

-----------------------------------------------------------------------------
\* Actions (extended with wait tracking)

\* Thread t expresses desire to acquire a stream
\* This is the "start waiting" event
RequestStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] = 0       \* Thread doesn't have a stream
    /\ want_stream[t] = FALSE       \* Not already waiting
    /\ want_stream' = [want_stream EXCEPT ![t] = TRUE]
    /\ UNCHANGED <<streams, free_mask, thread_bindings, cas_in_progress, op_count,
                   wait_count, total_waits, max_observed_wait>>

\* Thread t successfully acquires a stream
\* Records wait time and resets counter
AcquireStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] = 0
    /\ want_stream[t] = TRUE        \* Thread was waiting
    /\ cas_in_progress = 0
    /\ free_mask > 0
    /\ LET bit == FindLowestSetBit(free_mask)
           slot == bit + 1
           this_wait == wait_count[t]
       IN /\ cas_in_progress' = t
          /\ streams' = [streams EXCEPT ![slot] = "bound"]
          /\ free_mask' = ClearBit(free_mask, bit)
          /\ thread_bindings' = [thread_bindings EXCEPT ![t] = slot]
          /\ op_count' = op_count + 1
          \* Record wait statistics
          /\ want_stream' = [want_stream EXCEPT ![t] = FALSE]
          /\ wait_count' = [wait_count EXCEPT ![t] = 0]
          /\ total_waits' = total_waits + 1
          /\ max_observed_wait' = IF this_wait > max_observed_wait
                                   THEN this_wait
                                   ELSE max_observed_wait

\* Complete CAS for acquire
CompleteAcquire(t) ==
    /\ cas_in_progress = t
    /\ cas_in_progress' = 0
    /\ UNCHANGED <<streams, free_mask, thread_bindings, op_count, wait_vars>>

\* Thread t waits because pool is exhausted (increments wait counter)
WaitForStream(t) ==
    /\ op_count < MaxOperations
    /\ want_stream[t] = TRUE        \* Thread wants a stream
    /\ thread_bindings[t] = 0       \* Doesn't have one
    /\ free_mask = 0                \* Pool exhausted
    /\ cas_in_progress = 0
    /\ wait_count' = [wait_count EXCEPT ![t] = wait_count[t] + 1]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<streams, free_mask, thread_bindings, cas_in_progress,
                   want_stream, total_waits, max_observed_wait>>

\* Thread t releases its stream
ReleaseStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0
    /\ cas_in_progress = 0
    /\ LET slot == thread_bindings[t]
           bit == SlotToBit(slot)
       IN /\ streams' = [streams EXCEPT ![slot] = "free"]
          /\ free_mask' = SetBit(free_mask, bit)
          /\ thread_bindings' = [thread_bindings EXCEPT ![t] = 0]
          /\ cas_in_progress' = 0
          /\ op_count' = op_count + 1
    /\ UNCHANGED wait_vars

\* Thread t uses its bound stream
UseStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0
    /\ cas_in_progress = 0
    /\ LET slot == thread_bindings[t]
       IN /\ streams[slot] = "bound"
          /\ streams' = [streams EXCEPT ![slot] = "in_use"]
          /\ UNCHANGED <<free_mask, thread_bindings, cas_in_progress>>
          /\ op_count' = op_count + 1
    /\ UNCHANGED wait_vars

\* Thread t finishes using stream
FinishUse(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0
    /\ cas_in_progress = 0
    /\ LET slot == thread_bindings[t]
       IN /\ streams[slot] = "in_use"
          /\ streams' = [streams EXCEPT ![slot] = "bound"]
          /\ UNCHANGED <<free_mask, thread_bindings, cas_in_progress>>
          /\ op_count' = op_count + 1
    /\ UNCHANGED wait_vars

\* Cancel waiting (give up)
CancelWait(t) ==
    /\ op_count < MaxOperations
    /\ want_stream[t] = TRUE
    /\ thread_bindings[t] = 0
    /\ want_stream' = [want_stream EXCEPT ![t] = FALSE]
    /\ wait_count' = [wait_count EXCEPT ![t] = 0]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<streams, free_mask, thread_bindings, cas_in_progress,
                   total_waits, max_observed_wait>>

-----------------------------------------------------------------------------
\* Next state relation
Next ==
    \/ \E t \in 1..NumThreads :
        \/ RequestStream(t)
        \/ AcquireStream(t)
        \/ CompleteAcquire(t)
        \/ WaitForStream(t)
        \/ ReleaseStream(t)
        \/ UseStream(t)
        \/ FinishUse(t)
        \/ CancelWait(t)
    \/ UNCHANGED vars

\* Fairness
Fairness ==
    /\ \A t \in 1..NumThreads :
        /\ WF_vars(RequestStream(t))
        /\ WF_vars(AcquireStream(t))
        /\ WF_vars(CompleteAcquire(t))
        /\ WF_vars(WaitForStream(t))
        /\ WF_vars(ReleaseStream(t))
        /\ WF_vars(UseStream(t))
        /\ WF_vars(FinishUse(t))
        /\ WF_vars(CancelWait(t))

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
\* SAFETY PROPERTIES (from MPSStreamPool.tla)

MutualExclusion ==
    \A t1, t2 \in 1..NumThreads :
        (t1 /= t2 /\ thread_bindings[t1] > 0) =>
        thread_bindings[t1] /= thread_bindings[t2]

PoolIntegrity ==
    \A slot \in 2..NumStreams :
        LET bit == SlotToBit(slot)
        IN BitSet(free_mask, bit) =>
           (streams[slot] = "free" /\ \A t \in 1..NumThreads : thread_bindings[t] /= slot)

BoundStreamsHaveOwner ==
    \A slot \in 2..NumStreams :
        streams[slot] \in {"bound", "in_use"} =>
        \E t \in 1..NumThreads : thread_bindings[t] = slot

NoOrphanedBindings ==
    \A t \in 1..NumThreads :
        thread_bindings[t] > 0 =>
        streams[thread_bindings[t]] \in {"bound", "in_use"}

-----------------------------------------------------------------------------
\* BOUNDED WAIT PROPERTIES (NEW - Phase 3 aspirational)

\* Key property: No thread waits more than MaxWaitBound steps
\* This is the BOUNDED WAIT invariant
BoundedWaitInvariant ==
    \A t \in 1..NumThreads : wait_count[t] <= MaxWaitBound

\* Stronger property: If pool has capacity and fairness holds,
\* waiting threads eventually get a stream without exceeding bound
BoundedWaitProgress ==
    \A t \in 1..NumThreads :
        (want_stream[t] = TRUE /\ wait_count[t] < MaxWaitBound) ~>
        (thread_bindings[t] > 0 \/ want_stream[t] = FALSE)

\* Diagnostic: Track maximum observed wait across all runs
\* (Not an invariant - just for reporting)
WaitStatistics ==
    /\ max_observed_wait <= MaxOperations
    /\ total_waits >= 0

-----------------------------------------------------------------------------
\* COMBINED INVARIANTS

SafetyInvariant ==
    /\ TypeOK
    /\ MutualExclusion
    /\ PoolIntegrity
    /\ BoundStreamsHaveOwner
    /\ NoOrphanedBindings

\* Full invariant including bounded wait
FullInvariant ==
    /\ SafetyInvariant
    /\ BoundedWaitInvariant

\* Deadlock freedom (extended)
DeadlockFree ==
    op_count = MaxOperations \/
    cas_in_progress > 0 \/
    \E t \in 1..NumThreads :
        \/ /\ thread_bindings[t] = 0
           /\ (free_mask > 0 \/ want_stream[t] = FALSE)
        \/ thread_bindings[t] > 0
        \/ (want_stream[t] = TRUE /\ free_mask = 0)  \* Can wait

=============================================================================
\* Modification History
\* Created: 2025-12-19 by AI Worker N=1301
\* Purpose: Phase 3 "Bounded Wait" verification for progress/liveness
